from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from benchmark_v2.e7.original26_loader import _XlsxReader, _header_table

from .constants import (
    CONTROL_LOSS_ID, CONTROL_ROOT, HORIZONS, METRICS, MODEL_IDS, PROJECT_ROOT,
    REQUIRED_TRAIN_ARTIFACTS, SCOPE_MANIFEST, TRAINING_PROFILE_ID,
)
from .io_utils import file_identity, forbidden_source, read_json, read_metrics
from .variants import CONTROL_RUN_IDS

MAIN_SHEET = "原始指标"
DETAIL_SHEET = "模型明细"
REQUIRED_MAIN_COLUMNS = {
    "实验组", "模型", "model_id", "Horizon", "MAE", "RMSE", "R2", "Score",
    "metric_status", "run_status", "protocol_status", "source_file",
}
REQUIRED_DETAIL_COLUMNS = {
    "实验组", "模型", "model_id", "运行状态", "协议检查",
    "H3 MAE", "H3 RMSE", "H3 R²", "H3 Score",
    "H6 MAE", "H6 RMSE", "H6 R²", "H6 Score",
    "H10 MAE", "H10 RMSE", "H10 R²", "H10 Score",
}


def discover_original26(project_root: str | Path = PROJECT_ROOT) -> tuple[Path | None, list[str]]:
    matches: list[Path] = []
    for root, dirs, files in os.walk(Path(project_root), onerror=lambda _: None):
        dirs[:] = [
            name for name in dirs
            if not name.startswith(".") and name.lower() not in {"results_smoke", "smoke"}
            and "e5" not in name.lower() and "common_loss" not in name.lower()
        ]
        for name in files:
            if name.lower().startswith("original26") and name.lower().endswith(".xlsx"):
                candidate = Path(root) / name
                if forbidden_source(candidate) is None:
                    matches.append(candidate.resolve())
    matches.sort()
    if len(matches) == 1:
        return matches[0], []
    if not matches:
        return None, ["MISSING_ORIGINAL26_XLSX"]
    return None, ["AMBIGUOUS_ORIGINAL26_XLSX", *[str(item) for item in matches]]


def resolve_original26(value: str | Path | None) -> tuple[Path | None, list[str]]:
    if value:
        path = Path(value)
        if not path.is_file():
            return None, ["MISSING_ORIGINAL26_XLSX"]
        if forbidden_source(path):
            return None, ["FORBIDDEN_ORIGINAL26_SOURCE"]
        return path.resolve(), []
    return discover_original26()


def load_workbook_controls(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    reader = _XlsxReader(path)
    try:
        main_headers, rows = _header_table(reader.rows(MAIN_SHEET), REQUIRED_MAIN_COLUMNS)
        detail_headers, details = _header_table(reader.rows(DETAIL_SHEET), REQUIRED_DETAIL_COLUMNS)
        sheet_names = list(reader.sheets)
    finally:
        reader.close()
    selected = [row for row in rows if str(row.get("model_id", "")).lower() in MODEL_IDS]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(str(row["model_id"]).lower(), []).append(row)
    errors: list[str] = []
    controls: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_IDS:
        model_rows = grouped.get(model_id, [])
        horizons = [int(row["Horizon"]) for row in model_rows]
        if sorted(horizons) != list(HORIZONS) or len(set(horizons)) != len(horizons):
            errors.append(f"{model_id}:INVALID_HORIZONS")
            continue
        run_ids = {str(row["模型"]) for row in model_rows}
        groups = {str(row["实验组"]) for row in model_rows}
        sources = {str(row["source_file"]) for row in model_rows}
        if run_ids != {CONTROL_RUN_IDS[model_id]} or len(groups) != 1 or len(sources) != 1:
            errors.append(f"{model_id}:NON_UNIQUE_OR_UNEXPECTED_IDENTITY")
            continue
        if {str(row["run_status"]).upper() for row in model_rows} != {"COMPLETED"}:
            errors.append(f"{model_id}:RUN_STATUS_INVALID")
        if {str(row["protocol_status"]).upper() for row in model_rows} != {"PASS"}:
            errors.append(f"{model_id}:PROTOCOL_STATUS_INVALID")
        if {str(row["metric_status"]).upper() for row in model_rows} != {"OK"}:
            errors.append(f"{model_id}:METRIC_STATUS_INVALID")
        controls[model_id] = {
            "model_id": model_id,
            "run_id": next(iter(run_ids)),
            "group_id": next(iter(groups)),
            "source_file": next(iter(sources)),
            "metrics": {
                int(row["Horizon"]): {metric: float(row[metric]) for metric in METRICS}
                for row in model_rows
            },
        }
    detail_ids = [str(row.get("model_id", "")).lower() for row in details if row.get("model_id")]
    for model_id in MODEL_IDS:
        if detail_ids.count(model_id) != 1:
            errors.append(f"{model_id}:MODEL_DETAIL_NOT_UNIQUE")
            continue
        detail = next(row for row in details if str(row.get("model_id", "")).lower() == model_id)
        if str(detail.get("运行状态", "")).upper() != "COMPLETED" or str(detail.get("协议检查", "")).upper() != "PASS":
            errors.append(f"{model_id}:MODEL_DETAIL_STATUS_INVALID")
        if model_id in controls:
            for horizon in HORIZONS:
                for metric, label in (("MAE", "MAE"), ("RMSE", "RMSE"), ("R2", "R²"), ("Score", "Score")):
                    if not math.isclose(float(detail[f"H{horizon} {label}"]), controls[model_id]["metrics"][horizon][metric], rel_tol=1e-12, abs_tol=1e-12):
                        errors.append(f"{model_id}:DETAIL_METRIC_MISMATCH:H{horizon}:{metric}")
    if errors:
        raise ValueError(";".join(errors))
    return controls, {
        "path": str(Path(path).resolve()), "schema_valid": True, "sheets": sheet_names,
        "main_sheet": MAIN_SHEET, "detail_sheet": DETAIL_SHEET,
        "main_columns": main_headers, "detail_columns": detail_headers,
        "selected_model_ids": list(controls), "selected_row_count": len(selected),
    }


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "schema_version": payload.get("schema_version"),
            "resolved_config_hash": payload.get("resolved_config_hash"),
            "effective_config_hash": payload.get("effective_config_hash"),
            "protocol_hash": payload.get("protocol_hash"),
            "initial_model_state_hash": payload.get("initial_model_state_hash"),
            "initialization_seed": payload.get("initialization_seed"),
            "rng_state_present": bool(payload.get("seed_state")),
        }
    except Exception as exc:
        return {"metadata_error": f"{type(exc).__name__}:{exc}"}


def control_identity_errors(
    model_id: str, *, effective: dict[str, Any], resolved: dict[str, Any],
    status: dict[str, Any], protocol: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if status.get("status") != "COMPLETED" or status.get("run_mode") != "formal":
        errors.append("RUN_STATUS_INVALID")
    if protocol.get("status") != "PASS":
        errors.append("PROTOCOL_STATUS_INVALID")
    if effective.get("model_id") != model_id or resolved.get("model_id") != model_id:
        errors.append("MODEL_ID_ARTIFACT_MISMATCH")
    if {effective.get("loss"), resolved.get("loss")} != {CONTROL_LOSS_ID}:
        errors.append("INVALID_ORIGINAL_CONTROL_LOSS")
    if effective.get("training_batch_profile_id") != TRAINING_PROFILE_ID:
        errors.append("BATCH_PROFILE_INVALID")
    for key in ("train_batch_size", "val_batch_size", "test_batch_size"):
        if effective.get(key) != 4:
            errors.append(f"BATCH_INVALID:{key}")
    return errors


def audit_control_artifacts(workbook_controls: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scope = read_json(SCOPE_MANIFEST)
    entries = {entry["model_id"]: entry for entry in scope["entries"]}
    audit_rows = []
    for model_id in MODEL_IDS:
        workbook = workbook_controls[model_id]
        errors: list[str] = []
        entry = entries.get(model_id)
        if entry is None:
            errors.append("MISSING_SCOPE_ENTRY")
            run_dir = CONTROL_ROOT / "__missing__"
        else:
            if entry.get("run_id") != CONTROL_RUN_IDS[model_id]:
                errors.append("SCOPE_RUN_ID_MISMATCH")
            run_dir = (PROJECT_ROOT / entry["output_root"] / entry["run_id"]).resolve()
        if forbidden_source(run_dir):
            errors.append("FORBIDDEN_CONTROL_SOURCE")
        missing = [name for name in REQUIRED_TRAIN_ARTIFACTS if not (run_dir / name).is_file()]
        errors.extend(f"MISSING_ARTIFACT:{name}" for name in missing)
        effective = read_json(run_dir / "effective_config.json") if (run_dir / "effective_config.json").is_file() else {}
        resolved = read_json(run_dir / "resolved_config.json") if (run_dir / "resolved_config.json").is_file() else {}
        status = read_json(run_dir / "run_status.json") if (run_dir / "run_status.json").is_file() else {}
        protocol = read_json(run_dir / "protocol_check.json") if (run_dir / "protocol_check.json").is_file() else {}
        receipt = read_json(run_dir / "execution_receipt.json") if (run_dir / "execution_receipt.json").is_file() else {}
        checkpoint_meta = _checkpoint_metadata(run_dir / "best_checkpoint.pt") if (run_dir / "best_checkpoint.pt").is_file() else {}
        errors.extend(control_identity_errors(
            model_id, effective=effective, resolved=resolved, status=status, protocol=protocol,
        ))
        artifact_metrics = None
        if (run_dir / "metrics.csv").is_file():
            try:
                artifact_metrics = read_metrics(run_dir / "metrics.csv")
                for horizon in HORIZONS:
                    for metric in METRICS:
                        if not math.isclose(artifact_metrics[horizon][metric], workbook["metrics"][horizon][metric], rel_tol=1e-12, abs_tol=1e-12):
                            errors.append(f"WORKBOOK_ARTIFACT_METRIC_MISMATCH:H{horizon}:{metric}")
            except Exception as exc:
                errors.append(f"METRICS_INVALID:{type(exc).__name__}:{exc}")
        audit_rows.append({
            "model_id": model_id, "run_id": workbook["run_id"], "source_root": str(run_dir),
            "run_status": status.get("status"), "protocol_status": protocol.get("status"),
            "protocol_identity": effective.get("protocol_id") or effective.get("protocol_hash"),
            "batch": effective.get("train_batch_size"), "seed": effective.get("seed"),
            "loss_id": effective.get("loss"), "loss_source": "benchmark_v2.losses.masked_mse",
            "loss_profile": effective.get("loss_space"), "loss_hash": None,
            "resolved_config_hash": checkpoint_meta.get("resolved_config_hash"),
            "effective_config_hash": checkpoint_meta.get("effective_config_hash"),
            "checkpoint_hash": receipt.get("checkpoint_sha256"),
            "metrics_hash": receipt.get("metrics_bundle_hash"),
            "model_config_hash": receipt.get("model_config_hash") or effective.get("model_config_hash"),
            "source_closure_hash": receipt.get("source_closure_hash") or effective.get("source_closure_hash"),
            "artifact_completeness": not missing,
            "checkpoint_metadata": checkpoint_meta,
            "metrics": artifact_metrics,
            "resolved_config": resolved,
            "effective_config": effective,
            "validation_status": "PASS" if not errors else "FAIL",
            "validation_errors": errors,
            "ready": not errors,
        })
    ready = sum(bool(row["ready"]) for row in audit_rows)
    return {
        "schema_version": "e9_original26_control_audit_v1",
        "source_class": "ORIGINAL26_CONTROL",
        "control_count": len(audit_rows),
        "ready_count": ready,
        "ORIGINAL_MASKED_MSE_REFERENCE_READY": f"{ready}/6",
        "controls": audit_rows,
        "e5_consumed": False,
    }
