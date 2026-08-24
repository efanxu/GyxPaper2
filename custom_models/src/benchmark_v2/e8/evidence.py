from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from .config_audit import audit_internal_config
from .constants import (
    CANONICAL_ROOT, CURRENT_SCOPE_MANIFEST, EXTERNAL_DISPLAY_NAMES, EXTERNAL_MODEL_IDS,
    EXPECTED_BATCH, INTERNAL_DISPLAY_NAMES, INTERNAL_ROOT, INTERNAL_VARIANTS, PROJECT_ROOT,
)
from .io_utils import file_identity, forbidden_source, read_json, read_metrics_csv
from .original26_loader import Original26Model


def metrics_equal(left: dict[int, dict[str, float]], right: dict[int, dict[str, float]]) -> bool:
    return all(
        math.isclose(left[h][metric], right[h][metric], rel_tol=1e-12, abs_tol=1e-12)
        for h in (3, 6, 10) for metric in ("Score", "MAE", "RMSE", "R2")
    )


def _resource_stats(run_dir: Path, status: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "model_summary.json") if (run_dir / "model_summary.json").is_file() else {}
    best_epoch = summary.get("best_epoch")
    log_path = run_dir / "train_log.csv"
    if best_epoch is None and log_path.is_file():
        with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("is_best", "")).lower() == "true":
                    best_epoch = int(row["epoch"])
    training_time = summary.get("train_time_sec")
    if training_time is None and status.get("started_at") and status.get("finished_at"):
        training_time = (datetime.fromisoformat(status["finished_at"]) - datetime.fromisoformat(status["started_at"])).total_seconds()
    inference = summary.get("inference_efficiency") or {}
    return {
        "parameter_count": summary.get("parameter_count", summary.get("total_parameters")),
        "trainable_parameter_count": summary.get("trainable_parameter_count", summary.get("trainable_parameters")),
        "best_epoch": best_epoch, "training_time": training_time,
        "peak_memory": summary.get("peak_memory"),
        "inference_time": inference.get("inference_time_sec_full_test"),
        "throughput": inference.get("inference_windows_per_sec"),
        "checkpoint_size": checkpoint.stat().st_size if checkpoint.is_file() else None,
    }


def _external_entry(entry: dict[str, Any], workbook: Original26Model) -> dict[str, Any]:
    run_dir = (PROJECT_ROOT / entry["output_root"] / entry["run_id"]).resolve()
    errors: list[str] = []
    if forbidden := forbidden_source(run_dir):
        errors.append(f"FORBIDDEN_EVIDENCE_SOURCE:{forbidden}")
    if entry.get("model_id") not in EXTERNAL_MODEL_IDS:
        errors.append("MODEL_NOT_IN_FROZEN_EXTERNAL_WHITELIST")
    if entry.get("run_id") != workbook.run_id:
        errors.append("ORIGINAL26_RUN_ID_MISMATCH")
    for name in ("resolved_config.json", "effective_config.json", "run_status.json", "protocol_check.json", "metrics.csv", "best_checkpoint.pt"):
        if not (run_dir / name).is_file():
            errors.append(f"MISSING_ARTIFACT:{name}")
    resolved = read_json(run_dir / "resolved_config.json") if (run_dir / "resolved_config.json").is_file() else {}
    effective = read_json(run_dir / "effective_config.json") if (run_dir / "effective_config.json").is_file() else {}
    status = read_json(run_dir / "run_status.json") if (run_dir / "run_status.json").is_file() else {}
    protocol = read_json(run_dir / "protocol_check.json") if (run_dir / "protocol_check.json").is_file() else {}
    if status.get("status") != "COMPLETED" or status.get("run_mode") != "formal":
        errors.append("FORMAL_RUN_STATUS_INVALID")
    if protocol.get("status") != "PASS":
        errors.append("PROTOCOL_STATUS_INVALID")
    for key in ("train_batch_size", "val_batch_size", "test_batch_size", "gradient_accumulation_steps", "effective_train_batch_size"):
        if effective.get(key) != EXPECTED_BATCH[key]:
            errors.append(f"BATCH_IDENTITY_MISMATCH:{key}")
    if effective.get("training_batch_profile_id") != EXPECTED_BATCH["profile_id"]:
        errors.append("BATCH_PROFILE_MISMATCH")
    artifact_metrics = None
    if (run_dir / "metrics.csv").is_file():
        try:
            artifact_metrics = read_metrics_csv(run_dir / "metrics.csv")
            if not metrics_equal(artifact_metrics, workbook.metrics):
                errors.append("ORIGINAL26_ARTIFACT_MISMATCH")
        except Exception as exc:
            errors.append(f"INVALID_METRICS:{type(exc).__name__}:{exc}")
    return {
        "evidence_id": f"E8_EXTERNAL_{entry['model_id'].upper()}", "evidence_role": "ORIGINAL26_EXTERNAL_INTERACTION_CONTEXT",
        "model_id": entry["model_id"], "display_name": EXTERNAL_DISPLAY_NAMES[entry["model_id"]], "variant_id": None,
        "source_type": "original26_xlsx_plus_formal_artifact", "source_path": str(run_dir), "run_id": entry["run_id"],
        "run_status": status.get("status"), "metrics_source": {"xlsx": workbook.source_file, "artifact": str(run_dir / "metrics.csv")},
        "metrics_identity": file_identity(run_dir / "metrics.csv"), "checkpoint_path": str(run_dir / "best_checkpoint.pt"),
        "checkpoint_identity": file_identity(run_dir / "best_checkpoint.pt"),
        "protocol_identity": {key: effective.get(key) for key in ("protocol_id", "scope_id", "training_batch_profile_id", "train_batch_size", "val_batch_size", "test_batch_size")},
        "resolved_config_identity": file_identity(run_dir / "resolved_config.json"),
        "effective_config_identity": file_identity(run_dir / "effective_config.json"),
        "workbook_metrics": workbook.metrics, "artifact_metrics": artifact_metrics,
        "resource_stats": _resource_stats(run_dir, status, run_dir / "best_checkpoint.pt"),
        "validation_status": "PASS" if not errors else "FAIL", "validation_errors": errors, "ready": not errors,
        "resolved_config_model_id": resolved.get("model_id"),
    }


def _internal_entry(variant: str, root=INTERNAL_ROOT, canonical=CANONICAL_ROOT) -> dict[str, Any]:
    manifest = read_json(root / "experiment_manifest.json")
    definition = next(row for row in manifest["variants"] if row["variant_id"] == variant)
    if variant == "A0":
        evidence_dir, run_dir = root / "A0", canonical
        source_type = "canonical_reference"
    else:
        evidence_dir, run_dir = root / variant, root / variant / "STMGPrompt_ComponentAblation"
        source_type = "formal_component_ablation_artifact"
    effective_path = evidence_dir / "effective_config.json"
    resolved_path = run_dir / "resolved_config.json"
    metrics_path, checkpoint = run_dir / "metrics.csv", run_dir / "best_checkpoint.pt"
    status_path, protocol_path = run_dir / "run_status.json", run_dir / "protocol_check.json"
    errors: list[str] = []
    if forbidden := forbidden_source(run_dir):
        errors.append(f"FORBIDDEN_EVIDENCE_SOURCE:{forbidden}")
    for path, code in (
        (effective_path, "MISSING_EFFECTIVE_CONFIG"), (resolved_path, "MISSING_RESOLVED_CONFIG"),
        (metrics_path, "MISSING_METRICS"), (checkpoint, "MISSING_CHECKPOINT"),
        (protocol_path, "MISSING_PROTOCOL_CHECK"), (status_path, "MISSING_RUN_STATUS"),
    ):
        if not path.is_file():
            errors.append(code)
    effective = read_json(effective_path) if effective_path.is_file() else {}
    status = read_json(status_path) if status_path.is_file() else {}
    protocol = read_json(protocol_path) if protocol_path.is_file() else {}
    status_value = status.get("status")
    if status_value not in {"COMPLETED", "PROCESS_FINISHED"}:
        errors.append("FORMAL_RUN_STATUS_INVALID")
    if protocol and not bool(protocol.get("passed", protocol.get("status") == "PASS")):
        errors.append("PROTOCOL_STATUS_INVALID")
    for key in ("train_batch_size", "val_batch_size", "test_batch_size"):
        if effective.get(key) != EXPECTED_BATCH[key]:
            errors.append(f"INTERNAL_BATCH_PROFILE_MISMATCH:{key}")
    if effective.get("training_batch_profile_id") != EXPECTED_BATCH["profile_id"]:
        errors.append("INTERNAL_BATCH_PROFILE_MISMATCH:training_batch_profile_id")
    metrics = None
    if metrics_path.is_file():
        try:
            metrics = read_metrics_csv(metrics_path)
        except Exception as exc:
            errors.append(f"INVALID_METRICS:{type(exc).__name__}:{exc}")
    return {
        "evidence_id": f"E8_INTERNAL_{variant}", "evidence_role": "INTERNAL_FULL_REFERENCE" if variant == "A0" else "INTERNAL_CAUSAL_ABLATION",
        "model_id": "st_mgprompt", "display_name": INTERNAL_DISPLAY_NAMES[variant], "variant_id": variant,
        "source_type": source_type, "source_path": str(run_dir.resolve()),
        "run_id": definition.get("canonical_reference") if variant == "A0" else f"component_ablation_fixed_dual_seed2026/{variant}",
        "run_status": status_value, "metrics_source": str(metrics_path.resolve()), "metrics_identity": file_identity(metrics_path),
        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_identity": file_identity(checkpoint),
        "protocol_identity": {key: effective.get(key) for key in ("protocol_version", "training_batch_profile_id", "train_batch_size", "val_batch_size", "test_batch_size")},
        "resolved_config_identity": file_identity(resolved_path), "effective_config_identity": file_identity(effective_path),
        "metrics": metrics, "resource_stats": _resource_stats(run_dir, status, checkpoint),
        "validation_status": "PASS" if not errors else "FAIL", "validation_errors": errors, "ready": not errors,
    }


def build_evidence_manifest(workbook_models: dict[str, Original26Model]) -> dict[str, Any]:
    scope = read_json(CURRENT_SCOPE_MANIFEST)
    entries = {row["model_id"]: row for row in scope["entries"]}
    if set(EXTERNAL_MODEL_IDS) - set(entries):
        raise ValueError("Current scope manifest lacks a frozen E8 external model.")
    internal = [_internal_entry(variant) for variant in INTERNAL_VARIANTS]
    external = [_external_entry(entries[model_id], workbook_models[model_id]) for model_id in EXTERNAL_MODEL_IDS]
    return {
        "schema_version": "e8_evidence_manifest_v1", "scope_id": "e8_prompt_cross_fusion_seed2026",
        "internal_config_audit_status": audit_internal_config()["status"], "evidence": internal + external,
        "evidence_count": 11, "expected_external_model_ids": list(EXTERNAL_MODEL_IDS),
        "unexpected_external_models": sorted({row["model_id"] for row in external} - set(EXTERNAL_MODEL_IDS)),
        "e5_consumed": False, "a8_consumed": False, "excluded_models_consumed": False,
    }


def build_missing_workbook_manifest(error_code: str = "MISSING_ORIGINAL26_XLSX") -> dict[str, Any]:
    internal = [_internal_entry(variant) for variant in INTERNAL_VARIANTS]
    external = [{
        "evidence_id": f"E8_EXTERNAL_{model_id.upper()}", "evidence_role": "ORIGINAL26_EXTERNAL_INTERACTION_CONTEXT",
        "model_id": model_id, "display_name": EXTERNAL_DISPLAY_NAMES[model_id], "variant_id": None,
        "source_type": "missing_original26_xlsx", "source_path": None, "run_id": None, "run_status": None,
        "metrics_source": None, "metrics_identity": None, "checkpoint_path": None, "checkpoint_identity": None,
        "protocol_identity": None, "resolved_config_identity": None, "effective_config_identity": None,
        "validation_status": "FAIL", "validation_errors": [error_code], "ready": False,
    } for model_id in EXTERNAL_MODEL_IDS]
    return {
        "schema_version": "e8_evidence_manifest_v1", "scope_id": "e8_prompt_cross_fusion_seed2026",
        "internal_config_audit_status": audit_internal_config()["status"], "evidence": internal + external,
        "evidence_count": 11, "expected_external_model_ids": list(EXTERNAL_MODEL_IDS),
        "unexpected_external_models": [], "e5_consumed": False, "a8_consumed": False, "excluded_models_consumed": False,
    }
