from __future__ import annotations

import math
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from .config_audit import audit_internal_config
from .constants import (
    CANONICAL_ROOT,
    CURRENT_SCOPE_MANIFEST,
    EXTERNAL_DISPLAY_NAMES,
    EXTERNAL_MODEL_IDS,
    EXPECTED_BATCH,
    EXPECTED_GRAPH,
    INTERNAL_DISPLAY_NAMES,
    INTERNAL_ROOT,
    INTERNAL_VARIANTS,
    PROJECT_ROOT,
)
from .io_utils import forbidden_source, read_json, read_metrics_csv
from .original26_loader import Original26Model


def _identity(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _metrics_equal(left: dict[int, dict[str, float]], right: dict[int, dict[str, float]]) -> bool:
    for horizon in (3, 6, 10):
        for metric in ("Score", "MAE", "RMSE", "R2"):
            if not math.isclose(left[horizon][metric], right[horizon][metric], rel_tol=1e-12, abs_tol=1e-12):
                return False
    return True


def _resource_stats(run_dir: Path, status: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    summary_path = run_dir / "model_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    best_epoch = None
    log_path = run_dir / "train_log.csv"
    if log_path.is_file():
        with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("is_best", "")).lower() == "true":
                    best_epoch = int(row["epoch"])
    training_time_seconds = None
    if status.get("started_at") and status.get("finished_at"):
        training_time_seconds = (
            datetime.fromisoformat(status["finished_at"]) - datetime.fromisoformat(status["started_at"])
        ).total_seconds()
    return {
        "parameter_count": summary.get("parameter_count"),
        "trainable_parameter_count": summary.get("trainable_parameter_count"),
        "best_epoch": best_epoch,
        "training_time_seconds": training_time_seconds,
        "peak_memory": None,
        "inference_time": None,
        "throughput": None,
        "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
    }


def _external_entry(entry: dict[str, Any], workbook: Original26Model) -> dict[str, Any]:
    run_dir = (PROJECT_ROOT / entry["output_root"] / entry["run_id"]).resolve()
    errors: list[str] = []
    forbidden = forbidden_source(run_dir)
    if forbidden:
        errors.append(f"FORBIDDEN_EVIDENCE_SOURCE:{forbidden}")
    if entry.get("model_id") not in EXTERNAL_MODEL_IDS:
        errors.append("MODEL_NOT_IN_FROZEN_EXTERNAL_WHITELIST")
    if entry.get("run_id") != workbook.run_id:
        errors.append("ORIGINAL26_RUN_ID_MISMATCH")
    required = (
        "resolved_config.json",
        "effective_config.json",
        "run_status.json",
        "protocol_check.json",
        "metrics.csv",
        "best_checkpoint.pt",
    )
    for name in required:
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
            if not _metrics_equal(artifact_metrics, workbook.metrics):
                errors.append("ORIGINAL26_ARTIFACT_MISMATCH")
        except Exception as exc:
            errors.append(f"INVALID_METRICS:{type(exc).__name__}:{exc}")
    return {
        "evidence_id": f"E7_EXTERNAL_{entry['model_id'].upper()}",
        "evidence_role": "ORIGINAL26_EXTERNAL_CONTEXT",
        "model_id": entry["model_id"],
        "display_name": EXTERNAL_DISPLAY_NAMES[entry["model_id"]],
        "variant_id": None,
        "source_type": "original26_xlsx_plus_formal_artifact",
        "source_path": str(run_dir),
        "run_id": entry["run_id"],
        "run_status": status.get("status"),
        "metrics_source": {"xlsx": workbook.source_file, "artifact": str(run_dir / "metrics.csv")},
        "metrics_identity": _identity(run_dir / "metrics.csv"),
        "checkpoint_path": str(run_dir / "best_checkpoint.pt"),
        "checkpoint_identity": _identity(run_dir / "best_checkpoint.pt"),
        "protocol_identity": {
            "protocol_id": effective.get("protocol_id"),
            "scope_id": effective.get("scope_id"),
            "training_batch_profile_id": effective.get("training_batch_profile_id"),
            "train_batch_size": effective.get("train_batch_size"),
            "val_batch_size": effective.get("val_batch_size"),
            "test_batch_size": effective.get("test_batch_size"),
        },
        "graph_protocol_identity": {
            "graph_id": effective.get("graph_id"),
            "selected_k": effective.get("selected_k"),
        },
        "node_order_identity": effective.get("ordered_node_ids"),
        "resolved_config_identity": _identity(run_dir / "resolved_config.json"),
        "effective_config_identity": _identity(run_dir / "effective_config.json"),
        "workbook_metrics": workbook.metrics,
        "artifact_metrics": artifact_metrics,
        "resource_stats": _resource_stats(run_dir, status, run_dir / "best_checkpoint.pt"),
        "validation_status": "PASS" if not errors else "FAIL",
        "validation_errors": errors,
        "ready": not errors,
        "resolved_config_model_id": resolved.get("model_id"),
    }


def _internal_entry(variant: str) -> dict[str, Any]:
    manifest = read_json(INTERNAL_ROOT / "experiment_manifest.json")
    definition = next(row for row in manifest["variants"] if row["variant_id"] == variant)
    if variant == "A0":
        evidence_dir = INTERNAL_ROOT / "A0"
        run_dir = CANONICAL_ROOT
        status_value = read_json(run_dir / "run_status.json").get("status") if (run_dir / "run_status.json").is_file() else None
        checkpoint = run_dir / "best_checkpoint.pt"
        metrics_path = run_dir / "metrics.csv"
        effective_path = evidence_dir / "effective_config.json"
        resolved_path = run_dir / "resolved_config.json"
        source_type = "canonical_reference"
        existing_identity = read_json(evidence_dir / "reference.json") if (evidence_dir / "reference.json").is_file() else {}
    else:
        evidence_dir = INTERNAL_ROOT / variant
        run_dir = evidence_dir / "STMGPrompt_ComponentAblation"
        status_value = read_json(run_dir / "run_status.json").get("status") if (run_dir / "run_status.json").is_file() else None
        checkpoint = run_dir / "best_checkpoint.pt"
        metrics_path = run_dir / "metrics.csv"
        effective_path = evidence_dir / "effective_config.json"
        resolved_path = run_dir / "resolved_config.json"
        source_type = "formal_component_ablation_artifact"
        existing_identity = {}
    errors: list[str] = []
    forbidden = forbidden_source(run_dir)
    if forbidden:
        errors.append(f"FORBIDDEN_EVIDENCE_SOURCE:{forbidden}")
    for path, code in (
        (effective_path, "MISSING_EFFECTIVE_CONFIG"),
        (resolved_path, "MISSING_RESOLVED_CONFIG"),
        (metrics_path, "MISSING_METRICS"),
        (checkpoint, "MISSING_CHECKPOINT"),
        (run_dir / "protocol_check.json", "MISSING_PROTOCOL_CHECK"),
        (run_dir / "run_status.json", "MISSING_RUN_STATUS"),
    ):
        if not path.is_file():
            errors.append(code)
    effective = read_json(effective_path) if effective_path.is_file() else {}
    protocol = read_json(run_dir / "protocol_check.json") if (run_dir / "protocol_check.json").is_file() else {}
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
        "evidence_id": f"E7_INTERNAL_{variant}",
        "evidence_role": "INTERNAL_CAUSAL_ABLATION" if variant != "A0" else "INTERNAL_FULL_REFERENCE",
        "model_id": "st_mgprompt",
        "display_name": INTERNAL_DISPLAY_NAMES[variant],
        "variant_id": variant,
        "source_type": source_type,
        "source_path": str(run_dir.resolve()),
        "run_id": definition.get("canonical_reference") if variant == "A0" else f"component_ablation_fixed_dual_seed2026/{variant}",
        "run_status": status_value,
        "metrics_source": str(metrics_path.resolve()),
        "metrics_identity": _identity(metrics_path),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_identity": existing_identity.get("checkpoint_sha256") or _identity(checkpoint),
        "protocol_identity": {
            "protocol_version": effective.get("protocol_version"),
            "training_batch_profile_id": effective.get("training_batch_profile_id"),
            "train_batch_size": effective.get("train_batch_size"),
            "val_batch_size": effective.get("val_batch_size"),
            "test_batch_size": effective.get("test_batch_size"),
        },
        "graph_protocol_identity": {
            "benchmark_graph_id_expected": EXPECTED_GRAPH["graph_id"],
            "st_mgprompt_graph_provider": effective.get("graph_tag"),
            "macro_graph_source": effective.get("macro_graph_source"),
            "micro_graph_source": effective.get("micro_graph_source"),
        },
        "node_order_identity": {"field": effective.get("turbine_id_col"), "node_count": effective.get("num_nodes")},
        "resolved_config_identity": _identity(resolved_path),
        "effective_config_identity": _identity(effective_path),
        "metrics": metrics,
        "resource_stats": _resource_stats(run_dir, read_json(run_dir / "run_status.json") if (run_dir / "run_status.json").is_file() else {}, checkpoint),
        "validation_status": "PASS" if not errors else "FAIL",
        "validation_errors": errors,
        "ready": not errors,
    }


def build_evidence_manifest(workbook_models: dict[str, Original26Model]) -> dict[str, Any]:
    scope = read_json(CURRENT_SCOPE_MANIFEST)
    entries = {row["model_id"]: row for row in scope["entries"]}
    if set(EXTERNAL_MODEL_IDS) - set(entries):
        raise ValueError("Current scope manifest lacks a frozen E7 external model.")
    internal = [_internal_entry(variant) for variant in INTERNAL_VARIANTS]
    external = [_external_entry(entries[model_id], workbook_models[model_id]) for model_id in EXTERNAL_MODEL_IDS]
    all_evidence = internal + external
    unexpected = sorted({row["model_id"] for row in external} - set(EXTERNAL_MODEL_IDS))
    return {
        "schema_version": "e7_evidence_manifest_v1",
        "scope_id": "e7_graph_mechanism_seed2026",
        "internal_config_audit_status": audit_internal_config()["status"],
        "evidence": all_evidence,
        "evidence_count": len(all_evidence),
        "unexpected_external_models": unexpected,
        "e5_consumed": False,
        "a8_consumed": False,
        "excluded_models_consumed": False,
    }


def build_missing_workbook_manifest(error_code: str = "MISSING_ORIGINAL26_XLSX") -> dict[str, Any]:
    internal = [_internal_entry(variant) for variant in INTERNAL_VARIANTS]
    external = [
        {
            "evidence_id": f"E7_EXTERNAL_{model_id.upper()}",
            "evidence_role": "ORIGINAL26_EXTERNAL_CONTEXT",
            "model_id": model_id,
            "display_name": EXTERNAL_DISPLAY_NAMES[model_id],
            "variant_id": None,
            "source_type": "missing_original26_xlsx",
            "source_path": None,
            "run_id": None,
            "run_status": None,
            "metrics_source": None,
            "metrics_identity": None,
            "checkpoint_path": None,
            "checkpoint_identity": None,
            "protocol_identity": None,
            "graph_protocol_identity": None,
            "node_order_identity": None,
            "resolved_config_identity": None,
            "effective_config_identity": None,
            "validation_status": "FAIL",
            "validation_errors": [error_code],
            "ready": False,
        }
        for model_id in EXTERNAL_MODEL_IDS
    ]
    return {
        "schema_version": "e7_evidence_manifest_v1",
        "scope_id": "e7_graph_mechanism_seed2026",
        "internal_config_audit_status": audit_internal_config()["status"],
        "evidence": internal + external,
        "evidence_count": 10,
        "unexpected_external_models": [],
        "e5_consumed": False,
        "a8_consumed": False,
        "excluded_models_consumed": False,
    }
