from __future__ import annotations

from pathlib import Path
from typing import Any

from .config_diff import compare_configs
from .constants import (
    ARCHITECTURE_FAMILIES, CONTROL_LOSS_ID, LOSS_ID, MODEL_IDS, REQUIRED_TRAIN_ARTIFACTS,
    SCOPE_MANIFEST, TRAINING_PROFILE_ID, TRANSFER_ROOT,
)
from .io_utils import forbidden_source, read_json, read_metrics
from .variants import canonical_transfer_run_id


def _transfer_entry(model_id: str, control: dict[str, Any], transfer_root: str | Path) -> dict[str, Any]:
    run_id = canonical_transfer_run_id(model_id)
    run_dir = Path(transfer_root).resolve() / run_id
    errors: list[str] = []
    if forbidden_source(run_dir):
        errors.append("FORBIDDEN_TRANSFER_SOURCE")
    missing = [name for name in REQUIRED_TRAIN_ARTIFACTS if not (run_dir / name).is_file()]
    errors.extend(f"MISSING_ARTIFACT:{name}" for name in missing)
    effective = read_json(run_dir / "effective_config.json") if (run_dir / "effective_config.json").is_file() else {}
    resolved = read_json(run_dir / "resolved_config.json") if (run_dir / "resolved_config.json").is_file() else {}
    status = read_json(run_dir / "run_status.json") if (run_dir / "run_status.json").is_file() else {}
    protocol = read_json(run_dir / "protocol_check.json") if (run_dir / "protocol_check.json").is_file() else {}
    init = read_json(run_dir / "initialization_provenance.json") if (run_dir / "initialization_provenance.json").is_file() else {}
    if status.get("status") != "COMPLETED" or status.get("run_mode") != "formal":
        errors.append("TRANSFER_RUN_STATUS_INVALID")
    if protocol.get("status") != "PASS":
        errors.append("TRANSFER_PROTOCOL_STATUS_INVALID")
    if effective.get("model_id") != model_id or resolved.get("model_id") != model_id:
        errors.append("TRANSFER_MODEL_ID_INVALID")
    if effective.get("loss") != LOSS_ID or resolved.get("loss") != LOSS_ID:
        errors.append("TRANSFER_LOSS_ID_INVALID")
    if effective.get("training_batch_profile_id") != TRAINING_PROFILE_ID:
        errors.append("TRANSFER_BATCH_PROFILE_INVALID")
    for key in ("train_batch_size", "val_batch_size", "test_batch_size"):
        if effective.get(key) != 4:
            errors.append(f"TRANSFER_BATCH_INVALID:{key}")
    diff = compare_configs(control.get("effective_config", {}), effective, graph_model=model_id in {"dcrnn", "mtgnn"}) if effective else {
        "loss_only_diff_valid": False, "unexpected_differences": [{"path": "effective_config", "control": "present", "transfer": "missing", "allowed": False}],
    }
    if not diff.get("loss_only_diff_valid"):
        errors.append("INVALID_CONFIG_DIFF")
    metrics = None
    if (run_dir / "metrics.csv").is_file():
        try:
            metrics = read_metrics(run_dir / "metrics.csv")
        except Exception as exc:
            errors.append(f"TRANSFER_METRICS_INVALID:{type(exc).__name__}:{exc}")
    return {
        "evidence_id": f"E9_TRANSFER_{model_id.upper()}", "evidence_role": "E9_TRANSFER",
        "model_id": model_id, "architecture_family": ARCHITECTURE_FAMILIES[model_id],
        "run_id": run_id, "source_root": str(run_dir), "control_or_transfer": "transfer",
        "loss_id": effective.get("loss"), "loss_profile": effective.get("loss_profile"),
        "loss_state_hash": "NOT_GENERATED_REPOSITORY_POLICY" if (run_dir / "best_checkpoint.pt").is_file() else None,
        "batch": effective.get("train_batch_size"), "seed": effective.get("seed"),
        "protocol_identity": effective.get("protocol_id") or effective.get("protocol_hash"),
        "graph_protocol_identity": effective.get("graph_context_id") if model_id in {"dcrnn", "mtgnn"} else None,
        "resolved_config_hash": None, "effective_config_hash": None,
        "checkpoint_hash": None, "metrics_hash": None,
        "run_status": status.get("status"), "artifact_completeness": not missing,
        "metrics": metrics, "effective_config": effective, "resolved_config": resolved,
        "initial_model_state_hash": init.get("initial_model_state_hash"),
        "rng_manifest_present": bool(init.get("rng_manifest")),
        "loss_only_diff": diff,
        "validation_status": "PASS" if not errors else "FAIL", "validation_errors": errors,
        "ready": not errors,
    }


def build_evidence_manifest(control_audit: dict[str, Any], transfer_root: str | Path = TRANSFER_ROOT) -> dict[str, Any]:
    controls = {row["model_id"]: row for row in control_audit["controls"]}
    evidence = []
    transfers: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_IDS:
        control = controls[model_id]
        control_errors = list(control.get("validation_errors", []))
        evidence.append({
            "evidence_id": f"E9_CONTROL_{model_id.upper()}", "evidence_role": "ORIGINAL26_CONTROL",
            "model_id": model_id, "architecture_family": ARCHITECTURE_FAMILIES[model_id],
            "run_id": control.get("run_id"), "source_root": control.get("source_root"),
            "control_or_transfer": "control", "loss_id": control.get("loss_id", CONTROL_LOSS_ID),
            "loss_profile": control.get("loss_profile"), "loss_state_hash": None,
            "batch": control.get("batch"), "seed": control.get("seed"),
            "protocol_identity": control.get("protocol_identity"),
            "graph_protocol_identity": control.get("effective_config", {}).get("graph_context_id") if model_id in {"dcrnn", "mtgnn"} else None,
            "resolved_config_hash": control.get("resolved_config_hash"),
            "effective_config_hash": control.get("effective_config_hash"),
            "checkpoint_hash": control.get("checkpoint_hash"), "metrics_hash": control.get("metrics_hash"),
            "run_status": control.get("run_status"), "artifact_completeness": control.get("artifact_completeness"),
            "metrics": control.get("metrics"), "validation_status": "PASS" if not control_errors else "FAIL",
            "validation_errors": control_errors, "ready": not control_errors,
        })
        transfer = _transfer_entry(model_id, control, transfer_root)
        transfers[model_id] = transfer
        evidence.append(transfer)
    return {
        "schema_version": "e9_evidence_manifest_v1", "core_evidence_expected": 12,
        "evidence": evidence, "transfer_by_model": transfers,
        "source_classes": ["ORIGINAL26_CONTROL", "E9_TRANSFER", "OPTIONAL_STMG_REFERENCE"],
        "optional_stmg_reference": [], "e5_consumed": False,
    }


def build_transfer_readiness(transfer_root: str | Path = TRANSFER_ROOT) -> dict[str, Any]:
    """Validate transfer artifacts against frozen controls without the workbook.

    This status is for the remote training runner only.  It does not replace the
    original26 workbook audit required by CORE_E9_READY.
    """
    scope = read_json(SCOPE_MANIFEST)
    entries = {row["model_id"]: row for row in scope.get("entries", []) if row.get("model_id") in MODEL_IDS}
    rows = []
    for model_id in MODEL_IDS:
        entry = entries.get(model_id)
        if entry is None:
            rows.append({
                "model_id": model_id, "run_id": canonical_transfer_run_id(model_id),
                "ready": False, "validation_status": "FAIL",
                "validation_errors": ["MISSING_FROZEN_CONTROL_SCOPE_ENTRY"],
            })
            continue
        source_root = (SCOPE_MANIFEST.parents[4] / entry["output_root"] / entry["run_id"]).resolve()
        effective_path = source_root / "effective_config.json"
        control = {
            "effective_config": read_json(effective_path) if effective_path.is_file() else {},
            "source_root": str(source_root),
        }
        transfer = _transfer_entry(model_id, control, transfer_root)
        rows.append({
            "model_id": model_id, "run_id": transfer["run_id"], "source_root": transfer["source_root"],
            "ready": transfer["ready"], "validation_status": transfer["validation_status"],
            "validation_errors": transfer["validation_errors"],
            "loss_only_diff_valid": bool(transfer["loss_only_diff"].get("loss_only_diff_valid")),
            "run_status": transfer["run_status"], "artifact_completeness": transfer["artifact_completeness"],
        })
    ready_count = sum(bool(row["ready"]) for row in rows)
    return {
        "schema_version": "e9_transfer_readiness_v1",
        "MSMG_DWU_TRANSFER_READY": f"{ready_count}/6",
        "MSMG_DWU_TRANSFER_READY_BOOL": ready_count == 6,
        "requires_original26_for_core_e9": True,
        "rows": rows,
    }
