from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


METRIC_HORIZONS = (3, 6, 10)
METRIC_FIELDS = ("MAE", "RMSE", "R2")
CONFIG_FIELDS = (
    "model_name",
    "graph_operator",
    "decoder_context_mode",
    "macro_prompt_len",
    "macro_prompt_pooling",
    "cross_fusion_recent_len",
    "fusion_mode",
    "st_prompt_mode",
    "st_prompt_use_node_identity",
    "hidden_dim",
    "num_coupling_layers",
    "lookback",
    "max_pred_len",
    "eval_horizons",
    "test_sample_stride",
    "test_batch_size",
    "seed",
    "feature_cols",
    "target_col",
    "target_mask_col",
    "physical_power_min_kw",
    "physical_power_max_kw",
    "physical_clip_protocol",
    "prediction_accumulation",
    "windows_safe_mode",
)

_CONFIG_DEFAULTS = {
    "st_prompt_mode": "full",
    "macro_prompt_pooling": "attention",
    "st_prompt_use_node_identity": True,
}

_WINDOWS_EXCEPTIONS = {
    0xC0000005: "access_violation",
    0xC0000409: "stack_buffer_overrun_or_fast_fail",
    0xC0000374: "heap_corruption",
    0xC0000017: "no_memory",
}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    raise TypeError(f"Expected a config object or dict, got {type(value).__name__}.")


def _json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing: {path.name}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid JSON {path.name}: {type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return None, f"JSON object expected in {path.name}"
    return value, None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def classify_returncode(returncode: int | None) -> dict[str, Any]:
    if returncode is None:
        return {
            "returncode_signed": None,
            "returncode_unsigned": None,
            "returncode_hex": None,
            "windows_exception_classification": "not_started",
        }
    unsigned = int(returncode) & 0xFFFFFFFF
    signed = unsigned if unsigned < 0x80000000 else unsigned - 0x100000000
    if unsigned == 0:
        classification = "none"
    elif unsigned in _WINDOWS_EXCEPTIONS:
        classification = _WINDOWS_EXCEPTIONS[unsigned]
    elif unsigned >= 0xC0000000:
        classification = "windows_native_exit"
    else:
        classification = "process_nonzero_exit"
    return {
        "returncode_signed": signed,
        "returncode_unsigned": unsigned,
        "returncode_hex": f"0x{unsigned:08X}",
        "windows_exception_classification": classification,
    }


def _cpu_load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint in an isolated process so a native torch crash cannot kill the runner."""

    code = (
        "import json, sys, torch; "
        "x=torch.load(sys.argv[1], map_location='cpu', weights_only=False); "
        "print(json.dumps({'keys': list(x)[:20] if isinstance(x, dict) else [], "
        "'model_id': x.get('model_id') if isinstance(x, dict) else None, "
        "'best_epoch': x.get('best_epoch') if isinstance(x, dict) else None}))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code, str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        return {"valid": False, "error": f"checkpoint CPU load timed out: {exc}"}
    result = classify_returncode(completed.returncode)
    if completed.returncode != 0:
        return {
            "valid": False,
            "error": completed.stderr.strip() or completed.stdout.strip() or "checkpoint loader exited non-zero",
            **result,
        }
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"valid": False, "error": f"checkpoint loader returned invalid metadata: {exc}", **result}
    return {"valid": True, **payload, **result}


def _read_train_log(path: Path) -> tuple[bool, int | None, str | None]:
    if not path.exists():
        return False, None, "missing: train_log.csv"
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        return False, None, f"invalid train_log.csv: {type(exc).__name__}: {exc}"
    epochs: list[int] = []
    for row in rows:
        try:
            epoch = int(row.get("epoch", ""))
        except (TypeError, ValueError):
            continue
        if epoch > 0:
            epochs.append(epoch)
    if not epochs:
        return False, None, "train_log.csv has no complete epoch row"
    return True, max(epochs), None


def _config_matches(actual: dict[str, Any] | None, expected: dict[str, Any]) -> tuple[bool, list[str]]:
    if actual is None:
        return False, ["config.json or active_config.json missing/invalid"]
    differences = []
    for key in CONFIG_FIELDS:
        actual_value = actual.get(key, _CONFIG_DEFAULTS.get(key))
        if actual_value != expected.get(key):
            differences.append(key)
    return not differences, differences


def inspect_variant_artifacts(
    run_dir: str | Path,
    expected_variant: str,
    expected_config: Any,
    returncode: int | None = 0,
) -> dict[str, Any]:
    """Audit a formal variant without changing any artifact in ``run_dir``."""

    run_dir = Path(run_dir)
    expected = _as_dict(expected_config)
    errors: list[str] = []
    train_complete, train_error = _json(run_dir / "train_complete.json")
    eval_complete, eval_error = _json(run_dir / "evaluation_complete.json")
    active_config, active_error = _json(run_dir / "active_config.json")
    config_json, config_error = _json(run_dir / "config.json")
    actual_config = active_config or config_json

    if train_error:
        errors.append(train_error)
    if eval_error:
        errors.append(eval_error)
    if active_error and config_error:
        errors.append(active_error)
    config_ok, config_differences = _config_matches(actual_config, expected)
    if not config_ok:
        errors.append("config mismatch: " + ", ".join(config_differences))
    checkpoint_path = run_dir / "best_checkpoint.pt"
    checkpoint_probe = _cpu_load_checkpoint(checkpoint_path) if checkpoint_path.exists() else {"valid": False, "error": "missing: best_checkpoint.pt"}
    checkpoint_valid = bool(checkpoint_probe.get("valid"))
    if not checkpoint_valid:
        errors.append(str(checkpoint_probe.get("error", "best_checkpoint.pt failed CPU load")))

    log_ok, last_epoch, log_error = _read_train_log(run_dir / "train_log.csv")
    if log_error:
        errors.append(log_error)
    best_epoch = train_complete.get("best_epoch") if train_complete else None
    best_score = train_complete.get("best_val_score_h10") if train_complete else None
    training_complete = bool(
        train_complete
        and train_complete.get("status") == "completed"
        and checkpoint_valid
        and log_ok
        and isinstance(best_epoch, int)
        and best_epoch > 0
        and _finite(best_score)
        and config_ok
    )

    metrics: dict[int, dict[str, Any]] = {}
    metrics_complete = True
    for horizon in METRIC_HORIZONS:
        path = run_dir / f"metrics_eval_h{horizon}.json"
        value, error = _json(path)
        if error:
            errors.append(error)
            metrics_complete = False
            continue
        metrics[horizon] = value or {}
        score = value.get("Score", value.get("score")) if value else None
        if not value or not all(_finite(value.get(field)) for field in METRIC_FIELDS) or not _finite(score):
            errors.append(f"metrics_eval_h{horizon}.json has non-finite or missing MAE/RMSE/R2/Score")
            metrics_complete = False

    provenance_ok = bool(
        eval_complete
        and eval_complete.get("status") == "completed"
        and eval_complete.get("checkpoint") in {"best_checkpoint.pt", str(checkpoint_path)}
    )
    metadata, metadata_error = _json(run_dir / "prediction_metadata.json")
    if metadata_error:
        errors.append(metadata_error)
    metadata_provenance_ok = bool(
        metadata is not None
        and metadata.get("artifact_source_checkpoint") in {None, "best_checkpoint.pt", str(checkpoint_path)}
    )
    provenance_ok = provenance_ok and metadata_provenance_ok
    if not provenance_ok:
        errors.append("evaluation checkpoint provenance does not point to best_checkpoint.pt")

    evaluation_complete = bool(eval_complete and metrics_complete and provenance_ok and config_ok)
    protocol_consistent = bool(config_ok and expected_variant)
    protocol_report, protocol_error = _json(run_dir / "protocol_check.json")
    if protocol_error and (run_dir / "protocol_check.json").exists():
        errors.append(protocol_error)
    if protocol_report is not None and protocol_report.get("passed") is False:
        protocol_consistent = False
        errors.append("protocol_check.json reports failed protocol checks")
    if not protocol_consistent:
        errors.append("variant/config protocol mismatch")

    rc_info = classify_returncode(returncode)
    failure, _ = _json(run_dir / "failure.json")
    classification = rc_info["windows_exception_classification"]
    if training_complete and evaluation_complete:
        final_status = "completed" if returncode in (None, 0) else "completed_with_exit_warning"
    elif training_complete and not evaluation_complete:
        final_status = "training_completed_evaluation_incomplete"
    elif failure and failure.get("error_type"):
        final_status = "failed_python_exception"
    elif classification in set(_WINDOWS_EXCEPTIONS.values()) | {"windows_native_exit"}:
        final_status = "failed_windows_native_exit"
    elif not protocol_consistent:
        final_status = "failed_protocol_mismatch"
    else:
        final_status = "failed_incomplete_artifacts"

    return {
        "variant": expected_variant,
        "run_dir": str(run_dir.resolve()),
        "training_complete": training_complete,
        "evaluation_complete": evaluation_complete,
        "checkpoint_valid": checkpoint_valid,
        "metrics_complete": metrics_complete,
        "protocol_consistent": protocol_consistent,
        "config_match": config_ok,
        "config_differences": config_differences,
        "train_log_nonempty": log_ok,
        "last_epoch": last_epoch,
        "best_epoch": best_epoch,
        "best_val_score_h10": best_score,
        "metrics_horizons": sorted(metrics),
        "checkpoint_probe": checkpoint_probe,
        "artifact_errors": errors,
        "final_status": final_status,
        **rc_info,
    }


def is_completed_status(status: str | None) -> bool:
    return status in {"completed", "completed_with_exit_warning"}
