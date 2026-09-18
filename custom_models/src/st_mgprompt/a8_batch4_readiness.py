from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .a8_batch4_contract import (
    A8_DEFINITION,
    A8_MODEL_ID,
    A8_RUN_ID,
    A8_SCOPE_ID,
    A8_VARIANT,
    LOSS_ID,
    PRECISION_POLICY,
    TRAINING_PROFILE_ID,
)


REQUIRED_HORIZONS = (3, 6, 10)
A8_EXPLICIT_CONFIG = {
    "scope_id": A8_SCOPE_ID,
    "model_id": A8_MODEL_ID,
    "run_id": A8_RUN_ID,
    "component_ablation": A8_VARIANT,
    "variant": A8_VARIANT,
    "definition": A8_DEFINITION,
    "training_batch_profile_id": TRAINING_PROFILE_ID,
    "train_batch_size": 4,
    "val_batch_size": 4,
    "test_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "seed": 2026,
    "lookback": 144,
    "max_pred_len": 10,
    "loss_function": LOSS_ID,
    "precision_policy": PRECISION_POLICY,
    "amp_enabled": False,
    "formal_training": True,
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_metrics(run_root: str | Path) -> tuple[bool, list[str], dict[int, dict[str, Any]]]:
    root = Path(run_root)
    reasons: list[str] = []
    payloads: dict[int, dict[str, Any]] = {}
    for horizon in REQUIRED_HORIZONS:
        path = root / f"metrics_eval_h{horizon}.json"
        try:
            payload = _load(path)
        except FileNotFoundError:
            reasons.append(f"MISSING:{path.name}")
            continue
        except (OSError, ValueError):
            reasons.append(f"INVALID:{path.name}")
            continue
        if not isinstance(payload, dict):
            reasons.append(f"INVALID:{path.name}")
            continue
        payloads[horizon] = payload
        if payload.get("horizon") != horizon:
            reasons.append(f"HORIZON_CONFLICT:H{horizon}")
        for key in ("MAE", "RMSE", "R2", "Score"):
            if not _finite(payload.get(key)):
                reasons.append(f"NONFINITE:H{horizon}:{key}")
        count = payload.get("valid_target_count", payload.get("ValidCount"))
        if not _finite(count) or float(count) <= 0:
            reasons.append(f"INVALID_COUNT:H{horizon}")
    return not reasons, reasons, payloads


def checkpoint_loadable(path: str | Path) -> bool:
    selected = Path(path)
    if not selected.is_file() or selected.stat().st_size <= 0:
        return False
    try:
        import torch

        torch.load(selected, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def inspect_a8_run(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    reasons: list[str] = []
    try:
        status = _load(root / "run_status.json")
    except (FileNotFoundError, OSError, ValueError):
        status = {}
        reasons.append("MISSING_OR_INVALID:run_status.json")
    try:
        effective = _load(root / "effective_config.json")
    except (FileNotFoundError, OSError, ValueError):
        effective = {}
        reasons.append("MISSING_OR_INVALID:effective_config.json")
    missing_fields = [key for key in A8_EXPLICIT_CONFIG if key not in effective]
    conflicts = {
        key: {"expected": expected, "actual": effective.get(key)}
        for key, expected in A8_EXPLICIT_CONFIG.items()
        if key in effective and effective.get(key) != expected
    }
    if status.get("status") != "COMPLETED" or status.get("exit_code") != 0:
        reasons.append("NOT_COMPLETED")
    metrics_complete, metric_reasons, metrics = validate_metrics(root)
    reasons.extend(metric_reasons)
    checkpoint = root / "best_checkpoint.pt"
    loadable = checkpoint_loadable(checkpoint)
    if not loadable:
        reasons.append("CHECKPOINT_NOT_LOADABLE")
    if conflicts:
        reasons.append("EXPLICIT_CONFIG_CONFLICT")
    if missing_fields:
        reasons.append(f"MISSING_EXPLICIT_CONFIG_FIELDS:{','.join(missing_fields)}")
    ready = not reasons
    return {
        "status": "READY" if ready else "NOT_READY",
        "ready": ready,
        "scope_id": A8_SCOPE_ID,
        "model_id": A8_MODEL_ID,
        "run_id": A8_RUN_ID,
        "run_dir": str(root),
        "metrics": metrics,
        "metrics_complete": metrics_complete,
        "checkpoint_loadable": loadable,
        "explicit_config_conflicts": conflicts,
        "missing_config_fields": missing_fields,
        "reasons": reasons,
        "pid": status.get("pid"),
        "process_start_time": status.get("process_start_time"),
    }


__all__ = [
    "A8_EXPLICIT_CONFIG", "REQUIRED_HORIZONS", "checkpoint_loadable",
    "inspect_a8_run", "validate_metrics",
]
