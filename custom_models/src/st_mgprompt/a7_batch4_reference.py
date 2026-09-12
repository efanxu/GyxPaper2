from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from st_mgprompt.a7_batch4_contract import (
    A7_DEFINITION,
    A7_MODEL_ID,
    A7_REFERENCE_ID,
    A7_REFERENCE_RELATIVE_PATH,
    A7_RUN_RELATIVE_PATH,
    A7_RUN_ID,
    A7_SCOPE_ID,
    A7_VARIANT,
    LOSS_ID,
    PRECISION_POLICY,
    TRAINING_PROFILE_ID,
)
from st_mgprompt.a7_batch4_readiness import A7_EXPLICIT_CONFIG, inspect_a7_run

from benchmark_v2.artifacts import atomic_write_json
from benchmark_v2.runtime import PROJECT_ROOT


A7_RUN_ROOT = PROJECT_ROOT / A7_RUN_RELATIVE_PATH
A7_RUNTIME_REFERENCE_PATH = PROJECT_ROOT / A7_REFERENCE_RELATIVE_PATH
REFERENCE_KEYS = (
    "reference_type", "reference_id", "status", "scope_id", "model_id",
    "run_id", "variant", "definition", "training_profile_id",
    "train_batch_size", "val_batch_size", "test_batch_size",
    "gradient_accumulation_steps", "seed", "lookback", "max_pred_len",
    "loss_id", "precision", "formal_training", "source_path",
    "source_config_path", "checkpoint_path", "source_metrics_paths",
    "metrics_complete", "checkpoint_loadable", "config_conflicts",
    "metrics_validation_reasons",
)


def build_a7_reference(run_root: str | Path = A7_RUN_ROOT) -> dict[str, Any]:
    root = Path(run_root).resolve()
    inspected = inspect_a7_run(root)
    return {
        "reference_type": "FORMAL_A7_BATCH4_READ_ONLY",
        "reference_id": A7_REFERENCE_ID,
        "status": inspected["status"],
        "scope_id": A7_SCOPE_ID,
        "model_id": A7_MODEL_ID,
        "run_id": A7_RUN_ID,
        "variant": A7_VARIANT,
        "definition": A7_DEFINITION,
        "training_profile_id": TRAINING_PROFILE_ID,
        "train_batch_size": 4,
        "val_batch_size": 4,
        "test_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "seed": 2026,
        "lookback": 144,
        "max_pred_len": 10,
        "loss_id": LOSS_ID,
        "precision": PRECISION_POLICY,
        "formal_training": True,
        "source_path": str(root),
        "source_config_path": str(root / "effective_config.json"),
        "checkpoint_path": str(root / "best_checkpoint.pt"),
        "source_metrics_paths": [
            str(root / f"metrics_eval_h{horizon}.json") for horizon in (3, 6, 10)
        ],
        "metrics_complete": inspected["metrics_complete"],
        "checkpoint_loadable": inspected["checkpoint_loadable"],
        "config_conflicts": {
            **inspected["explicit_config_conflicts"],
            **{
                key: {"expected": A7_EXPLICIT_CONFIG[key], "actual": None}
                for key in inspected["missing_config_fields"]
            },
        },
        "metrics_validation_reasons": [
            reason for reason in inspected["reasons"]
            if reason.startswith(("MISSING:metrics_", "INVALID:metrics_", "HORIZON_", "NONFINITE:", "INVALID_COUNT:"))
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_reference(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def validate_a7_reference(
    training_profile: str | None = TRAINING_PROFILE_ID,
    *,
    reference_path: str | Path = A7_RUNTIME_REFERENCE_PATH,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    if training_profile != TRAINING_PROFILE_ID:
        raise ValueError("A7 reference requires uniform_train_batch4_v1.")
    selected_path = Path(reference_path)
    published = _load_reference(selected_path)
    if published is None:
        missing = build_a7_reference(run_root or A7_RUN_ROOT)
        missing["status"] = "NOT_READY"
        missing["metrics_validation_reasons"] = ["MISSING_RUNTIME_REFERENCE"]
        return missing
    missing_keys = [key for key in REFERENCE_KEYS if key not in published]
    source = Path(run_root or str(published.get("source_path", ""))).resolve()
    current = build_a7_reference(source)
    expected_identity = {
        key: current[key]
        for key in (
            "reference_type", "reference_id", "scope_id", "model_id", "run_id",
            "variant", "definition", "training_profile_id", "train_batch_size",
            "val_batch_size", "test_batch_size", "gradient_accumulation_steps",
            "seed", "lookback", "max_pred_len", "loss_id", "precision",
            "formal_training", "source_path", "source_config_path",
            "checkpoint_path", "source_metrics_paths",
        )
    }
    conflicts = {
        key: {"expected": expected, "actual": published.get(key)}
        for key, expected in expected_identity.items()
        if published.get(key) != expected
    }
    all_paths = [
        Path(str(published.get("source_config_path", ""))),
        Path(str(published.get("checkpoint_path", ""))),
        *[Path(str(value)) for value in published.get("source_metrics_paths", [])],
    ]
    same_run = len(all_paths) == 5 and all(path.parent.resolve() == source for path in all_paths)
    ready = (
        not missing_keys
        and not conflicts
        and same_run
        and published.get("status") == "READY"
        and current["status"] == "READY"
    )
    result = {key: current[key] for key in REFERENCE_KEYS}
    result.update({
        "status": "READY" if ready else "NOT_READY",
        "config_conflicts": {**current["config_conflicts"], **conflicts},
        "metrics_validation_reasons": [
            *current["metrics_validation_reasons"],
            *([f"MISSING_REFERENCE_FIELDS:{','.join(missing_keys)}"] if missing_keys else []),
            *([] if same_run else ["REFERENCE_PATHS_NOT_FROM_ONE_A7_RUN"]),
        ],
        "created_at": published.get("created_at"),
        "reference_path": str(selected_path),
    })
    return result


def create_a7_reference(
    output_path: str | Path = A7_RUNTIME_REFERENCE_PATH,
    training_profile: str | None = TRAINING_PROFILE_ID,
    *,
    run_root: str | Path = A7_RUN_ROOT,
) -> dict[str, Any]:
    if training_profile != TRAINING_PROFILE_ID:
        raise ValueError("A7 reference requires uniform_train_batch4_v1.")
    reference = build_a7_reference(run_root)
    if reference["status"] != "READY":
        raise RuntimeError("A7 reference cannot be published before readiness.")
    atomic_write_json(Path(output_path), reference)
    return reference


__all__ = [
    "A7_RUN_ROOT", "A7_RUNTIME_REFERENCE_PATH", "REFERENCE_KEYS",
    "build_a7_reference", "create_a7_reference", "validate_a7_reference",
]
