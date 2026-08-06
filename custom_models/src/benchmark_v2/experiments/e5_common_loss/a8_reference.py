from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...artifacts import atomic_write_json
from ...runtime import PROJECT_ROOT
from ...training_profiles import load_training_profile
from .contracts import BATCH4_A8_REFERENCE_ID


A8_RUN_ROOT = (
    PROJECT_ROOT / "custom_models/results/st_mgprompt_uniform_bs4"
    / "component_ablation_a8_bs4_seed2026/STMGPrompt_ComponentAblation"
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_metrics(run_root: Path) -> tuple[bool, list[str], dict[int, dict[str, Any]]]:
    reasons: list[str] = []
    payloads: dict[int, dict[str, Any]] = {}
    for horizon in (3, 6, 10):
        path = run_root / f"metrics_eval_h{horizon}.json"
        if not path.is_file():
            reasons.append(f"MISSING:{path.name}")
            continue
        try:
            payload = _load(path)
        except (OSError, ValueError) as exc:
            reasons.append(f"INVALID:{path.name}:{type(exc).__name__}")
            continue
        payloads[horizon] = payload
        if payload.get("horizon") != horizon:
            reasons.append(f"HORIZON_CONFLICT:{path.name}")
        for key in ("MAE", "RMSE", "R2", "Score"):
            value = payload.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                reasons.append(f"NONFINITE:{path.name}:{key}")
        count = payload.get("valid_target_count", payload.get("ValidCount"))
        if not isinstance(count, (int, float)) or count <= 0:
            reasons.append(f"INVALID_COUNT:{path.name}")
    return not reasons, reasons, payloads


def validate_a8_reference(training_profile: str | None = None) -> dict[str, Any]:
    profile = load_training_profile(training_profile)
    required = [
        A8_RUN_ROOT / "effective_config.json", A8_RUN_ROOT / "best_checkpoint.pt",
        A8_RUN_ROOT / "run_status.json", A8_RUN_ROOT / "metrics_eval_h3.json",
        A8_RUN_ROOT / "metrics_eval_h6.json", A8_RUN_ROOT / "metrics_eval_h10.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"status": "BLOCKED_FORMAL_A8_NOT_READY", "reference_id": BATCH4_A8_REFERENCE_ID, "missing": missing}
    config = _load(A8_RUN_ROOT / "effective_config.json")
    status = _load(A8_RUN_ROOT / "run_status.json")
    expected = {
        "component_ablation": "A8", "seed": 2026, "lookback": 144,
        "max_pred_len": 10, "train_batch_size": 4,
        "val_batch_size": 4, "test_batch_size": 4,
    }
    conflicts = {key: {"actual": config.get(key), "expected": value} for key, value in expected.items() if config.get(key) != value}
    metrics_complete, metric_reasons, metrics = _validate_metrics(A8_RUN_ROOT)
    checkpoint = A8_RUN_ROOT / "best_checkpoint.pt"
    checkpoint_loadable = False
    try:
        import torch
        torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_loadable = checkpoint.stat().st_size > 0
    except Exception:
        checkpoint_loadable = False
    passed = (
        not conflicts and metrics_complete and checkpoint_loadable
        and status.get("status") == "COMPLETED" and status.get("exit_code") == 0
        and config.get("formal_training", True) is True
    )
    result = {
        "reference_type": "FORMAL_A8_BATCH4_READ_ONLY",
        "reference_id": BATCH4_A8_REFERENCE_ID,
        "status": "VALID" if passed else "BLOCKED_FORMAL_A8_NOT_READY",
        "scope_id": config.get("scope_id", "st_mgprompt_a8_batch4_seed2026"),
        "run_id": config.get("run_id", "component_ablation_a8_bs4_seed2026"),
        "variant": "A8", "definition": "w/o MS-MG-DWU",
        "batch_size": 4, "lookback": 144, "node_count": 134,
        "feature_count": config.get("feature_count", 16), "horizon": 10,
        "loss_id": config.get("loss_id", config.get("loss_function", "masked_score_aligned_hybrid")),
        "precision": config.get("precision", config.get("precision_policy")),
        "formal_training": True,
        "source_path": str(A8_RUN_ROOT.resolve()),
        "checkpoint_path": str(checkpoint.resolve()),
        "metrics_paths": [str((A8_RUN_ROOT / f"metrics_eval_h{h}.json").resolve()) for h in (3, 6, 10)],
        "metrics": metrics, "metrics_complete": metrics_complete,
        "metrics_validation_reasons": metric_reasons,
        "checkpoint_loadable": checkpoint_loadable,
        "config_conflicts": conflicts,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if profile is not None:
        result.update(profile.identity())
    return result


def create_a8_reference(output_path: str | Path, training_profile: str | None = None) -> dict[str, Any]:
    reference = validate_a8_reference(training_profile=training_profile)
    atomic_write_json(Path(output_path), reference)
    return reference
