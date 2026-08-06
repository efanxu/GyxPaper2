from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from ...artifacts import atomic_write_json
from ...runtime import PROJECT_ROOT
from .a8_reference import A8_RUN_ROOT, A8_RUNTIME_REFERENCE_PATH, validate_a8_reference
from .contracts import FORMAL_OUTPUT_ROOT_RELATIVE
from .scope27_contract import E5_SCOPE27_ID, TRAINING_PROFILE_ID


DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/E5/E5_SCOPE27_VARIANT_MANIFEST.json"
CORE_EXPECTED = {
    "scope_id": E5_SCOPE27_ID,
    "loss_id": "masked_score_aligned_hybrid",
    "training_batch_profile_id": TRAINING_PROFILE_ID,
    "train_batch_size": 4,
    "val_batch_size": 4,
    "test_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "seed": 2026,
    "lookback": 144,
    "max_pred_len": 10,
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_scope27_manifest(manifest: Mapping[str, Any] | str | Path | None = None) -> dict[str, Any]:
    if manifest is None:
        payload = _load(DEFAULT_MANIFEST_PATH)
    elif isinstance(manifest, Mapping):
        payload = dict(manifest)
    else:
        payload = _load(Path(manifest))
    if not isinstance(payload, dict):
        raise ValueError("E5 scope27 manifest must be an object.")
    if payload.get("scope_id") != E5_SCOPE27_ID or payload.get("training_profile_id") != TRAINING_PROFILE_ID:
        raise ValueError("E5 scope27 requires the static Batch4 manifest.")
    entries = list(payload.get("entries", []))
    if len(entries) != 27:
        raise ValueError("E5 scope27 manifest must contain 27 entries.")
    benchmark = [row for row in entries if row.get("entry_type") != "REFERENCE_ONLY_FORMAL_A8"]
    if any(not str(row.get("e5_run_id", "")).endswith("_bs4_seed2026") for row in benchmark):
        raise ValueError("E5 scope27 manifest contains a non-Batch4 benchmark run-id.")
    reference = [row for row in entries if row.get("entry_type") == "REFERENCE_ONLY_FORMAL_A8"]
    if len(reference) != 1 or reference[0].get("e5_run_id") != "STMGPrompt_A8_loss_msa_hybrid_bs4_seed2026_reference":
        raise ValueError("E5 scope27 A8 reference id is invalid.")
    return payload


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def metrics_ready(run_dir: str | Path) -> tuple[bool, list[str]]:
    root = Path(run_dir)
    reasons: list[str] = []
    for horizon in (3, 6, 10):
        path = root / f"metrics_eval_h{horizon}.json"
        try:
            payload = _load(path)
        except FileNotFoundError:
            reasons.append(f"MISSING:{path.name}")
            continue
        except (OSError, ValueError):
            reasons.append(f"INVALID:{path.name}")
            continue
        if not isinstance(payload, dict) or payload.get("horizon") != horizon:
            reasons.append(f"HORIZON_CONFLICT:H{horizon}")
            continue
        for key in ("Score", "MAE", "RMSE", "R2"):
            if not _finite(payload.get(key)):
                reasons.append(f"NONFINITE:H{horizon}:{key}")
        count = payload.get("valid_target_count", payload.get("ValidCount"))
        if not _finite(count) or float(count) <= 0:
            reasons.append(f"INVALID_COUNT:H{horizon}")
    return not reasons, reasons


def _checkpoint_loadable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        import torch

        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def inspect_benchmark_run(
    entry: Mapping[str, Any],
    run_dir: str | Path,
    *,
    scope_id: str = E5_SCOPE27_ID,
    training_profile_id: str = TRAINING_PROFILE_ID,
) -> dict[str, Any]:
    root = Path(run_dir)
    run_id = str(entry.get("e5_run_id") or entry.get("run_id"))
    if not root.is_dir():
        return {
            "model_id": entry["model_id"], "run_id": run_id,
            "run_dir": str(root), "status": "RUN_MISSING", "ready": False,
            "reasons": ["RUN_MISSING"], "explicit_config_conflicts": {},
            "metrics_complete": False,
            "checkpoint_loadable": entry.get("entry_type") != "TRAIN_COMMON_LOSS",
        }
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
    expected = {
        **CORE_EXPECTED,
        "scope_id": scope_id,
        "training_batch_profile_id": training_profile_id,
        "model_id": entry["model_id"],
        "run_id": run_id,
        "loss_id": entry["loss_id"],
        "formal_training": entry.get("entry_type") == "TRAIN_COMMON_LOSS",
    }
    missing_fields = [key for key in expected if key not in effective]
    conflicts = {
        key: {"expected": value, "actual": effective.get(key)}
        for key, value in expected.items()
        if key in effective and effective.get(key) != value
    }
    if status.get("status") != "COMPLETED" or status.get("exit_code") != 0:
        reasons.append("NOT_COMPLETED")
    metrics_complete, metric_reasons = metrics_ready(root)
    reasons.extend(metric_reasons)
    checkpoint_loadable = True
    if entry.get("entry_type") == "TRAIN_COMMON_LOSS":
        checkpoint_loadable = _checkpoint_loadable(root / "best_checkpoint.pt")
        if not checkpoint_loadable:
            reasons.append("CHECKPOINT_NOT_LOADABLE")
    if conflicts:
        reasons.append("EXPLICIT_CONFIG_CONFLICT")
    if missing_fields:
        reasons.append(f"MISSING_EXPLICIT_CONFIG_FIELDS:{','.join(missing_fields)}")
    ready = not reasons
    return {
        "model_id": entry["model_id"], "run_id": run_id,
        "run_dir": str(root), "status": "COMPLETED" if ready else "INCOMPLETE",
        "ready": ready, "reasons": reasons,
        "explicit_config_conflicts": conflicts,
        "missing_config_fields": missing_fields,
        "metrics_complete": metrics_complete,
        "checkpoint_loadable": checkpoint_loadable,
        "pid": status.get("pid"),
        "process_start_time": status.get("process_start_time"),
    }


def build_readiness(
    *,
    output_root: str | Path | None = None,
    report_path: str | Path | None = None,
    manifest: Mapping[str, Any] | str | Path | None = None,
    training_profile: str = TRAINING_PROFILE_ID,
    a8_reference_path: str | Path = A8_RUNTIME_REFERENCE_PATH,
    a8_run_root: str | Path = A8_RUN_ROOT,
    legacy_scope29: bool = False,
) -> dict[str, Any]:
    del legacy_scope29
    if training_profile != TRAINING_PROFILE_ID:
        raise ValueError("E5 scope27 readiness requires uniform_train_batch4_v1.")
    selected_manifest = load_scope27_manifest(manifest)
    root = Path(output_root or (PROJECT_ROOT / FORMAL_OUTPUT_ROOT_RELATIVE))
    rows = []
    for entry in selected_manifest["entries"]:
        if entry["entry_type"] == "REFERENCE_ONLY_FORMAL_A8":
            reference = validate_a8_reference(
                training_profile=training_profile,
                reference_path=a8_reference_path,
                run_root=a8_run_root,
            )
            ready = reference["status"] == "READY"
            rows.append({
                "entry_id": entry["entry_id"], "model_id": entry["model_id"],
                "run_id": entry["e5_run_id"], "entry_type": entry["entry_type"],
                "ready": ready,
                "metrics_complete": bool(reference["metrics_complete"]),
                "checkpoint_loadable": bool(reference["checkpoint_loadable"]),
                "explicit_config_conflicts": reference["config_conflicts"],
                "reasons": [] if ready else [
                    *reference["metrics_validation_reasons"],
                    *(["A8_REFERENCE_NOT_READY"] if not reference["metrics_validation_reasons"] else []),
                ],
                "reference": reference,
            })
            continue
        run_id = str(entry.get("e5_run_id") or entry.get("run_id"))
        inspected = inspect_benchmark_run(entry, root / run_id)
        rows.append({"entry_id": entry["entry_id"], "entry_type": entry["entry_type"], **inspected})
    report = {
        "schema_version": "e5_result_readiness_v3",
        "scope_id": selected_manifest["scope_id"],
        "status": "READY" if all(row["ready"] for row in rows) else "NOT_READY",
        "expected_total_entries": 27,
        "ready_entries": sum(bool(row["ready"]) for row in rows),
        "counts": {"trainable": 24, "evaluate_only": 2, "a8_reference": 1, "total": 27},
        "output_root": str(root),
        "entries": rows,
    }
    if report_path is not None:
        atomic_write_json(Path(report_path), report)
    return report


__all__ = [
    "CORE_EXPECTED", "DEFAULT_MANIFEST_PATH", "build_readiness",
    "inspect_benchmark_run", "load_scope27_manifest", "metrics_ready",
]
