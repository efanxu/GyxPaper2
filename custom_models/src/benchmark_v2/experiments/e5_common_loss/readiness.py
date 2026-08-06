from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ...artifacts import atomic_write_json
from ...runtime import PROJECT_ROOT
from .a8_reference import validate_a8_reference
from .contracts import FORMAL_OUTPUT_ROOT_RELATIVE
from .variant_manifest import build_variant_manifest


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_ready(run_dir: Path) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for horizon in (3, 6, 10):
        path = run_dir / f"metrics_eval_h{horizon}.json"
        if not path.is_file():
            reasons.append(f"MISSING:{path.name}")
            continue
        try:
            payload = _load(path)
        except (OSError, ValueError):
            reasons.append(f"INVALID:{path.name}")
            continue
        for key in ("Score", "MAE", "RMSE", "R2"):
            value = payload.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                reasons.append(f"NONFINITE:{path.name}:{key}")
        count = payload.get("valid_target_count", payload.get("ValidCount"))
        if not isinstance(count, (int, float)) or count <= 0:
            reasons.append(f"INVALID_COUNT:{path.name}")
    return not reasons, reasons


def build_readiness(
    *, output_root: str | Path | None = None, report_path: str | Path | None = None,
    training_profile: str | None = None, legacy_scope29: bool = False,
) -> dict[str, Any]:
    del legacy_scope29
    manifest = build_variant_manifest(training_profile=training_profile)
    root = Path(output_root or (PROJECT_ROOT / FORMAL_OUTPUT_ROOT_RELATIVE))
    rows = []
    for entry in manifest["entries"]:
        if entry["entry_type"] == "REFERENCE_ONLY_FORMAL_A8":
            reference = validate_a8_reference(training_profile=training_profile)
            ready = reference.get("status") == "VALID"
            rows.append({"entry_id": entry["entry_id"], "model_id": entry["model_id"], "entry_type": entry["entry_type"], "ready": ready, "metrics_complete": bool(reference.get("metrics_complete")), "checkpoint_loadable": bool(reference.get("checkpoint_loadable")), "reasons": [] if ready else [reference.get("status")]})
            continue
        run_dir = root / entry["e5_run_id"]
        reasons: list[str] = []
        try:
            status = _load(run_dir / "run_status.json")
            effective = _load(run_dir / "effective_config.json")
        except (OSError, ValueError):
            status, effective = {}, {}
            reasons.append("RUN_MISSING_OR_INVALID")
        if status.get("status") != "COMPLETED" or status.get("exit_code") != 0:
            reasons.append("NOT_COMPLETED")
        explicit = {
            "model_id": entry["model_id"], "run_id": entry["e5_run_id"],
            "loss_id": entry["loss_id"], "training_batch_profile_id": training_profile,
        }
        for key, expected in explicit.items():
            actual = effective.get(key)
            if expected is not None and actual != expected:
                reasons.append(f"EXPLICIT_CONFIG_CONFLICT:{key}")
        metrics_complete, metric_reasons = _metrics_ready(run_dir)
        reasons.extend(metric_reasons)
        checkpoint_loadable = True
        if entry["entry_type"] == "TRAIN_COMMON_LOSS":
            checkpoint = run_dir / "best_checkpoint.pt"
            checkpoint_loadable = checkpoint.is_file() and checkpoint.stat().st_size > 0
            if checkpoint_loadable:
                try:
                    import torch
                    torch.load(checkpoint, map_location="cpu", weights_only=False)
                except Exception:
                    checkpoint_loadable = False
            if not checkpoint_loadable:
                reasons.append("CHECKPOINT_NOT_LOADABLE")
        rows.append({"entry_id": entry["entry_id"], "model_id": entry["model_id"], "entry_type": entry["entry_type"], "ready": not reasons, "metrics_complete": metrics_complete, "checkpoint_loadable": checkpoint_loadable, "reasons": reasons})
    report = {
        "schema_version": "e5_result_readiness_v2", "scope_id": manifest["scope_id"],
        "status": "READY" if all(row["ready"] for row in rows) else "NOT_READY",
        "expected_total_entries": 27, "ready_entries": sum(row["ready"] for row in rows),
        "counts": manifest["counts"], "output_root": str(root), "entries": rows,
    }
    if report_path is not None:
        atomic_write_json(Path(report_path), report)
    return report
