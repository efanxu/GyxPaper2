from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...artifacts import atomic_write_json
from ...runtime import PROJECT_ROOT
from ...training_profiles import batch_identity
from .a8_reference import validate_a8_reference
from .contracts import FORMAL_OUTPUT_ROOT_RELATIVE
from .variant_manifest import build_variant_manifest


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_readiness(
    *,
    output_root: str | Path | None = None,
    report_path: str | Path | None = None,
    training_profile: str | None = None,
    legacy_scope29: bool = False,
) -> dict[str, Any]:
    if not legacy_scope29:
        raise RuntimeError(
            "SUPERSEDED_E5_SCOPE29_COMMAND: current active scope is "
            "e5_batch4_scope27_seed2026; use the scope27 gate or active CLI."
        )
    manifest = build_variant_manifest(training_profile=training_profile)
    root = Path(output_root or (PROJECT_ROOT / FORMAL_OUTPUT_ROOT_RELATIVE))
    a8 = validate_a8_reference(training_profile=training_profile)
    expected_batch_identity = batch_identity(training_profile)
    rows = []
    for entry in manifest["entries"]:
        if entry["entry_type"] == "REFERENCE_ONLY_FORMAL_A8":
            ready = a8.get("status") == "VALID"
            rows.append(
                {
                    "entry_id": entry["entry_id"],
                    "model_id": entry["model_id"],
                    "entry_type": entry["entry_type"],
                    "expected": True,
                    "found": ready,
                    "formal": bool(a8.get("formal_complete")),
                    "completed": bool(a8.get("formal_complete")),
                    "protocol_match": bool(a8.get("protocol_match")),
                    "loss_match": a8.get("source_loss_id")
                    == "masked_score_aligned_hybrid",
                    "config_match": not a8.get("config_mismatches", {"missing": True}),
                    "metrics_complete": bool(a8.get("metrics_complete")),
                    "reference_valid": ready,
                    "ready": ready,
                    "blocked_reason": None
                    if ready
                    else a8.get("status", "A8_REFERENCE_INVALID"),
                }
            )
            continue
        run_dir = root / entry["e5_run_id"]
        status_path = run_dir / "run_status.json"
        effective_path = run_dir / "effective_config.json"
        metrics_paths = [run_dir / f"metrics_eval_h{h}.json" for h in (3, 6, 10)]
        found = run_dir.is_dir()
        status = _load(status_path) if status_path.is_file() else {}
        effective = _load(effective_path) if effective_path.is_file() else {}
        completed = status.get("status") == "COMPLETED"
        formal = status.get("run_mode") == "formal"
        loss = effective.get("loss", {})
        loss_match = (
            loss.get("id") == entry["loss_id"]
            and loss.get("source_hash") == entry["loss_source_hash"]
            and loss.get("profile_hash") == entry["loss_profile_hash"]
        )
        protocol_match = (
            effective.get("provenance", {}).get("e5_common_loss_protocol_hash")
            == entry["e5_common_loss_protocol_hash"]
        )
        config_match = (
            effective.get("provenance", {}).get("base_model_config_hash")
            == entry["base_model_config_hash"]
        )
        metrics_complete = all(path.is_file() for path in metrics_paths)
        batch_match = all(
            effective.get(key) == value
            for key, value in expected_batch_identity.items()
        )
        ready = all(
            (
                found,
                completed,
                formal,
                loss_match,
                protocol_match,
                config_match,
                metrics_complete,
                batch_match,
            )
        )
        reasons = []
        for condition, reason in (
            (found, "RUN_NOT_FOUND"),
            (formal, "NOT_FORMAL"),
            (completed, "NOT_COMPLETED"),
            (protocol_match, "E5_PROTOCOL_MISMATCH"),
            (loss_match, "LOSS_IDENTITY_MISMATCH"),
            (config_match, "BASE_CONFIG_MISMATCH"),
            (metrics_complete, "METRICS_INCOMPLETE"),
            (batch_match, "BLOCKED_MIXED_BATCH_PROFILE"),
        ):
            if not condition:
                reasons.append(reason)
        rows.append(
            {
                "entry_id": entry["entry_id"],
                "model_id": entry["model_id"],
                "entry_type": entry["entry_type"],
                "expected": True,
                "found": found,
                "formal": formal,
                "completed": completed,
                "protocol_match": protocol_match,
                "loss_match": loss_match,
                "config_match": config_match,
                "metrics_complete": metrics_complete,
                "batch_profile_match": batch_match,
                "reference_valid": None,
                "ready": ready,
                "blocked_reason": None if ready else reasons,
            }
        )
    report = {
        "schema_version": "e5_result_readiness_v1",
        "status": "READY" if all(row["ready"] for row in rows) else "NOT_READY",
        "require_complete": True,
        "expected_total_entries": 29,
        "ready_entries": sum(row["ready"] for row in rows),
        "output_root": str(root),
        **expected_batch_identity,
        "entries": rows,
    }
    if report_path is not None:
        atomic_write_json(Path(report_path), report)
    return report
