"""Explicit metadata helpers for the active Original Batch4 scope26."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


CURRENT_SCOPE26_ID = "benchmark_v2_batch4_scope26_seed2026"
CURRENT_TRAINING_PROFILE_ID = "uniform_train_batch4_v1"
CURRENT_EXPERIMENT_PROFILE_ID = "default_benchmark_v1"
CURRENT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3]
    / "custom_models/docs/benchmark_v2/BATCH4/CURRENT_BATCH4_SCOPE26_MANIFEST.json"
)


class OriginalScope26Error(ValueError):
    pass


def load_current_scope_manifest(path: str | Path | None = None) -> dict[str, Any]:
    payload = json.loads(Path(path or CURRENT_MANIFEST_PATH).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OriginalScope26Error("Current Original scope26 manifest must be an object.")
    return payload


def current_scope_entries(manifest: Mapping[str, Any] | None = None) -> tuple[Mapping[str, Any], ...]:
    value = (manifest or load_current_scope_manifest()).get("entries")
    if not isinstance(value, list):
        raise OriginalScope26Error("Current Original scope26 manifest has no entries list.")
    return tuple(value)


def current_scope_entry(model_id: str, manifest: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    for entry in current_scope_entries(manifest):
        if entry.get("model_id") == model_id:
            return entry
    raise OriginalScope26Error(f"Model is not in current Original scope26: {model_id}")


def is_current_scope26_request(
    *, model_id: str, formal_scope_id: str | None,
    experiment_profile: str | None, training_profile: str | None,
    trainable: bool, manifest: Mapping[str, Any] | None = None,
) -> bool:
    if formal_scope_id != CURRENT_SCOPE26_ID:
        return False
    if experiment_profile not in (None, "", "default", CURRENT_EXPERIMENT_PROFILE_ID):
        return False
    if training_profile != CURRENT_TRAINING_PROFILE_ID:
        return False
    try:
        entry = current_scope_entry(model_id, manifest)
    except OriginalScope26Error:
        return False
    return entry.get("entry_type") == ("TRAINABLE" if trainable else "EVALUATE_ONLY")


def validate_current_scope26_request(**kwargs: Any) -> None:
    if kwargs.get("formal_scope_id") != CURRENT_SCOPE26_ID:
        return
    if not is_current_scope26_request(**kwargs):
        raise OriginalScope26Error("Invalid current Original scope26 request metadata.")


def apply_current_scope_identity(
    runtime: Any, *, run_id: str, output_root: str | Path,
    formal_scope_id: str | None, source_revision: str | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> Any:
    del source_revision
    if formal_scope_id != CURRENT_SCOPE26_ID:
        return runtime
    active = manifest or load_current_scope_manifest()
    entry = current_scope_entry(runtime.model_id, active)
    expected_root = (Path(__file__).resolve().parents[3] / entry["output_root"]).resolve()
    if Path(output_root).resolve() != expected_root:
        raise OriginalScope26Error("Current Original output root conflicts with the manifest.")
    if run_id != entry["run_id"]:
        raise OriginalScope26Error(f"Current Original run-id conflict for {runtime.model_id}.")
    formal_training = entry["entry_type"] == "TRAINABLE"
    runtime.effective_config.update({
        "scope_id": CURRENT_SCOPE26_ID,
        "entry_id": entry["entry_id"],
        "entry_type": entry["entry_type"],
        "model_id": runtime.model_id,
        "run_id": run_id,
        "output_root": entry["output_root"],
        "formal_training": formal_training,
    })
    return runtime


__all__ = [
    "CURRENT_EXPERIMENT_PROFILE_ID", "CURRENT_MANIFEST_PATH",
    "CURRENT_SCOPE26_ID", "CURRENT_TRAINING_PROFILE_ID", "OriginalScope26Error",
    "apply_current_scope_identity", "current_scope_entries", "current_scope_entry",
    "is_current_scope26_request", "load_current_scope_manifest",
    "validate_current_scope26_request",
]
