from __future__ import annotations

from pathlib import Path
from typing import Any

from .loss_profile import CLI_PROFILE_ID, DEFAULT_PROFILE_ID, get_profile_metadata, loss_for_profile


def normalize_profile(profile: str | None) -> str:
    if profile in (None, "", DEFAULT_PROFILE_ID):
        return DEFAULT_PROFILE_ID
    get_profile_metadata(profile)
    return str(profile)


def apply_experiment_profile(
    runtime, profile: str | None, *, run_id: str | None = None,
    output_root: str | Path | None = None,
    preflight_identity: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
):
    selected = normalize_profile(profile)
    if selected == DEFAULT_PROFILE_ID:
        return runtime
    metadata = get_profile_metadata(selected)
    preflight = dict(preflight_identity or {})
    runtime.effective_config.update({
        "experiment_profile_id": metadata["profile_id"],
        "loss": {"id": metadata["loss_id"], "source_function": metadata["source_function"]},
        "loss_id": metadata["loss_id"],
        "run_id": run_id,
        "output_root": None if output_root is None else str(Path(output_root)),
        "preflight_metadata": {key: preflight.get(key) for key in (
            "status", "scope_id", "model_id", "run_id", "batch_size",
            "precision", "device", "forward_pass", "backward_pass",
            "finite", "output_shape", "created_at",
        ) if key in preflight},
        "scope_id": preflight.get("scope_id") or dict(provenance or {}).get("active_scope_id"),
        "training_batch_profile_id": preflight.get("training_profile_id"),
        "provenance": {"base_model_id": runtime.model_id, **dict(provenance or {})},
    })
    return runtime


def runtime_profile(runtime) -> str:
    return CLI_PROFILE_ID if runtime.effective_config.get("experiment_profile_id") else DEFAULT_PROFILE_ID


def runtime_loss(runtime):
    return loss_for_profile(runtime_profile(runtime))
