from __future__ import annotations

from pathlib import Path
from typing import Any

from .loss_profile import (
    CLI_PROFILE_ID,
    DEFAULT_PROFILE_ID,
    get_profile_metadata,
    loss_for_profile,
    stable_hash,
)


def canonical_base_model_config_hash(model_id: str) -> str:
    from ...configs import resolve_moving_average_config, resolve_persistence_config
    from ...hardware_preflight import _RESOLVERS
    from ...protocol import load_protocol

    protocol = load_protocol()
    if model_id == "persistence":
        config = resolve_persistence_config(protocol, run_mode="formal")
    elif model_id == "moving_average":
        config = resolve_moving_average_config(protocol, run_mode="formal")
    else:
        config = _RESOLVERS[model_id](protocol, run_mode="formal")
    return stable_hash(config)


def normalize_profile(profile: str | None) -> str:
    if profile in (None, "", DEFAULT_PROFILE_ID):
        return DEFAULT_PROFILE_ID
    if profile != CLI_PROFILE_ID:
        get_profile_metadata(profile)
    return profile


def apply_experiment_profile(
    runtime,
    profile: str | None,
    *,
    run_id: str | None = None,
    output_root: str | Path | None = None,
    preflight_identity: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
):
    selected = normalize_profile(profile)
    if selected == DEFAULT_PROFILE_ID:
        return runtime
    metadata = get_profile_metadata(selected)
    base = dict(runtime.effective_config)
    overlay = {
        "experiment_profile_id": metadata["profile_id"],
        "loss": {
            "id": metadata["loss_id"],
            "source_hash": metadata["loss_source_hash"],
            "profile_hash": metadata["loss_profile_hash"],
        },
        "run_id": run_id,
        "output_root": None if output_root is None else str(Path(output_root)),
        "preflight_identity": preflight_identity,
        "provenance": {
            "base_model_id": runtime.model_id,
            "base_model_config_hash": canonical_base_model_config_hash(
                runtime.model_id
            ),
            "base_model_source_closure_hash": base.get(
                "source_closure_hash",
                base.get("source_hash", base.get("upstream_source_sha256")),
            ),
            "e5_common_loss_protocol_hash": metadata[
                "e5_common_loss_protocol_hash"
            ],
            **(provenance or {}),
        },
    }
    experiment_material = {**base, **overlay}
    overlay["experiment_config_hash"] = stable_hash(experiment_material)
    runtime.effective_config.update(overlay)
    return runtime


def runtime_profile(runtime) -> str:
    return (
        CLI_PROFILE_ID
        if runtime.effective_config.get("experiment_profile_id")
        else DEFAULT_PROFILE_ID
    )


def runtime_loss(runtime):
    return loss_for_profile(runtime_profile(runtime))
