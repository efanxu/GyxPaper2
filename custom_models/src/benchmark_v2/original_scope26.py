"""Identity helpers for the active Original Batch4 scope26.

This module deliberately reads only the current Original manifest.  The E5
scope27 manifest and its legacy run maps are not execution inputs here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .model_source_identity import canonical_model_source_identity
from .registry import load_registry
from .training_profiles import stable_hash


CURRENT_SCOPE26_ID = "benchmark_v2_batch4_scope26_seed2026"
CURRENT_TRAINING_PROFILE_ID = "uniform_train_batch4_v1"
CURRENT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3]
    / "custom_models/docs/benchmark_v2/BATCH4/CURRENT_BATCH4_SCOPE26_MANIFEST.json"
)


class OriginalScope26Error(ValueError):
    """Raised when current Original identity cannot be established."""


def load_current_scope_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(path or CURRENT_MANIFEST_PATH).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OriginalScope26Error("Current Original scope26 manifest must be an object.")
    return payload


def current_scope_entries(
    manifest: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    value = (manifest or load_current_scope_manifest()).get("entries")
    if not isinstance(value, list):
        raise OriginalScope26Error("Current Original scope26 manifest has no entries list.")
    return tuple(value)


def current_scope_entry(
    model_id: str,
    manifest: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    for entry in current_scope_entries(manifest):
        if entry.get("model_id") == model_id:
            return entry
    raise OriginalScope26Error(f"Model is not in current Original scope26: {model_id}")


def is_current_scope26_request(
    *,
    model_id: str,
    formal_scope_id: str | None,
    experiment_profile: str | None,
    training_profile: str | None,
    trainable: bool,
    manifest: Mapping[str, Any] | None = None,
) -> bool:
    if formal_scope_id != CURRENT_SCOPE26_ID:
        return False
    if experiment_profile not in (None, "", "default"):
        return False
    if training_profile != CURRENT_TRAINING_PROFILE_ID:
        return False
    try:
        entry = current_scope_entry(model_id, manifest)
    except OriginalScope26Error:
        return False
    return entry.get("entry_type") == ("TRAINABLE" if trainable else "EVALUATE_ONLY")


def validate_current_scope26_request(
    *,
    model_id: str,
    formal_scope_id: str | None,
    experiment_profile: str | None,
    training_profile: str | None,
    trainable: bool,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    if formal_scope_id != CURRENT_SCOPE26_ID:
        return
    if not is_current_scope26_request(
        model_id=model_id,
        formal_scope_id=formal_scope_id,
        experiment_profile=experiment_profile,
        training_profile=training_profile,
        trainable=trainable,
        manifest=manifest,
    ):
        raise OriginalScope26Error(
            "Invalid current Original scope26 request binding: "
            f"scope={formal_scope_id!r}, model={model_id!r}, "
            f"experiment_profile={experiment_profile!r}, "
            f"training_profile={training_profile!r}, trainable={trainable!r}"
        )


def current_manifest_hash(path: str | Path | None = None) -> str:
    payload = load_current_scope_manifest(path)
    material = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def current_model_config_hash(model_id: str) -> str:
    """Hash the current model resolver without consulting another manifest."""

    from .configs import resolve_moving_average_config, resolve_persistence_config
    from .hardware_preflight import _RESOLVERS
    from .protocol import load_protocol

    protocol = load_protocol()
    if model_id == "persistence":
        config = resolve_persistence_config(protocol, run_mode="formal")
    elif model_id == "moving_average":
        config = resolve_moving_average_config(protocol, run_mode="formal")
    else:
        try:
            config = _RESOLVERS[model_id](protocol, run_mode="formal")
        except KeyError as exc:
            raise OriginalScope26Error(
                f"No current model config resolver for {model_id!r}"
            ) from exc
    return stable_hash(config)


def resolve_source_revision(
    *,
    project_root: str | Path | None = None,
    explicit_source_revision: str | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Return real Git identity, or require explicit identity outside Git."""

    root = Path(project_root or CURRENT_MANIFEST_PATH.parents[4]).resolve()
    git_dir = root / ".git"
    if git_dir.exists():
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise OriginalScope26Error(
                "Git worktree is present but git rev-parse HEAD failed."
            )
        revision = completed.stdout.strip()
        if not revision:
            raise OriginalScope26Error("git rev-parse HEAD returned an empty revision.")
        if explicit_source_revision and explicit_source_revision != revision:
            raise OriginalScope26Error(
                "Explicit source revision does not match git rev-parse HEAD."
            )
        return {"git_commit": revision, "source_revision_type": "git"}

    if not explicit_source_revision:
        raise OriginalScope26Error(
            "No .git directory is available; --source-revision is required."
        )
    policy = (manifest or load_current_scope_manifest()).get(
        "source_revision_policy", {}
    )
    declared = policy.get("declared_source_revision")
    if declared and declared != explicit_source_revision:
        raise OriginalScope26Error(
            "Explicit source revision does not match the active manifest declaration."
        )
    return {
        "git_commit": explicit_source_revision,
        "source_revision_type": "explicit_no_git",
    }


def apply_current_scope_identity(
    runtime: Any,
    *,
    run_id: str,
    output_root: str | Path,
    formal_scope_id: str | None,
    source_revision: str | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> Any:
    """Attach current Original identity to a runtime effective config."""

    if formal_scope_id != CURRENT_SCOPE26_ID:
        return runtime
    active = manifest or load_current_scope_manifest()
    entry = current_scope_entry(runtime.model_id, active)
    expected_root = (
        Path(__file__).resolve().parents[3] / entry["output_root"]
    ).resolve()
    actual_root = Path(output_root).resolve()
    if expected_root != actual_root:
        raise OriginalScope26Error(
            f"Current Original output root mismatch: {actual_root} != {expected_root}"
        )
    if run_id != entry["run_id"]:
        raise OriginalScope26Error(
            f"Current Original run-id mismatch for {runtime.model_id}: "
            f"{run_id!r} != {entry['run_id']!r}"
        )

    source = canonical_model_source_identity(runtime.model_id)
    expected_source = entry["source_identity"]["canonical_combined_hash"]
    if source["canonical_combined_hash"] != expected_source:
        raise OriginalScope26Error(
            f"Current Original source identity mismatch for {runtime.model_id}."
        )
    config_hash = current_model_config_hash(runtime.model_id)
    expected_config = entry["model_config_identity"]["config_hash"]
    if config_hash != expected_config:
        raise OriginalScope26Error(
            f"Current Original model config identity mismatch for {runtime.model_id}."
        )
    revision = resolve_source_revision(
        explicit_source_revision=source_revision, manifest=active
    )
    formal_training = entry["entry_type"] == "TRAINABLE"
    root_relative = entry["output_root"]
    provenance = dict(runtime.effective_config.get("provenance") or {})
    provenance.update(
        {
            "active_scope_id": CURRENT_SCOPE26_ID,
            "base_model_id": runtime.model_id,
            "base_model_config_hash": config_hash,
            "base_model_source_closure_hash": source["canonical_combined_hash"],
            "base_model_source_identity_schema_version": source[
                "source_identity_schema_version"
            ],
            "base_model_source_closure_files": source["source_closure_files"],
        }
    )
    runtime.effective_config.update(
        {
            "active_scope_id": CURRENT_SCOPE26_ID,
            "current_scope_manifest_hash": current_manifest_hash(),
            "current_scope_entry_id": entry["entry_id"],
            "run_id": run_id,
            "output_root": root_relative,
            "model_config_hash": config_hash,
            "source_closure_hash": source["canonical_combined_hash"],
            "source_identity": source,
            "source_revision": revision,
            "formal_training": formal_training,
            "provenance": provenance,
        }
    )
    return runtime


__all__ = [
    "CURRENT_MANIFEST_PATH",
    "CURRENT_SCOPE26_ID",
    "CURRENT_TRAINING_PROFILE_ID",
    "OriginalScope26Error",
    "apply_current_scope_identity",
    "current_manifest_hash",
    "current_model_config_hash",
    "current_scope_entries",
    "current_scope_entry",
    "is_current_scope26_request",
    "load_current_scope_manifest",
    "resolve_source_revision",
    "validate_current_scope26_request",
]
