"""Deterministic identity-field generator for the active Batch4 scopes.

The scope contracts, ordinals, and frozen run ids are inputs.  Only fields
derived from the current source/config/protocol identity are regenerated.
Writes use canonical JSON with ``allow_nan=False`` and never touch result
directories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .model_factories.registry import get_model_factory
from .model_source_identity import canonical_model_source_identity
from .original_scope26 import CURRENT_SCOPE26_ID, current_model_config_hash
from .precision import expected_model_precision_identity
from .protocol import load_protocol
from .runtime import PROJECT_ROOT
from .training_profiles import batch_identity, load_training_profile, stable_hash


ORIGINAL_MANIFEST_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/BATCH4/CURRENT_BATCH4_SCOPE26_MANIFEST.json"
ORIGINAL_RUN_MAP_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/BATCH4/ORIGINAL_SCOPE26_RUN_ID_MAP.json"
ORIGINAL_EXCLUSIONS_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/BATCH4/CURRENT_SCOPE_EXCLUSIONS.json"
E5_MANIFEST_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/E5/E5_SCOPE27_VARIANT_MANIFEST.json"
E5_RUN_MAP_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/E5/E5_SCOPE27_RUN_ID_MAP.json"
E5_POLICY_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/E5/E5_SCOPE27_READINESS_POLICY.json"
E5_ACTIVE_SCOPE_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/E5/E5_ACTIVE_SCOPE.json"
E5_OUTPUT_ROOT = "custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026"
E5_LEGACY_OUTPUT_ROOT = "custom_models/results/benchmark_v2/common_loss_architecture_seed2026"
E5_A8_REFERENCE_ID = "STMGPrompt_A8_loss_msa_hybrid_bs4_seed2026_reference"
E5_SCOPE_ID = "e5_batch4_scope27_seed2026"
TRAINING_PROFILE_ID = "uniform_train_batch4_v1"
E5_LOSS_ID = "masked_score_aligned_hybrid"
E5_TRAINABLE = (
    "gru", "dlinear", "lightts", "tide", "transformer", "patchtst",
    "itransformer", "timexer", "timesnet", "micn", "wpmixer",
    "multipatchformer", "timemixer", "tsmixer", "frets", "crossformer",
    "timefilter", "gcn", "stgcn", "dcrnn", "graph_wavenet", "mtgnn",
    "agcrn", "stid",
)
E5_EVALUATE_ONLY = ("persistence", "moving_average")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _source_identity(model_id: str) -> dict[str, Any]:
    identity = canonical_model_source_identity(model_id)
    records = list(identity["source_closure_files"])
    return {
        "source_identity_schema_version": identity["source_identity_schema_version"],
        "source_closure_paths": [record["path"] for record in records],
        "source_closure_files": records,
        "canonical_combined_hash": identity["canonical_combined_hash"],
        "source_closure_cycles": identity.get("source_closure_cycles", []),
    }


def _config_identity(model_id: str) -> dict[str, str]:
    protocol = load_protocol()
    config = get_model_factory(model_id).resolve_config(protocol, run_mode="formal")
    return {"config_hash": stable_hash(config)}


def generate_original() -> dict[str, Any]:
    payload = _load(ORIGINAL_MANIFEST_PATH)
    if payload.get("scope_id") != CURRENT_SCOPE26_ID:
        raise ValueError("The current Original manifest is not scope26.")
    for entry in payload.get("entries", []):
        model_id = str(entry["model_id"])
        entry["source_identity"] = _source_identity(model_id)
        entry["model_config_identity"] = _config_identity(model_id)
    run_map = _load(ORIGINAL_RUN_MAP_PATH)
    run_map["scope_id"] = CURRENT_SCOPE26_ID
    run_map["output_root"] = payload["output_root"]
    run_map["entries"] = [
        {
            "ordinal": entry["ordinal"],
            "model_id": entry["model_id"],
            "run_id": entry["run_id"],
            "output_root": entry["output_root"],
        }
        for entry in payload["entries"]
    ]
    _write(ORIGINAL_MANIFEST_PATH, payload)
    _write(ORIGINAL_RUN_MAP_PATH, run_map)
    exclusions = _load(ORIGINAL_EXCLUSIONS_PATH)
    exclusions["scope_id"] = CURRENT_SCOPE26_ID
    exclusions["output_root"] = payload["output_root"]
    exclusions["training_profile_id"] = payload["training_profile_id"]
    _write(ORIGINAL_EXCLUSIONS_PATH, exclusions)
    return {
        "scope_id": CURRENT_SCOPE26_ID,
        "manifest": str(ORIGINAL_MANIFEST_PATH),
        "run_map": str(ORIGINAL_RUN_MAP_PATH),
        "exclusions": str(ORIGINAL_EXCLUSIONS_PATH),
        "entry_count": len(payload.get("entries", [])),
    }


def _e5_loss_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload.get("loss_identity") or {})
    if value.get("loss_id") != E5_LOSS_ID:
        raise ValueError("The active E5 manifest does not declare the Batch4 loss.")
    return value


def generate_e5() -> dict[str, Any]:
    payload = _load(E5_MANIFEST_PATH)
    if payload.get("scope_id") != E5_SCOPE_ID:
        raise ValueError("The current E5 manifest is not scope27.")
    profile = load_training_profile(TRAINING_PROFILE_ID)
    if profile is None:
        raise ValueError(f"Training profile is unavailable: {TRAINING_PROFILE_ID}")
    original = _load(ORIGINAL_MANIFEST_PATH)
    original_entries = {
        entry["model_id"]: entry for entry in original.get("entries", [])
    }
    protocol = load_protocol()
    payload["output_root"] = E5_OUTPUT_ROOT
    payload["legacy_output_root"] = E5_LEGACY_OUTPUT_ROOT
    payload["legacy_output_policy"] = "LEGACY_OR_HISTORICAL_READ_ONLY_NOT_CURRENT_BATCH4_OUTPUT"
    payload["training_profile_id"] = TRAINING_PROFILE_ID
    payload["training_profile_hash"] = profile.profile_hash
    payload["batch_identity"] = batch_identity(TRAINING_PROFILE_ID)
    payload["benchmark_protocol_hash"] = protocol.protocol_hash
    payload["source_identity_schema_version"] = "model_source_identity_v1"
    payload["e5_loss_id"] = E5_LOSS_ID
    payload["counts"] = {
        "trainable": 24,
        "evaluate_only": 2,
        "a8_reference": 1,
        "total_evidence": 27,
    }
    active_entries: list[dict[str, Any]] = []
    for entry in payload.get("entries", []):
        model_id = str(entry["model_id"])
        if model_id == "st_mgprompt_a8":
            entry["entry_id"] = E5_A8_REFERENCE_ID
            entry["e5_run_id"] = E5_A8_REFERENCE_ID
            entry["output_root"] = None
            entry["expected_output_root"] = None
            entry["reference_id"] = E5_A8_REFERENCE_ID
            active_entries.append(entry)
            continue
        if model_id not in set(E5_TRAINABLE) | set(E5_EVALUATE_ONLY):
            continue
        original_entry = original_entries[model_id]
        identity = _source_identity(model_id)
        entry["source_identity"] = identity
        entry["base_model_source_hash"] = identity["canonical_combined_hash"]
        entry["base_model_source_closure_hash"] = identity["canonical_combined_hash"]
        entry["base_model_config_hash"] = _config_identity(model_id)["config_hash"]
        entry["precision_identity"] = expected_model_precision_identity(
            model_id, TRAINING_PROFILE_ID
        )
        entry["output_root"] = E5_OUTPUT_ROOT
        entry["expected_output_root"] = E5_OUTPUT_ROOT
        entry["expected_run_dir"] = f"{E5_OUTPUT_ROOT}/{entry['e5_run_id']}"
        entry["loss_id"] = E5_LOSS_ID
        entry["training_profile_id"] = TRAINING_PROFILE_ID
        entry["training_profile_hash"] = profile.profile_hash
        entry["protocol_hash"] = protocol.protocol_hash
        entry["entry_type"] = (
            "TRAIN_COMMON_LOSS"
            if model_id in E5_TRAINABLE
            else "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC"
        )
        entry["ordinal"] = len(active_entries) + 1
        entry["original_entry_id"] = original_entry.get("entry_id")
        active_entries.append(entry)
    # Preserve the protocol-defined denominator order: 24 trainable, two
    # evaluate-only entries, then the read-only A8 reference.
    order = {
        model_id: index
        for index, model_id in enumerate(
            (*E5_TRAINABLE, *E5_EVALUATE_ONLY, "st_mgprompt_a8")
        )
    }
    active_entries.sort(key=lambda row: order[row["model_id"]])
    for ordinal, entry in enumerate(active_entries, 1):
        entry["ordinal"] = ordinal
    payload["entries"] = active_entries
    payload["loss_identity"] = _e5_loss_identity(payload)
    a8_identity = dict(payload.get("a8_reference_identity") or {})
    a8_identity["reference_id"] = E5_A8_REFERENCE_ID
    a8_identity["read_only"] = True
    a8_identity["retrained_in_e5"] = False
    a8_identity["checkpoint_copied"] = False
    a8_identity["metrics_copied"] = False
    payload["a8_reference_identity"] = a8_identity

    run_map = _load(E5_RUN_MAP_PATH)
    run_map.update(
        {
            "scope_id": E5_SCOPE_ID,
            "source_identity_schema_version": "model_source_identity_v1",
            "training_profile_id": TRAINING_PROFILE_ID,
            "training_profile_hash": profile.profile_hash,
            "output_root": E5_OUTPUT_ROOT,
            "legacy_output_root": E5_LEGACY_OUTPUT_ROOT,
            "a8_reference_id": E5_A8_REFERENCE_ID,
            "unique_run_id_count": len(active_entries),
            "entries": [
                {
                    "ordinal": entry["ordinal"],
                    "entry_id": entry["entry_id"],
                    "model_id": entry["model_id"],
                    "e5_run_id": entry["e5_run_id"],
                    "entry_type": entry["entry_type"],
                    "output_root": entry.get("output_root"),
                }
                for entry in active_entries
            ],
            "runs": {
                entry["model_id"]: entry["e5_run_id"]
                for entry in active_entries
            },
        }
    )
    policy = _load(E5_POLICY_PATH)
    policy.update(
        {
            "scope_id": E5_SCOPE_ID,
            "status": "ACTIVE_POLICY",
            "output_root": E5_OUTPUT_ROOT,
            "legacy_output_root": E5_LEGACY_OUTPUT_ROOT,
            "training_profile_id": TRAINING_PROFILE_ID,
            "training_profile_hash": profile.profile_hash,
            "denominators": {
                "trainable": "24/24",
                "evaluate_only": "2/2",
                "a8_reference": "1/1",
                "full_e5": "27/27",
            },
            "hardware_artifact_gate": "EXACT_PASS_REQUIRED_FOR_24_TRAINABLE",
            "a8_reference_id": E5_A8_REFERENCE_ID,
            "legacy_policy": "LEGACY_OR_HISTORICAL_READ_ONLY_NOT_CURRENT_BATCH4_OUTPUT",
        }
    )
    _write(E5_MANIFEST_PATH, payload)
    _write(E5_RUN_MAP_PATH, run_map)
    _write(E5_POLICY_PATH, policy)
    pointer = _load(E5_ACTIVE_SCOPE_PATH)
    pointer["active_manifest_schema_version"] = payload["schema_version"]
    pointer["source_identity_schema_version"] = "model_source_identity_v1"
    _write(E5_ACTIVE_SCOPE_PATH, pointer)
    return {
        "scope_id": E5_SCOPE_ID,
        "manifest": str(E5_MANIFEST_PATH),
        "run_map": str(E5_RUN_MAP_PATH),
        "readiness_policy": str(E5_POLICY_PATH),
        "output_root": E5_OUTPUT_ROOT,
        "entry_count": len(active_entries),
        "a8_reference_id": E5_A8_REFERENCE_ID,
    }


def generate_all() -> dict[str, Any]:
    return {"original": generate_original(), "e5": generate_e5()}


__all__ = ["generate_all", "generate_e5", "generate_original"]
