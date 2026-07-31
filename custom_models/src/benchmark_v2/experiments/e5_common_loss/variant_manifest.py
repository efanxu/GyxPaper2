from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from ...configs import resolve_moving_average_config, resolve_persistence_config
from ...hardware_preflight import _RESOLVERS
from ...protocol import load_protocol
from ...registry import load_registry
from ...runtime import PROJECT_ROOT
from ...training_profiles import batch_identity, load_training_profile
from .contracts import (
    A8_REFERENCE_ID,
    BATCH4_A8_REFERENCE_ID,
    BATCH4_FORMAL_OUTPUT_ROOT_RELATIVE,
    FORMAL_OUTPUT_ROOT_RELATIVE,
    NONTRAINABLE_MODELS,
    TRAINABLE_MODELS,
    validate_registry_contract,
)
from .loss_profile import CLI_PROFILE_ID, get_profile_metadata, stable_hash


RUNBOOK_ROOT = PROJECT_ROOT / "custom_models/docs/benchmark_v2"
RUNBOOK_DIRS = ("E1_A", "E1_B", "E2_A", "E2_B", "E2_C", "E2_D", "E3_B", "E3_C")
RUN_ID_PATTERN = re.compile(
    r"--model\s+([a-z0-9_]+).*?--run-id\s+['\"]?([A-Za-z0-9_.-]+_seed2026)",
    re.IGNORECASE,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_base_run_ids() -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    for dirname in RUNBOOK_DIRS:
        path = RUNBOOK_ROOT / dirname / f"{dirname}_RUNBOOK.md"
        text = path.read_text(encoding="utf-8")
        for model_id, run_id in RUN_ID_PATTERN.findall(text):
            key = model_id.casefold()
            candidate = {"base_run_id": run_id, "source_runbook": str(path)}
            previous = found.get(key)
            if previous and previous["base_run_id"] != run_id:
                raise ValueError(
                    f"Inconsistent frozen run-id for {key}: "
                    f"{previous['base_run_id']} vs {run_id}"
                )
            found[key] = candidate
    expected = set(TRAINABLE_MODELS) | set(NONTRAINABLE_MODELS)
    missing = sorted(expected - set(found))
    if missing:
        raise ValueError(f"Missing frozen base run-id(s): {missing}")
    for model_id, value in found.items():
        if model_id in expected and not value["base_run_id"].endswith("_seed2026"):
            raise ValueError(f"Frozen run-id does not end in _seed2026: {value}")
    return {key: found[key] for key in sorted(expected)}


def e5_run_id(
    base_run_id: str, training_profile: str | None = None
) -> str:
    if not base_run_id.endswith("_seed2026"):
        raise ValueError(f"Invalid frozen base run-id: {base_run_id}")
    suffix = (
        "_loss_msa_hybrid_bs4_seed2026"
        if training_profile is not None
        else "_loss_msa_hybrid_seed2026"
    )
    return base_run_id[: -len("_seed2026")] + suffix


def _base_config(model_id: str) -> dict[str, Any]:
    protocol = load_protocol()
    if model_id == "persistence":
        return resolve_persistence_config(protocol, run_mode="formal")
    if model_id == "moving_average":
        return resolve_moving_average_config(protocol, run_mode="formal")
    return _RESOLVERS[model_id](protocol, run_mode="formal")


def _source_hash(entry) -> str:
    path = PROJECT_ROOT / str(entry.source_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return file_sha256(path)


def build_variant_manifest(
    training_profile: str | None = None,
) -> dict[str, Any]:
    registry = load_registry()
    contract = validate_registry_contract(registry)
    if contract["status"] != "PASS":
        raise ValueError(f"E5 registry contract failed: {contract}")
    base_ids = extract_base_run_ids()
    profile = get_profile_metadata(CLI_PROFILE_ID)
    profile_identity = batch_identity(training_profile)
    output_root = PROJECT_ROOT / (
        BATCH4_FORMAL_OUTPUT_ROOT_RELATIVE
        if training_profile is not None
        else FORMAL_OUTPUT_ROOT_RELATIVE
    )
    entries = []
    for model_id in NONTRAINABLE_MODELS + TRAINABLE_MODELS:
        entry = registry.get(model_id)
        base_run_id = base_ids[model_id]["base_run_id"]
        trainable = model_id in TRAINABLE_MODELS
        config = _base_config(model_id)
        entries.append(
            {
                "entry_id": f"e5_{model_id}",
                "model_id": model_id,
                "display_name": entry.display_name,
                "category": entry.category,
                "entry_type": (
                    "TRAIN_COMMON_LOSS"
                    if trainable
                    else "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC"
                ),
                "base_run_id": base_run_id,
                "base_run_id_source": base_ids[model_id]["source_runbook"],
                "e5_run_id": e5_run_id(base_run_id, training_profile),
                "base_model_config_hash": stable_hash(config),
                "base_model_source_hash": _source_hash(entry),
                "base_model_source_closure_hash": _source_hash(entry),
                "training_mode": "TRAIN" if trainable else "EVALUATE_ONLY",
                "trained_with_common_loss": bool(trainable),
                "diagnostic_loss_applied": True,
                "common_loss_evaluation_applied": True,
                "training_loss": (
                    profile["loss_id"] if trainable else "NOT_APPLICABLE"
                ),
                "loss_id": profile["loss_id"],
                "loss_source_hash": profile["loss_source_hash"],
                "loss_profile_hash": profile["loss_profile_hash"],
                "e5_common_loss_protocol_hash": profile[
                    "e5_common_loss_protocol_hash"
                ],
                "uses_physical_support": bool(entry.requires_graph)
                and model_id not in {"mtgnn", "agcrn", "stid"},
                "graph_policy": config.get("graph_policy", config.get("graph_support_names")),
                "node_identity_policy": config.get("node_identity_policy"),
                "temporal_identity_policy": config.get("temporal_identity_policy"),
                "expected_output_root": str(output_root),
                "expected_run_dir": str(
                    output_root / e5_run_id(base_run_id, training_profile)
                ),
                "formal_status": "NOT_RUN",
                "preflight_status": "NOT_RUN",
                "reference_status": None,
                "comparison_role": (
                    "TRAINABLE_ARCHITECTURE"
                    if trainable
                    else "NON_TRAINABLE_REFERENCE"
                ),
                **profile_identity,
            }
        )
    entries.append(
        {
            "entry_id": (
                BATCH4_A8_REFERENCE_ID
                if training_profile is not None
                else A8_REFERENCE_ID
            ),
            "model_id": "st_mgprompt_a8",
            "display_name": "ST-MGPrompt A8 (w/o MS-MG-DWU)",
            "category": "st_mgprompt",
            "entry_type": "REFERENCE_ONLY_FORMAL_A8",
            "base_run_id": "component_ablation_fixed_dual_seed2026/A8",
            "e5_run_id": (
                BATCH4_A8_REFERENCE_ID
                if training_profile is not None
                else A8_REFERENCE_ID
            ),
            "training_mode": "REFERENCE_ONLY",
            "trained_with_common_loss": True,
            "retrained_in_e5": False,
            "diagnostic_loss_applied": False,
            "loss_id": profile["loss_id"],
            "loss_source_hash": profile["loss_source_hash"],
            "loss_profile_hash": profile["loss_profile_hash"],
            "e5_common_loss_protocol_hash": profile[
                "e5_common_loss_protocol_hash"
            ],
            "expected_output_root": None,
            "formal_status": "REFERENCE_ONLY",
            "preflight_status": "NOT_APPLICABLE",
            "reference_status": "PENDING_VALIDATION",
            "comparison_role": "ST_MGPROMPT_STRUCTURE_REFERENCE",
            **profile_identity,
        }
    )
    counts = {
        "TRAIN_COMMON_LOSS": sum(
            item["entry_type"] == "TRAIN_COMMON_LOSS" for item in entries
        ),
        "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC": sum(
            item["entry_type"] == "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC"
            for item in entries
        ),
        "REFERENCE_ONLY_FORMAL_A8": sum(
            item["entry_type"] == "REFERENCE_ONLY_FORMAL_A8" for item in entries
        ),
        "TOTAL": len(entries),
    }
    expected_counts = {
        "TRAIN_COMMON_LOSS": 26,
        "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC": 2,
        "REFERENCE_ONLY_FORMAL_A8": 1,
        "TOTAL": 29,
    }
    if counts != expected_counts:
        raise ValueError(f"E5 variant count mismatch: {counts}")
    return {
        "schema_version": "e5_variant_manifest_v1",
        "profile_id": profile["profile_id"],
        "cli_profile_id": CLI_PROFILE_ID,
        **profile_identity,
        "counts": counts,
        "registry_contract": contract,
        "entries": entries,
    }


def build_run_id_map(training_profile: str | None = None) -> dict[str, Any]:
    base = extract_base_run_ids()
    return {
        "schema_version": "e5_run_id_map_v1",
        "rule": "replace terminal _seed2026 with _loss_msa_hybrid_seed2026",
        "runs": {
            model_id: {
                **value,
                "e5_run_id": e5_run_id(
                    value["base_run_id"], training_profile
                ),
                "mode": (
                    "TRAIN"
                    if model_id in TRAINABLE_MODELS
                    else "EVALUATE_ONLY"
                ),
            }
            for model_id, value in base.items()
        },
        "a8_reference_id": (
            BATCH4_A8_REFERENCE_ID
            if training_profile is not None
            else A8_REFERENCE_ID
        ),
        **batch_identity(training_profile),
    }
