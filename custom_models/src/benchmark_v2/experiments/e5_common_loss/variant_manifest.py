from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...model_factories.registry import get_model_factory
from ...precision import expected_model_precision_identity
from ...protocol import load_protocol
from ...registry import load_registry
from ...runtime import PROJECT_ROOT
from ...training_profiles import batch_identity
from .contracts import BATCH4_A8_REFERENCE_ID, BATCH4_FORMAL_OUTPUT_ROOT_RELATIVE, NONTRAINABLE_MODELS, TRAINABLE_MODELS, validate_registry_contract
from .loss_profile import CLI_PROFILE_ID, get_profile_metadata


RUNBOOK_ROOT = PROJECT_ROOT / "custom_models/docs/benchmark_v2"
RUNBOOK_DIRS = ("E1_A", "E1_B", "E2_A", "E2_B", "E2_C", "E2_D", "E3_B", "E3_C")
RUN_ID_PATTERN = re.compile(r"--model\s+([a-z0-9_]+).*?--run-id\s+['\"]?([A-Za-z0-9_.-]+_seed2026)", re.IGNORECASE)


def extract_base_run_ids() -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    expected = set(TRAINABLE_MODELS) | set(NONTRAINABLE_MODELS)
    for dirname in RUNBOOK_DIRS:
        path = RUNBOOK_ROOT / dirname / f"{dirname}_RUNBOOK.md"
        if not path.is_file():
            continue
        for model_id, run_id in RUN_ID_PATTERN.findall(path.read_text(encoding="utf-8")):
            key = model_id.casefold()
            if key in expected:
                found[key] = {"base_run_id": run_id, "source_runbook": str(path)}
    missing = sorted(expected - set(found))
    if missing:
        raise ValueError(f"Missing base run-id(s): {missing}")
    return {key: found[key] for key in sorted(expected)}


def e5_run_id(base_run_id: str, training_profile: str | None = "uniform_train_batch4_v1") -> str:
    if not base_run_id.endswith("_seed2026"):
        raise ValueError(f"Invalid base run-id: {base_run_id}")
    if training_profile != "uniform_train_batch4_v1":
        raise ValueError("E5 scope27 run-id requires uniform_train_batch4_v1.")
    suffix = "_loss_msa_hybrid_bs4_seed2026"
    return base_run_id[:-len("_seed2026")] + suffix


def _base_config(model_id: str) -> dict[str, Any]:
    return get_model_factory(model_id).resolve_config(load_protocol(), run_mode="formal")


def build_variant_manifest(training_profile: str | None = "uniform_train_batch4_v1") -> dict[str, Any]:
    if training_profile != "uniform_train_batch4_v1":
        raise ValueError("E5 scope27 manifest requires uniform_train_batch4_v1.")
    registry = load_registry()
    contract = validate_registry_contract(registry)
    if contract["status"] != "PASS":
        raise ValueError(f"E5 registry contract failed: {contract}")
    base_ids = extract_base_run_ids()
    profile = get_profile_metadata(CLI_PROFILE_ID)
    output_root = PROJECT_ROOT / BATCH4_FORMAL_OUTPUT_ROOT_RELATIVE
    entries = []
    for ordinal, model_id in enumerate((*TRAINABLE_MODELS, *NONTRAINABLE_MODELS), 1):
        registry_entry = registry.get(model_id)
        trainable = model_id in TRAINABLE_MODELS
        base_run_id = base_ids[model_id]["base_run_id"]
        run_id = e5_run_id(base_run_id, training_profile)
        entries.append({
            "ordinal": ordinal, "entry_id": f"e5_{model_id}", "model_id": model_id,
            "display_name": registry_entry.display_name,
            "entry_type": "TRAIN_COMMON_LOSS" if trainable else "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC",
            "e5_run_id": run_id, "run_id": run_id,
            "training_mode": "TRAIN" if trainable else "EVALUATE_ONLY",
            "formal_training": trainable, "loss_id": profile["loss_id"],
            "model_config": _base_config(model_id),
            "precision_identity": expected_model_precision_identity(model_id, training_profile),
            "expected_output_root": str(output_root),
            "expected_run_dir": str(output_root / run_id),
            "uses_physical_support": bool(registry_entry.requires_graph) and model_id not in {"mtgnn", "agcrn", "stid"},
            **batch_identity(training_profile),
        })
    entries.append({
        "ordinal": 27, "entry_id": BATCH4_A8_REFERENCE_ID,
        "model_id": "st_mgprompt_a8", "display_name": "ST-MGPrompt A8 (w/o MS-MG-DWU)",
        "entry_type": "REFERENCE_ONLY_FORMAL_A8", "e5_run_id": BATCH4_A8_REFERENCE_ID,
        "run_id": BATCH4_A8_REFERENCE_ID, "training_mode": "REFERENCE_ONLY",
        "formal_training": True, "batch_size": 4, "loss_id": profile["loss_id"],
        "expected_output_root": None, "reference_status": "PENDING_VALIDATION",
        **batch_identity(training_profile),
    })
    return {
        "schema_version": "e5_variant_manifest_v3", "scope_id": "e5_batch4_scope27_seed2026",
        "profile_id": profile["profile_id"], "training_profile_id": training_profile,
        "counts": {"trainable": 24, "evaluate_only": 2, "a8_reference": 1, "total": 27},
        "exclusions": ["segrnn", "msgnet"], "entries": entries,
    }


def build_run_id_map(training_profile: str | None = "uniform_train_batch4_v1") -> dict[str, Any]:
    manifest = build_variant_manifest(training_profile)
    return {
        "schema_version": "e5_run_id_map_v2", "scope_id": manifest["scope_id"],
        "entries": [{"ordinal": row["ordinal"], "model_id": row["model_id"], "run_id": row["run_id"], "entry_type": row["entry_type"]} for row in manifest["entries"]],
        "runs": {row["model_id"]: row["run_id"] for row in manifest["entries"]},
        "a8_reference_id": BATCH4_A8_REFERENCE_ID, **batch_identity(training_profile),
    }
