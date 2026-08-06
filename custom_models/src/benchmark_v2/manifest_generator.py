"""Deterministic explicit-manifest generator for active Batch4 scopes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .original_scope26 import CURRENT_SCOPE26_ID
from .runtime import PROJECT_ROOT


ORIGINAL_MANIFEST_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/BATCH4/CURRENT_BATCH4_SCOPE26_MANIFEST.json"
ORIGINAL_RUN_MAP_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/BATCH4/ORIGINAL_SCOPE26_RUN_ID_MAP.json"
ORIGINAL_EXCLUSIONS_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/BATCH4/CURRENT_SCOPE_EXCLUSIONS.json"
E5_MANIFEST_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/E5/E5_SCOPE27_VARIANT_MANIFEST.json"
E5_RUN_MAP_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/E5/E5_SCOPE27_RUN_ID_MAP.json"
E5_POLICY_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/E5/E5_SCOPE27_READINESS_POLICY.json"
E5_SCOPE_ID = "e5_batch4_scope27_seed2026"
E5_OUTPUT_ROOT = "custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026"
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
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def generate_original() -> dict[str, Any]:
    payload = _load(ORIGINAL_MANIFEST_PATH)
    if payload.get("scope_id") != CURRENT_SCOPE26_ID:
        raise ValueError("The current Original manifest is not scope26.")
    entries = sorted(payload.get("entries", []), key=lambda row: int(row["ordinal"]))
    if len(entries) != 26:
        raise ValueError("Original manifest must contain 26 entries.")
    payload["entries"] = entries
    payload["counts"] = {"trainable": 24, "evaluate_only": 2, "total": 26}
    run_map = _load(ORIGINAL_RUN_MAP_PATH)
    run_map.update({
        "scope_id": CURRENT_SCOPE26_ID,
        "output_root": payload["output_root"],
        "entries": [{"ordinal": row["ordinal"], "model_id": row["model_id"], "run_id": row["run_id"], "output_root": row["output_root"]} for row in entries],
    })
    _write(ORIGINAL_MANIFEST_PATH, payload)
    _write(ORIGINAL_RUN_MAP_PATH, run_map)
    return {"scope_id": CURRENT_SCOPE26_ID, "entry_count": len(entries), "manifest": str(ORIGINAL_MANIFEST_PATH), "run_map": str(ORIGINAL_RUN_MAP_PATH)}


def generate_e5() -> dict[str, Any]:
    payload = _load(E5_MANIFEST_PATH)
    if payload.get("scope_id") != E5_SCOPE_ID:
        raise ValueError("The current E5 manifest is not scope27.")
    order = {model_id: index for index, model_id in enumerate((*E5_TRAINABLE, *E5_EVALUATE_ONLY, "st_mgprompt_a8"))}
    entries = [row for row in payload.get("entries", []) if row.get("model_id") in order]
    entries.sort(key=lambda row: order[row["model_id"]])
    if len(entries) != 27:
        raise ValueError("E5 manifest must contain 27 entries.")
    for ordinal, row in enumerate(entries, 1):
        row["ordinal"] = ordinal
        row["training_profile_id"] = TRAINING_PROFILE_ID
        row["loss_id"] = E5_LOSS_ID
        if row["model_id"] != "st_mgprompt_a8":
            row["output_root"] = E5_OUTPUT_ROOT
            row["expected_output_root"] = E5_OUTPUT_ROOT
    payload.update({
        "entries": entries, "output_root": E5_OUTPUT_ROOT,
        "training_profile_id": TRAINING_PROFILE_ID, "e5_loss_id": E5_LOSS_ID,
        "counts": {"trainable": 24, "evaluate_only": 2, "a8_reference": 1, "total_evidence": 27},
    })
    run_map = _load(E5_RUN_MAP_PATH)
    run_map.update({
        "scope_id": E5_SCOPE_ID, "training_profile_id": TRAINING_PROFILE_ID,
        "output_root": E5_OUTPUT_ROOT, "unique_run_id_count": 27,
        "entries": [{"ordinal": row["ordinal"], "entry_id": row["entry_id"], "model_id": row["model_id"], "e5_run_id": row["e5_run_id"], "entry_type": row["entry_type"], "output_root": row.get("output_root")} for row in entries],
        "runs": {row["model_id"]: row["e5_run_id"] for row in entries},
    })
    _write(E5_MANIFEST_PATH, payload)
    _write(E5_RUN_MAP_PATH, run_map)
    return {"scope_id": E5_SCOPE_ID, "entry_count": len(entries), "manifest": str(E5_MANIFEST_PATH), "run_map": str(E5_RUN_MAP_PATH)}


def generate_all() -> dict[str, Any]:
    return {"original": generate_original(), "e5": generate_e5()}


__all__ = ["generate_all", "generate_e5", "generate_original"]
