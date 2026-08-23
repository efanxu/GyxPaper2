"""Deterministic explicit-manifest generator for the active Original scope."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .original_scope26 import CURRENT_SCOPE26_ID
from .runtime import PROJECT_ROOT


ORIGINAL_MANIFEST_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/BATCH4/CURRENT_BATCH4_SCOPE26_MANIFEST.json"
ORIGINAL_RUN_MAP_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/BATCH4/ORIGINAL_SCOPE26_RUN_ID_MAP.json"
ORIGINAL_EXCLUSIONS_PATH = PROJECT_ROOT / "custom_models/docs/benchmark_v2/BATCH4/CURRENT_SCOPE_EXCLUSIONS.json"


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


__all__ = ["generate_original"]
