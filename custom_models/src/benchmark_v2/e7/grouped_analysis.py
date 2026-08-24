from __future__ import annotations

import os
from pathlib import Path

from .constants import DOC_ROOT, PROJECT_ROOT

REQUIRED_FROZEN_DEFINITION_FILES = (
    "E10_GROUP_DEFINITION_MANIFEST.json",
    "test_window_identity_manifest.json",
    "shared_difficult_top10_window_ids.json",
)


def grouped_definition_audit() -> dict:
    candidates: list[Path] = []
    search_roots = (
        PROJECT_ROOT / "custom_models" / "docs",
        PROJECT_ROOT / "custom_models" / "results",
    )
    required = set(REQUIRED_FROZEN_DEFINITION_FILES)
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for root, dirs, files in os.walk(search_root, onerror=lambda _: None):
            dirs[:] = [name for name in dirs if not name.startswith(".") and name not in {"results_smoke", "smoke"}]
            for name in required.intersection(files):
                candidates.append(Path(root) / name)
    by_name = {path.name: str(path.resolve()) for path in candidates}
    missing = [name for name in REQUIRED_FROZEN_DEFINITION_FILES if name not in by_name]
    threshold_root = PROJECT_ROOT / "custom_models" / "results" / "st_mgprompt_volatility_group_analysis" / "volatility_group_analysis_seed2026"
    thresholds = {
        "volatility_thresholds": str(threshold_root / "volatility_thresholds.json") if (threshold_root / "volatility_thresholds.json").is_file() else None,
        "ramp_thresholds": str(threshold_root / "ramp_thresholds.json") if (threshold_root / "ramp_thresholds.json").is_file() else None,
    }
    return {
        "status": "READY" if not missing else "BLOCKED_DEFINITION_MISSING",
        "required_files": by_name,
        "missing_files": missing,
        "threshold_artifacts_found": thresholds,
        "reason": None if not missing else "Threshold values exist, but the frozen common test-window identity and Shared difficult Top10% definition are not auditable.",
    }


def write_group_gap(path: str | Path = DOC_ROOT / "E7_GROUP_DEFINITION_GAP.md") -> Path:
    audit = grouped_definition_audit()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = """# E7 Group Definition Gap

Status: `BLOCKED_DEFINITION_MISSING`

Training-fitted volatility and Ramp threshold artifacts exist, but E7 cannot locate all of the following frozen public definitions:

- `E10_GROUP_DEFINITION_MANIFEST.json`
- `test_window_identity_manifest.json`
- `shared_difficult_top10_window_ids.json`

Therefore E7 does not invent thresholds, does not construct model-specific groups, and does not publish grouped metrics. This does not change core 10-evidence readiness.
"""
    target.write_text(text, encoding="utf-8")
    return target
