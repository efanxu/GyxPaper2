from __future__ import annotations

import os
from pathlib import Path

from .constants import DOC_ROOT, PROJECT_ROOT

REQUIRED_FROZEN_DEFINITION_FILES = (
    "E10_GROUP_DEFINITION_MANIFEST.json", "test_window_identity_manifest.json", "shared_difficult_top10_window_ids.json",
)


def grouped_definition_audit() -> dict:
    found: list[Path] = []
    required = set(REQUIRED_FROZEN_DEFINITION_FILES)
    for search_root in (PROJECT_ROOT / "custom_models" / "docs", PROJECT_ROOT / "custom_models" / "results"):
        if not search_root.is_dir():
            continue
        for root, dirs, files in os.walk(search_root, onerror=lambda _: None):
            dirs[:] = [name for name in dirs if not name.startswith(".") and name not in {"results_smoke", "smoke"}]
            for name in required.intersection(files):
                found.append(Path(root) / name)
    by_name = {path.name: str(path.resolve()) for path in found}
    missing = [name for name in REQUIRED_FROZEN_DEFINITION_FILES if name not in by_name]
    threshold_root = PROJECT_ROOT / "custom_models" / "results" / "st_mgprompt_volatility_group_analysis" / "volatility_group_analysis_seed2026"
    return {
        "status": "READY" if not missing else "BLOCKED_DEFINITION_MISSING", "required_files": by_name,
        "missing_files": missing,
        "threshold_artifacts_found": {
            "volatility_thresholds": str(threshold_root / "volatility_thresholds.json") if (threshold_root / "volatility_thresholds.json").is_file() else None,
            "ramp_thresholds": str(threshold_root / "ramp_thresholds.json") if (threshold_root / "ramp_thresholds.json").is_file() else None,
        },
        "reason": None if not missing else "Threshold artifacts exist, but the frozen common test-window and Shared difficult Top10% identities are incomplete.",
    }


def write_group_gap(path: str | Path = DOC_ROOT / "E8_GROUP_DEFINITION_GAP.md") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# E8 Group Definition Gap\n\nStatus: `BLOCKED_DEFINITION_MISSING`\n\n"
        "Training-fitted volatility and Ramp threshold artifacts exist, but E8 cannot locate all frozen public definitions:\n\n"
        "- `E10_GROUP_DEFINITION_MANIFEST.json`\n- `test_window_identity_manifest.json`\n"
        "- `shared_difficult_top10_window_ids.json`\n\n"
        "E8 therefore does not invent thresholds or publish grouped metrics. Core 11-evidence readiness is independent of this gap.\n",
        encoding="utf-8",
    )
    return target
