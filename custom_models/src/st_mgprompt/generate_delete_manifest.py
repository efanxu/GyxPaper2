from __future__ import annotations

import json
from pathlib import Path

from .experiment_protocol import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OLD_RESULT_ROOTS = {
    "old_component_A0_A10": (
        "custom_models/results/st_mgprompt_component_ablation/component_ablation_p1_seed2026",
        "Old Dynamic-VADSP A0-A10; old A1 was migrated as Canonical Full.",
    ),
    "old_precision_P0_P5": (
        "custom_models/results/st_mgprompt_precision/precision_ablation_fair_main_seed2026_all_p0_p5",
        "Old Dynamic-VADSP P0-P5 results are invalid under the Fixed-Dual protocol.",
    ),
    "old_method_full_P0_P5": (
        "custom_models/results/st_mgprompt_method_full_ablation/method_full_msmgdwu_seed2026_p0_p5",
        "Old Dynamic-VADSP P0-P5 method-full results are invalid under the Fixed-Dual protocol.",
    ),
    "old_method_full_test_P0_P5": (
        "custom_models/results/st_mgprompt_method_full_test_ablation/method_full_msmgdwu_seed2026_p0_p5_test",
        "Old Dynamic-VADSP P0-P5 test artifacts are invalid under the Fixed-Dual protocol.",
    ),
    "old_supplementary_smoke": (
        "custom_models/results/st_mgprompt_supplementary_ablation_smoke",
        "Supplementary V/L/S alias experiments were retired by the formal refactor.",
    ),
    "old_static_mg_candidate": (
        "custom_models/results/st_mgprompt_full_candidate/full_fixed_dual_static_mg_seed2026",
        "Static-MG candidate is not Canonical Full and was explicitly retired.",
    ),
    "old_static_mg_candidate_smoke": (
        "custom_models/results/st_mgprompt_full_candidate_smoke",
        "Static-MG candidate-only smoke artifacts were explicitly retired.",
    ),
    "old_volatility_supplementary": (
        "custom_models/results/st_mgprompt_volatility_group_analysis/volatility_group_analysis_seed2026",
        "Old A0 Dynamic-VADSP versus A1 Fixed-Dual supplementary analysis is obsolete.",
    ),
    "old_volatility_supplementary_smoke": (
        "custom_models/results/st_mgprompt_volatility_group_analysis_smoke/volatility_group_analysis_smoke_seed2026",
        "Old volatility supplementary-only smoke analysis is obsolete.",
    ),
}

OLD_CODE_FILES = {
    "custom_models/src/st_mgprompt/run_supplementary_ablation.py": "Supplementary-only runner.",
    "custom_models/src/st_mgprompt/test_supplementary_ablation.py": "Supplementary-only tests.",
    "custom_models/src/st_mgprompt/fixed_dual_static_mg_candidate.py": "Retired Static-MG candidate-only helper.",
    "custom_models/src/st_mgprompt/run_fixed_dual_static_mg_candidate.py": "Retired Static-MG candidate-only runner.",
    "custom_models/src/st_mgprompt/test_fixed_dual_static_mg_candidate.py": "Retired Static-MG candidate-only tests.",
    "custom_models/src/st_mgprompt/run_volatility_group_analysis.py": "Retired old A0/A1 supplementary-only runner.",
    "custom_models/src/st_mgprompt/volatility_group_analysis.py": "Retired old A0/A1 supplementary-only implementation.",
    "custom_models/src/st_mgprompt/test_volatility_group_analysis.py": "Retired old A0/A1 supplementary-only tests.",
}


def build_manifest() -> dict:
    entries = []
    for experiment, (relative_root, reason) in OLD_RESULT_ROOTS.items():
        root = PROJECT_ROOT / relative_root
        if not root.is_dir():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            entries.append(
                {
                    "absolute_path": str(path.resolve()),
                    "file_type": path.suffix.lower() or "no_extension",
                    "old_experiment": experiment,
                    "deletion_reason": reason,
                    "migrated": experiment == "old_component_A0_A10" and "\\A1\\" in str(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    for relative, reason in OLD_CODE_FILES.items():
        path = PROJECT_ROOT / relative
        if path.is_file():
            entries.append(
                {
                    "absolute_path": str(path.resolve()),
                    "file_type": "python_source",
                    "old_experiment": "supplementary_only_code",
                    "deletion_reason": reason,
                    "migrated": False,
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "manifest_version": 1,
        "project_root": str(PROJECT_ROOT.resolve()),
        "canonical_migration_verified": True,
        "safety_rule": "Delete one explicit file path at a time; no wildcard or recursive deletion.",
        "entry_count": len(entries),
        "entries": entries,
        "candidate_empty_directories_after_file_deletion": [
            str((PROJECT_ROOT / relative).resolve())
            for relative, _ in OLD_RESULT_ROOTS.values()
        ],
    }


def main() -> None:
    manifest = build_manifest()
    path = PROJECT_ROOT / "DELETE_MANIFEST_refactor.json"
    write_json(path, manifest)
    print(json.dumps({"path": str(path), "entry_count": manifest["entry_count"]}, indent=2))


if __name__ == "__main__":
    main()
