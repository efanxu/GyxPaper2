from __future__ import annotations

from typing import Any

from .constants import INTERNAL_ROOT, INTERNAL_TARGETS
from .io_utils import read_json

IGNORED_IDENTITY_FIELDS = {"component_ablation", "run_id", "output_root"}
SEMANTIC_DEFAULTS = {
    "macro_to_fine_exclude_recent_len": 0,
    "macro_to_fine_mode": "query_attention",
    "share_cross_attention_projections": False,
}


def _diff(left: dict[str, Any], right: dict[str, Any], ignored=IGNORED_IDENTITY_FIELDS) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "A0": left.get(key, SEMANTIC_DEFAULTS.get(key)),
            "ablation": right.get(key, SEMANTIC_DEFAULTS.get(key)),
        }
        for key in sorted(set(left) | set(right) | set(SEMANTIC_DEFAULTS))
        if left.get(key, SEMANTIC_DEFAULTS.get(key))
        != right.get(key, SEMANTIC_DEFAULTS.get(key))
        and key not in ignored
    }


def _directions(config: dict[str, Any]) -> dict[str, bool]:
    enabled = bool(config.get("use_cross_fusion"))
    return {
        "fine_to_coarse": enabled and not bool(config.get("disable_reverse_cross")),
        "coarse_to_fine": enabled and not bool(config.get("disable_macro_to_fine_cross")),
    }


def audit_internal_config(root=INTERNAL_ROOT) -> dict:
    manifest_path = root / "experiment_manifest.json"
    manifest = read_json(manifest_path)
    definitions = {row["variant_id"]: row for row in manifest["variants"]}
    a0_path = root / "A0" / "effective_config.json"
    a0 = read_json(a0_path)
    comparisons = []
    for variant, (component, allowed) in INTERNAL_TARGETS.items():
        config_path = root / variant / "effective_config.json"
        existing_diff_path = root / variant / "effective_config_diff.json"
        config = read_json(config_path)
        differences = _diff(a0, config)
        actual = tuple(differences)
        expected_manifest = tuple(definitions[variant].get("expected_unique_diff", []))
        unexpected = sorted(set(actual) - set(allowed))
        missing = sorted(set(allowed) - set(actual))
        valid = not unexpected and not missing and set(expected_manifest) == set(allowed)
        comparisons.append({
            "comparison": f"A0_vs_{variant}", "variant_id": variant, "target_component": component,
            "allowed_differences": list(allowed), "actual_differences": differences,
            "unexpected_differences": unexpected, "missing_expected_differences": missing,
            "config_diff_valid": valid, "CONFIG_DIFF_INVALID": not valid,
            "source_evidence": [str(manifest_path), str(a0_path), str(config_path), str(existing_diff_path),
                "custom_models/src/st_mgprompt/model.py", "custom_models/src/st_mgprompt/coupling_block.py",
                "custom_models/src/st_mgprompt/cross_fusion.py", "custom_models/src/st_mgprompt/decoder.py"],
        })
    a5 = read_json(root / "A5" / "effective_config.json")
    a6 = read_json(root / "A6" / "effective_config.json")
    a0_a6_differences = _diff(a0, a6)
    direction_valid = False
    return {
        "schema_version": "e8_internal_config_diff_v2",
        "status": "PASS" if all(row["config_diff_valid"] for row in comparisons) else "FAIL",
        "comparisons": comparisons,
        "direction_decomposition": {
            "A0_directions": _directions(a0), "A5_directions": _directions(a5), "A6_directions": _directions(a6),
            "A0_A6_differences": a0_a6_differences,
            "A5_A6_non_direction_differences": _diff(a5, a6),
            "DIRECTION_DECOMPOSITION_VALID": direction_valid,
            "allowed_conclusions": [],
            "reason": (
                "No current A0-A8 variant removes exactly one Cross Fusion direction. "
                "A6 changes the Macro-to-Fine query window and A7 shares attention projections."
            ),
        },
        "source_semantics": {
            "A4_change": "macro_prompt_pooling: attention -> mean; macro_prompt_len remains 4",
            "A5_change": "cross_fusion_recent_len: 24 -> 6; both attention directions remain active",
            "A6_change": "macro_to_fine_exclude_recent_len: 0 -> 24; both attention directions remain active",
            "A7_change": "share_cross_attention_projections: false -> true; both directions share one attention module",
        },
    }
