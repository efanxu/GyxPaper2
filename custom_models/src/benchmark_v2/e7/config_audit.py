from __future__ import annotations

from typing import Any

from .constants import INTERNAL_ROOT, INTERNAL_TARGETS
from .io_utils import read_json

IGNORED_IDENTITY_FIELDS = {"component_ablation", "run_id", "output_root"}


def _diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {"A0": left.get(key), "ablation": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key) and key not in IGNORED_IDENTITY_FIELDS
    }


def audit_internal_config() -> dict:
    manifest_path = INTERNAL_ROOT / "experiment_manifest.json"
    manifest = read_json(manifest_path)
    definitions = {row["variant_id"]: row for row in manifest["variants"]}
    a0_path = INTERNAL_ROOT / "A0" / "effective_config.json"
    a0 = read_json(a0_path)
    comparisons = []
    for variant, (component, allowed) in INTERNAL_TARGETS.items():
        config_path = INTERNAL_ROOT / variant / "effective_config.json"
        existing_diff_path = INTERNAL_ROOT / variant / "effective_config_diff.json"
        config = read_json(config_path)
        differences = _diff(a0, config)
        actual = tuple(differences)
        expected_manifest = tuple(definitions[variant].get("expected_unique_diff", []))
        unexpected = sorted(set(actual) - set(allowed))
        missing = sorted(set(allowed) - set(actual))
        valid = not unexpected and not missing and set(expected_manifest) == set(allowed)
        comparisons.append(
            {
                "comparison": f"A0_vs_{variant}",
                "variant_id": variant,
                "target_component": component,
                "allowed_differences": list(allowed),
                "actual_differences": differences,
                "unexpected_differences": unexpected,
                "missing_expected_differences": missing,
                "config_diff_valid": valid,
                "CONFIG_DIFF_INVALID": not valid,
                "source_evidence": [
                    str(manifest_path),
                    str(a0_path),
                    str(config_path),
                    str(existing_diff_path),
                    "custom_models/src/st_mgprompt/model.py",
                    "custom_models/src/st_mgprompt/graph_layers.py",
                ],
            }
        )
    return {
        "schema_version": "e7_internal_config_diff_v1",
        "status": "PASS" if all(row["config_diff_valid"] for row in comparisons) else "FAIL",
        "comparisons": comparisons,
    }
