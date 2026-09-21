from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import STMGPromptConfig, resolve_project_path
from .experiment_protocol import CANONICAL_ID, canonical_config


EMPIRICAL_PROTOCOL_ID = "empirical_analysis_v1"
EMPIRICAL_RESULT_ROOT = "custom_models/results/empirical_analysis_v1"

# These fields are frozen across every controlled empirical comparison.  Runtime
# smoke reductions are applied only after the unique-difference audit.
FROZEN_PROTOCOL_FIELDS = (
    "model_input_path",
    "eval_target_path",
    "target_col",
    "target_mask_col",
    "input_patv_col",
    "feature_cols",
    "lookback",
    "max_pred_len",
    "eval_horizons",
    "split_ratios",
    "train_sample_stride",
    "val_sample_stride",
    "test_sample_stride",
    "batch_size",
    "eval_batch_size",
    "train_batch_size",
    "val_batch_size",
    "test_batch_size",
    "epochs",
    "patience",
    "early_stopping_min_delta",
    "seed",
    "enable_physical_clip_eval",
    "physical_power_min_kw",
    "physical_power_max_kw",
    "physical_clip_protocol",
    "checkpoint_selection_metric",
    "checkpoint_selection_mode",
    "primary_val_horizon",
    "result_protocol",
    "protocol_profile",
    "source_scope",
)


@dataclass(frozen=True)
class EmpiricalVariant:
    variant_id: str
    display_name: str
    family: str
    base_config: str
    config_overrides: dict[str, Any]
    expected_unique_diff: tuple[str, ...]
    trainable: bool
    paired_reference: str | None
    implementation_status: str = "ready"
    implementation_note: str = ""

    @property
    def reference_only(self) -> bool:
        return not self.trainable and self.implementation_status == "reference"

    @property
    def ready(self) -> bool:
        return self.implementation_status in {"ready", "reference"}

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["expected_unique_diff"] = list(self.expected_unique_diff)
        row["reference_only"] = self.reference_only
        return row


def _variant(
    variant_id: str,
    display_name: str,
    family: str,
    overrides: dict[str, Any] | None = None,
    *,
    trainable: bool = True,
    paired_reference: str | None = None,
    status: str = "ready",
    note: str = "",
) -> EmpiricalVariant:
    values = overrides or {}
    return EmpiricalVariant(
        variant_id=variant_id,
        display_name=display_name,
        family=family,
        base_config="CANONICAL_FULL",
        config_overrides=values,
        expected_unique_diff=tuple(values),
        trainable=trainable,
        paired_reference=paired_reference,
        implementation_status=status,
        implementation_note=note,
    )


_T_VARIANTS = (
    _variant("T0", "Canonical Fixed Dual reference", "T", trainable=False, paired_reference=CANONICAL_ID, status="reference"),
    _variant("T1", "Fine-only", "T", {"vadsp_gate_mode": "fine_only"}, paired_reference="T0"),
    _variant("T2", "Coarse-only", "T", {"vadsp_gate_mode": "coarse_only"}, paired_reference="T0"),
    _variant("T3", "Shared temporal transform dual", "T", {"temporal_branch_transform": "shared"}, paired_reference="T0", note="Shares the temporal transform object and retains explicit branch output projections."),
    _variant("T4", "Downsample-upsample Coarse dual", "T", {"coarse_alignment_mode": "causal_downsample_upsample"}, paired_reference="T0", note="Uses causal anchor aggregation and causal repeat recovery with an access trace."),
    _variant(
        "T5",
        "Fixed Dual concat/MLP interaction",
        "T",
        {"cross_granularity_interaction": "direct_concat_mlp"},
        paired_reference="T0",
        note="Bypasses Macro Prompt and Cross-Fusion and uses an explicit capacity-recorded concat/MLP projection.",
    ),
)

_G_VARIANTS = (
    _variant("G0", "Matched Micro/Fine + Macro/Coarse", "G", trainable=False, paired_reference=CANONICAL_ID, status="reference"),
    _variant("G1", "Shared Micro graph", "G", {"branch_graph_assignment": "shared_micro"}, paired_reference="G0"),
    _variant("G2", "Shared Macro graph", "G", {"branch_graph_assignment": "shared_macro"}, paired_reference="G0"),
    _variant("G3", "Swapped Micro/Macro graphs", "G", {"branch_graph_assignment": "swapped"}, paired_reference="G0"),
    _variant("G4", "Distance-only dual graph", "G", {"graph_prior_component": "distance_only"}, paired_reference="G0"),
    _variant("G5", "Statistics-only dual graph", "G", {"graph_prior_component": "statistics_only"}, paired_reference="G0"),
    _variant("G6", "Free adaptive graph", "G", {"adaptive_support_mode": "free"}, paired_reference="G0"),
    _variant("G7", "Fixed dual graph", "G", {"adaptive_support_mode": "fixed"}, paired_reference="G0"),
    _variant("G8", "Degree-preserving random rewiring", "G", {"graph_rewire_mode": "degree_preserving_random"}, paired_reference="G0"),
)

_D_VARIANTS = (
    _variant(
        "D0",
        "No diffusion state",
        "D",
        {"graph_operator": "simple"},
        paired_reference="D4",
        note="Uses the existing simple/local graph control and creates no diffusion state.",
    ),
    _variant(
        "D1",
        "Forward first-order diffusion",
        "D",
        {
            "diffusion_order_micro": 1,
            "diffusion_order_macro": 1,
            "diffusion_use_bidirectional": False,
            "diffusion_direction": "forward",
        },
        paired_reference="D4",
    ),
    _variant(
        "D2",
        "Transpose-direction first-order diffusion",
        "D",
        {
            "diffusion_order_micro": 1,
            "diffusion_order_macro": 1,
            "diffusion_use_bidirectional": False,
            "diffusion_direction": "reverse",
        },
        paired_reference="D4",
    ),
    _variant(
        "D3",
        "Bidirectional first-order diffusion",
        "D",
        {"diffusion_order_micro": 1, "diffusion_order_macro": 1},
        paired_reference="D4",
    ),
    _variant(
        "D4",
        "Bidirectional second-order canonical reference",
        "D",
        trainable=False,
        paired_reference=CANONICAL_ID,
        status="reference",
    ),
    _variant(
        "D5",
        "Bidirectional third-order diffusion",
        "D",
        {"diffusion_order_micro": 3, "diffusion_order_macro": 3},
        paired_reference="D4",
    ),
)

_F_VARIANTS = (
    _variant("F0", "Independent dual branches", "F", {"use_macro_prompt": False, "use_cross_fusion": False}, paired_reference="F7"),
    _variant("F1", "Element-wise add", "F", {"fusion_mode": "add"}, paired_reference="F7"),
    _variant("F2", "Concat + MLP", "F", {"fusion_mode": "concat"}, paired_reference="F7"),
    _variant("F3", "Unified gated fusion", "F", {"fusion_mode": "unified_gated"}, paired_reference="F7"),
    _variant("F4", "Macro-to-Fine only", "F", {"disable_reverse_cross": True}, paired_reference="F7"),
    _variant("F5", "Fine-to-Coarse only", "F", {"disable_macro_to_fine_cross": True}, paired_reference="F7"),
    _variant("F6", "Bidirectional Cross-Fusion, shared projections", "F", {"share_cross_attention_projections": True}, trainable=False, paired_reference="A7", status="reference"),
    _variant("F7", "Bidirectional Cross-Fusion, independent projections", "F", trainable=False, paired_reference=CANONICAL_ID, status="reference"),
    _variant("F8", "Mean-pooling Macro Prompt", "F", {"macro_prompt_pooling": "mean"}, trainable=False, paired_reference="A4", status="reference"),
)

_L_VARIANTS = (
    _variant("L0", "Equal Smooth L1", "L", {"granularity_weight_mode": "static", "site_weight_mode": "static"}, paired_reference="L3"),
    _variant("L1", "Horizon difficulty-rate only", "L", {"site_weight_mode": "static"}, paired_reference="L3"),
    _variant("L2", "Dynamic node difficulty only", "L", {"granularity_weight_mode": "static"}, paired_reference="L3"),
    _variant("L3", "Complete MS-MG-DWU", "L", trainable=False, paired_reference=CANONICAL_ID, status="reference"),
    _variant("L4", "Static increasing horizon weights", "L", {"granularity_weight_mode": "static_increasing"}, status="pending", paired_reference="L3", note="Implemented in empirical step 8."),
    _variant("L5", "Uncertainty weighting", "L", {"granularity_weight_mode": "uncertainty_precision", "site_weight_mode": "static"}, paired_reference="L3"),
    _variant("L6", "Dynamic weight average", "L", {"granularity_weight_mode": "dynamic_weight_average"}, status="pending", paired_reference="L3", note="Implemented in empirical step 8."),
    _variant(
        "L7",
        "Score-aligned hybrid",
        "L",
        {"use_msmg_dwu": False, "loss_function": "masked_score_aligned_hybrid", "loss_protocol": "fair_main"},
        trainable=False,
        paired_reference="A8",
        status="reference",
    ),
)

# R scenarios are inference-time groups/perturbations, not additional model
# trainings.  R0 anchors that family to the canonical predictions; scenario
# membership is intentionally deferred to step 9 where train-only thresholds
# are fitted and frozen.
_R_VARIANTS = (
    _variant("R0", "Canonical robustness reference", "R", trainable=False, paired_reference=CANONICAL_ID, status="reference"),
)

EMPIRICAL_FAMILIES: dict[str, dict[str, EmpiricalVariant]] = {
    family: {item.variant_id: item for item in variants}
    for family, variants in {
        "T": _T_VARIANTS,
        "G": _G_VARIANTS,
        "D": _D_VARIANTS,
        "F": _F_VARIANTS,
        "L": _L_VARIANTS,
        "R": _R_VARIANTS,
    }.items()
}
EMPIRICAL_VARIANTS = {
    variant_id: variant
    for family in EMPIRICAL_FAMILIES.values()
    for variant_id, variant in family.items()
}


def get_empirical_variant(variant_id: str, family: str | None = None) -> EmpiricalVariant:
    variant_id = str(variant_id).upper()
    mapping = EMPIRICAL_VARIANTS if family is None else EMPIRICAL_FAMILIES.get(str(family).upper())
    if mapping is None:
        raise ValueError(f"Unknown empirical family {family!r}. Valid families are {', '.join(EMPIRICAL_FAMILIES)}.")
    try:
        return mapping[variant_id]
    except KeyError as exc:
        raise ValueError(f"Unknown empirical variant {variant_id!r}.") from exc


def apply_empirical_variant(
    config: STMGPromptConfig | None,
    variant_id: str,
    family: str | None = None,
) -> STMGPromptConfig:
    variant = get_empirical_variant(variant_id, family)
    if variant.implementation_status == "pending":
        raise NotImplementedError(
            f"{variant.variant_id} is registered but pending: {variant.implementation_note}"
        )
    result = canonical_config(config)
    # Empirical runs have their own identity namespace and must never be
    # interpreted as an A-series artifact.
    result.component_ablation = None
    result.variant = variant.variant_id
    result.definition = variant.display_name
    for key, value in variant.config_overrides.items():
        if not hasattr(result, key):
            raise AttributeError(f"Unknown STMGPromptConfig field in {variant.variant_id}: {key}")
        setattr(result, key, deepcopy(value))
    result.validate()
    return result


def empirical_config_diff(
    config: STMGPromptConfig,
    variant_id: str,
    family: str | None = None,
) -> dict[str, Any]:
    variant = get_empirical_variant(variant_id, family)
    baseline = canonical_config().to_dict()
    effective = config.to_dict()
    ignored = {"eligible_for_fair_main_table", "variant", "definition"}
    differences = {
        key: {"canonical": baseline.get(key), "effective": effective.get(key)}
        for key in baseline
        if key not in ignored and baseline.get(key) != effective.get(key)
    }
    expected = set(variant.expected_unique_diff)
    actual = set(differences)
    frozen_changes = sorted(actual.intersection(FROZEN_PROTOCOL_FIELDS))
    return {
        "protocol_id": EMPIRICAL_PROTOCOL_ID,
        "variant_id": variant.variant_id,
        "family": variant.family,
        "implementation_status": variant.implementation_status,
        "expected_diff_fields": sorted(expected),
        "actual_diff_fields": sorted(actual),
        "missing_expected_diff_fields": sorted(expected - actual),
        "unexpected_diff_fields": sorted(actual - expected),
        "frozen_protocol_changes": frozen_changes,
        "differences": differences,
        "passed": actual == expected and not frozen_changes,
    }


def assert_empirical_expected_diff(
    config: STMGPromptConfig,
    variant_id: str,
    family: str | None = None,
) -> dict[str, Any]:
    report = empirical_config_diff(config, variant_id, family)
    if not report["passed"]:
        raise ValueError(
            f"Empirical unique-difference audit failed for {variant_id}: "
            f"missing={report['missing_expected_diff_fields']}, "
            f"unexpected={report['unexpected_diff_fields']}, "
            f"frozen={report['frozen_protocol_changes']}"
        )
    return report


def dry_run_report(variant: EmpiricalVariant) -> dict[str, Any]:
    if variant.implementation_status == "pending":
        return {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "variant_id": variant.variant_id,
            "display_name": variant.display_name,
            "family": variant.family,
            "implementation_status": "pending",
            "trainable": variant.trainable,
            "paired_reference": variant.paired_reference,
            "expected_diff_fields": list(variant.expected_unique_diff),
            "declared_overrides": deepcopy(variant.config_overrides),
            "passed": True,
            "execution_ready": False,
            "note": variant.implementation_note,
        }
    config = apply_empirical_variant(None, variant.variant_id, variant.family)
    report = assert_empirical_expected_diff(config, variant.variant_id, variant.family)
    report.update(
        {
            "display_name": variant.display_name,
            "trainable": variant.trainable,
            "paired_reference": variant.paired_reference,
            "declared_overrides": deepcopy(variant.config_overrides),
            "execution_ready": True,
        }
    )
    return report


def write_empirical_matrix(
    path: str | Path,
    variants: Iterable[EmpiricalVariant] | None = None,
) -> Path:
    destination = resolve_project_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected = list(variants or EMPIRICAL_VARIANTS.values())
    rows = []
    for variant in selected:
        row = variant.to_dict()
        row["config_overrides"] = json.dumps(row["config_overrides"], ensure_ascii=False, sort_keys=True)
        row["expected_unique_diff"] = json.dumps(row["expected_unique_diff"], ensure_ascii=False)
        rows.append(row)
    fieldnames = list(rows[0]) if rows else list(EmpiricalVariant.__dataclass_fields__)
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


__all__ = [
    "EMPIRICAL_PROTOCOL_ID",
    "EMPIRICAL_RESULT_ROOT",
    "FROZEN_PROTOCOL_FIELDS",
    "EmpiricalVariant",
    "EMPIRICAL_FAMILIES",
    "EMPIRICAL_VARIANTS",
    "get_empirical_variant",
    "apply_empirical_variant",
    "empirical_config_diff",
    "assert_empirical_expected_diff",
    "dry_run_report",
    "write_empirical_matrix",
]
