from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import STMGPromptConfig


CANONICAL_ID = "full_fixed_dual_keep_msmgdwu_seed2026"
CANONICAL_RUN_ID = CANONICAL_ID
PRECISION_RUN_ID = "precision_ablation_fixed_dual_seed2026"
COMPONENT_RUN_ID = "component_ablation_fixed_dual_seed2026"

CANONICAL_RESULT_ROOT = "custom_models/results/st_mgprompt_canonical"
PRECISION_RESULT_ROOT = "custom_models/results/st_mgprompt_precision"
COMPONENT_RESULT_ROOT = "custom_models/results/st_mgprompt_component_ablation"

GRAPH_SIMPLE = "simple"
GRAPH_BI_DIFFUSION = "bidirectional_diffusion"
GRAPH_BI_DIFFUSION_DISPLAY = "bi_diffusion"

LEGACY_A8_CONFIG_OVERRIDES = {
    "use_msmg_dwu": False,
    "loss_function": "masked_score_aligned_hybrid",
    "loss_protocol": "fair_main",
}


@dataclass(frozen=True)
class ExperimentVariant:
    variant_id: str
    display_name: str
    experiment_family: str
    trainable: bool
    canonical_reference: str | None
    base_config: str
    config_overrides: dict[str, Any]
    expected_unique_diff: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["expected_unique_diff"] = list(self.expected_unique_diff)
        return value


_PRECISION = (
    ExperimentVariant(
        "P0", "Bi-Diffusion + Last-State + H64 + C1", "precision", False,
        CANONICAL_ID, "CANONICAL_FULL", {}, (),
    ),
    ExperimentVariant(
        "P1", "Simple + Last-State + H64 + C1", "precision", True,
        None, "CANONICAL_FULL", {"graph_operator": GRAPH_SIMPLE}, ("graph_operator",),
    ),
    ExperimentVariant(
        "P2", "Simple + Full-History Cross-Attn + H64 + C1", "precision", True,
        None, "CANONICAL_FULL",
        {"graph_operator": GRAPH_SIMPLE, "decoder_context_mode": "full_history_cross_attention"},
        ("graph_operator", "decoder_context_mode"),
    ),
    ExperimentVariant(
        "P3", "Bi-Diffusion + Full-History Cross-Attn + H64 + C1", "precision", True,
        None, "CANONICAL_FULL", {"decoder_context_mode": "full_history_cross_attention"},
        ("decoder_context_mode",),
    ),
    ExperimentVariant(
        "P4", "Bi-Diffusion + Full-History Cross-Attn + H96 + C1", "precision", True,
        None, "CANONICAL_FULL",
        {"decoder_context_mode": "full_history_cross_attention", "hidden_dim": 96},
        ("decoder_context_mode", "hidden_dim"),
    ),
    ExperimentVariant(
        "P5", "Bi-Diffusion + Full-History Cross-Attn + H96 + C2", "precision", True,
        None, "CANONICAL_FULL",
        {
            "decoder_context_mode": "full_history_cross_attention",
            "hidden_dim": 96,
            "num_coupling_layers": 2,
        },
        ("decoder_context_mode", "hidden_dim", "num_coupling_layers"),
    ),
)

_COMPONENT = (
    ExperimentVariant(
        "A0", "Canonical Full", "component_ablation", False,
        CANONICAL_ID, "CANONICAL_FULL", {}, (),
    ),
    ExperimentVariant(
        "A1", "w/o Spatial Graph", "component_ablation", True,
        None, "CANONICAL_FULL", {"use_graph_in_temporal_encoder": False},
        ("use_graph_in_temporal_encoder",),
    ),
    ExperimentVariant(
        "A2", "w/o Adaptive Graph", "component_ablation", True,
        None, "CANONICAL_FULL", {"use_adaptive_graph": False},
        ("use_adaptive_graph",),
    ),
    ExperimentVariant(
        "A3", "w/o Diffusion", "component_ablation", True,
        None, "CANONICAL_FULL", {"graph_operator": GRAPH_SIMPLE},
        ("graph_operator",),
    ),
    ExperimentVariant(
        "A4", "Mean-Pooling Macro Prompt", "component_ablation", True,
        None, "CANONICAL_FULL", {"macro_prompt_pooling": "mean"},
        ("macro_prompt_pooling",),
    ),
    ExperimentVariant(
        "A5", "Short-context Reverse Cross", "component_ablation", True,
        None, "CANONICAL_FULL", {"cross_fusion_recent_len": 6},
        ("cross_fusion_recent_len",),
    ),
    ExperimentVariant(
        "A6", "Fine-to-Coarse Only Cross Fusion", "component_ablation", True,
        None, "CANONICAL_FULL", {"disable_macro_to_fine_cross": True},
        ("disable_macro_to_fine_cross",),
    ),
    ExperimentVariant(
        "A7", "w/o MS-MG-DWU", "component_ablation", True,
        None, "CANONICAL_FULL", LEGACY_A8_CONFIG_OVERRIDES,
        ("use_msmg_dwu", "loss_function", "loss_protocol"),
    ),
)

PRECISION_VARIANTS = {item.variant_id: item for item in _PRECISION}
COMPONENT_ABLATION_VARIANTS = {item.variant_id: item for item in _COMPONENT}
ALL_VARIANTS = {**PRECISION_VARIANTS, **COMPONENT_ABLATION_VARIANTS}

OBSOLETE_COMPONENT_VARIANTS = {"A8", "A9", "A10"}
OBSOLETE_COMPONENT_NAMES = {"w/o VADSP", "w/o Trend Prior"}

FORMAL_PROTOCOL_FIELDS = (
    "model_input_path", "eval_target_path", "target_col", "target_mask_col", "input_patv_col",
    "lookback", "max_pred_len", "eval_horizons", "split_ratios",
    "train_sample_stride", "val_sample_stride", "test_sample_stride",
    "train_batch_size", "val_batch_size", "test_batch_size",
    "epochs", "patience", "early_stopping_min_delta", "seed",
    "enable_physical_clip_eval", "physical_power_min_kw", "physical_power_max_kw",
    "checkpoint_selection_metric", "checkpoint_selection_mode", "primary_val_horizon",
    "amp_enabled", "windows_safe_mode",
)

ARCHITECTURE_AND_LOSS_FIELDS = (
    "vadsp_gate_mode", "use_vadsp", "graph_operator", "decoder_context_mode",
    "hidden_dim", "num_coupling_layers", "use_graph_in_temporal_encoder",
    "use_adaptive_graph", "use_trend_prior_graph", "use_macro_prompt",
    "disable_reverse_cross", "disable_macro_to_fine_cross", "use_cross_fusion", "use_st_prompt",
    "macro_prompt_len", "macro_prompt_pooling", "cross_fusion_recent_len", "fusion_mode", "st_prompt_mode",
    "st_prompt_use_node_identity",
    "decoder_input_strategy", "use_msmg_dwu", "loss_function", "loss_protocol",
    "granularity_weight_mode", "site_weight_mode",
)

SEMANTIC_CONFIG_DEFAULTS = {
    # Pre-field formal artifacts remain readable, while old A4/A7 values still
    # fail the semantic audit through their other changed fields.
    "st_prompt_mode": "full",
    "macro_prompt_pooling": "attention",
    "st_prompt_use_node_identity": True,
    "disable_macro_to_fine_cross": False,
}


def canonical_overrides() -> dict[str, Any]:
    return {
        "model_name": "STMGPrompt_ComponentAblation",
        "component_ablation": None,
        "model_input_path": "dataset/sdwpf_model_input_base.parquet",
        "eval_target_path": "dataset/sdwpf_eval_target.parquet",
        "target_col": "Patv_raw",
        "target_mask_col": "valid_target_mask",
        "input_patv_col": "Patv_clean_for_input",
        "lookback": 144,
        "max_pred_len": 10,
        "eval_horizons": [3, 6, 10],
        "split_ratios": [0.8, 0.1, 0.1],
        "train_sample_stride": 6,
        "val_sample_stride": 3,
        "test_sample_stride": 1,
        "batch_size": 32,
        "eval_batch_size": 4,
        "train_batch_size": 32,
        "val_batch_size": 4,
        "test_batch_size": 4,
        "epochs": 20,
        "patience": 6,
        "early_stopping_min_delta": 0.01,
        "seed": 2026,
        "enable_physical_clip_eval": True,
        "physical_power_min_kw": 0.0,
        "physical_power_max_kw": 1500.0,
        "checkpoint_selection_metric": "val_official_score_h10",
        "checkpoint_selection_mode": "min",
        "primary_val_horizon": 10,
        "amp_enabled": True,
        "windows_safe_mode": True,
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
        "use_vadsp": True,
        "vadsp_gate_mode": "fixed_dual",
        "use_trend_prior_graph": True,
        "use_adaptive_graph": True,
        "use_graph_temporal_encoder": True,
        "use_graph_in_temporal_encoder": True,
        "use_stmg_coupling_block": True,
        "use_macro_prompt": True,
        "use_cross_fusion": True,
        "use_st_prompt": True,
        "macro_prompt_len": 4,
        "macro_prompt_pooling": "attention",
        "cross_fusion_recent_len": 24,
        "disable_reverse_cross": False,
        "disable_macro_to_fine_cross": False,
        "fusion_mode": "cross",
        "st_prompt_mode": "full",
        "st_prompt_use_node_identity": True,
        "graph_operator": GRAPH_BI_DIFFUSION,
        "decoder_context_mode": "last_state",
        "decoder_history_len": None,
        "decoder_input_strategy": "direct_multi_output_prompt_query",
        "hidden_dim": 64,
        "num_coupling_layers": 1,
        "use_msmg_dwu": True,
        "loss_function": "msmg_dwu_loss",
        "loss_protocol": "method_full",
        "granularity_weight_mode": "difficulty_rate",
        "site_weight_mode": "dynamic",
        "diagnostics_level": "minimal",
        "prediction_accumulation": "streaming",
    }


def canonical_config(config: STMGPromptConfig | None = None) -> STMGPromptConfig:
    from .config import STMGPromptConfig
    result = deepcopy(config) if config is not None else STMGPromptConfig()
    for key, value in canonical_overrides().items():
        setattr(result, key, deepcopy(value))
    return result


def get_variant(variant_id: str, family: str | None = None) -> ExperimentVariant:
    variant_id = str(variant_id).upper()
    if family == "component_ablation":
        if variant_id in OBSOLETE_COMPONENT_VARIANTS:
            raise ValueError("Obsolete component-ablation variant. Valid variants are A0-A7.")
        mapping = COMPONENT_ABLATION_VARIANTS
    elif family == "precision":
        mapping = PRECISION_VARIANTS
    else:
        mapping = ALL_VARIANTS
    try:
        return mapping[variant_id]
    except KeyError as exc:
        valid = ", ".join(mapping)
        raise ValueError(f"Unknown {family or 'formal'} variant {variant_id!r}. Valid variants are {valid}.") from exc


def apply_variant(config: STMGPromptConfig, variant_id: str, family: str | None = None) -> STMGPromptConfig:
    variant = get_variant(variant_id, family)
    result = canonical_config(config)
    if variant.experiment_family == "component_ablation":
        result.component_ablation = variant.variant_id
    else:
        result.component_ablation = None
    for key, value in variant.config_overrides.items():
        setattr(result, key, deepcopy(value))
    return result


def semantic_config(config: STMGPromptConfig | dict[str, Any]) -> dict[str, Any]:
    raw = config if isinstance(config, dict) else config.to_dict()
    value = {
        key: deepcopy(raw.get(key, SEMANTIC_CONFIG_DEFAULTS.get(key)))
        for key in (*FORMAL_PROTOCOL_FIELDS, *ARCHITECTURE_AND_LOSS_FIELDS)
    }
    value["graph_operator"] = (
        GRAPH_BI_DIFFUSION_DISPLAY
        if value.get("graph_operator") == GRAPH_BI_DIFFUSION
        else value.get("graph_operator")
    )
    return value


def config_diff(
    config: STMGPromptConfig | dict[str, Any],
    variant_id: str,
    family: str | None = None,
) -> dict[str, Any]:
    variant = get_variant(variant_id, family)
    base = semantic_config(canonical_config())
    effective = semantic_config(config)
    differences = {
        key: {"canonical": base.get(key), "effective": effective.get(key)}
        for key in effective
        if effective.get(key) != base.get(key)
    }
    expected = set(variant.expected_unique_diff)
    unexpected = sorted(set(differences) - expected)
    missing = sorted(expected - set(differences))
    return {
        "variant": variant.variant_id,
        "canonical_id": CANONICAL_ID,
        "effective_diff_count": len(differences),
        "expected_diff_fields": list(variant.expected_unique_diff),
        "actual_diff_fields": sorted(differences),
        "unexpected_diff_fields": unexpected,
        "missing_expected_diff_fields": missing,
        "unexpected_effective_diff_count": len(unexpected),
        "differences": differences,
        "passed": not unexpected and not missing,
    }


def assert_expected_diff(config: STMGPromptConfig, variant_id: str, family: str | None = None) -> dict[str, Any]:
    result = config_diff(config, variant_id, family)
    if not result["passed"]:
        raise RuntimeError(
            f"{variant_id} effective config differs from Canonical Full outside the formal protocol: "
            f"unexpected={result['unexpected_diff_fields']}, missing={result['missing_expected_diff_fields']}"
        )
    return result


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def canonical_directory(project_root: str | Path) -> Path:
    return Path(project_root) / CANONICAL_RESULT_ROOT / CANONICAL_ID


def directory_files(directory: str | Path) -> list[dict[str, Any]]:
    directory = Path(directory)
    required = ("config.json", "best_checkpoint.pt")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Canonical artifact is incomplete; missing {missing} in {directory}")
    return [{"path": path.name, "size_bytes": path.stat().st_size} for path in sorted(directory.iterdir()) if path.is_file()]


def write_reference(alias: str, alias_directory: str | Path, project_root: str | Path) -> Path:
    alias = alias.upper()
    if alias not in {"A0", "P0"}:
        raise ValueError("Only A0 and P0 may reference Canonical Full.")
    directory = canonical_directory(project_root)
    payload = {
        "alias": alias,
        "canonical_id": CANONICAL_ID,
        "canonical_directory": str(directory.resolve()),
        "checkpoint_path": "best_checkpoint.pt",
        "config_path": "config.json",
        "files": directory_files(directory),
        "trainable": False,
    }
    return write_json(Path(alias_directory) / "reference.json", payload)


def resolve_reference(reference_path: str | Path) -> Path:
    payload = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    if payload.get("trainable") is not False or payload.get("canonical_id") != CANONICAL_ID:
        raise ValueError(f"Invalid canonical reference: {reference_path}")
    directory = Path(payload["canonical_directory"])
    directory_files(directory)
    return directory


MATRIX_FIELDS = (
    "variant", "display_name", "experiment_family", "trainable", "canonical_reference",
    "vadsp_gate_mode", "graph_operator", "decoder_context_mode", "hidden_dim",
    "num_coupling_layers", "spatial_graph", "adaptive_graph", "diffusion",
    "macro_prompt", "macro_to_fine_cross", "reverse_cross", "cross_fusion", "st_prompt", "loss_name",
    "macro_prompt_len", "macro_prompt_pooling", "cross_fusion_recent_len", "fusion_mode", "st_prompt_mode",
    "st_prompt_use_node_identity",
    "granularity_weight_mode", "site_weight_mode", "seed",
)


def matrix_row(variant: ExperimentVariant) -> dict[str, Any]:
    config = apply_variant(canonical_config(), variant.variant_id, variant.experiment_family)
    graph_operator = semantic_config(config)["graph_operator"]
    return {
        "variant": variant.variant_id,
        "display_name": variant.display_name,
        "experiment_family": variant.experiment_family,
        "trainable": variant.trainable,
        "canonical_reference": variant.canonical_reference or "",
        "vadsp_gate_mode": config.vadsp_gate_mode,
        "graph_operator": graph_operator,
        "decoder_context_mode": config.decoder_context_mode,
        "hidden_dim": config.hidden_dim,
        "num_coupling_layers": config.num_coupling_layers,
        "spatial_graph": config.use_graph_in_temporal_encoder,
        "adaptive_graph": config.use_adaptive_graph,
        "diffusion": graph_operator == GRAPH_BI_DIFFUSION_DISPLAY,
        "macro_prompt": config.use_macro_prompt,
        "macro_to_fine_cross": config.use_cross_fusion and not config.disable_macro_to_fine_cross,
        "reverse_cross": config.use_cross_fusion and not config.disable_reverse_cross,
        "cross_fusion": config.use_cross_fusion,
        "st_prompt": config.use_st_prompt,
        "macro_prompt_len": config.macro_prompt_len,
        "macro_prompt_pooling": config.macro_prompt_pooling,
        "cross_fusion_recent_len": config.cross_fusion_recent_len,
        "fusion_mode": config.fusion_mode,
        "st_prompt_mode": config.st_prompt_mode,
        "st_prompt_use_node_identity": config.st_prompt_use_node_identity,
        "loss_name": config.loss_function,
        "granularity_weight_mode": config.granularity_weight_mode,
        "site_weight_mode": config.site_weight_mode,
        "seed": config.seed,
    }


def write_variant_matrix(path: str | Path, variants: Iterable[ExperimentVariant]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_FIELDS)
        writer.writeheader()
        for variant in variants:
            writer.writerow(matrix_row(variant))
    return path
