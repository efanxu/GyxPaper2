from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_16_FEATURES = [
    "Wspd",
    "Wdir",
    "Etmp",
    "Itmp",
    "Ndir",
    "Pab1",
    "Pab2",
    "Pab3",
    "Prtv",
    "T2m",
    "Sp",
    "RelH",
    "Wspd_w",
    "Wdir_w",
    "Tp",
    "Patv_clean_for_input",
]

FORBIDDEN_INPUT_COLS = {
    "TurbID",
    "Tmstamp",
    "Patv_raw",
    "valid_target_mask",
    "Modification_Reason",
}
FORBIDDEN_INPUT_SUBSTRINGS = ("anomaly", "audit", "mask", "imputation", "imputed")

COMPONENT_ABLATION_IDS = tuple(f"A{i}" for i in range(9))


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class STMGPromptConfig:
    protocol_version: str = "custom_model_stmgprompt_coupled_mask_aware_fair_v4_3_corrected"
    model_input_path: str = "dataset/sdwpf_model_input_base.parquet"
    eval_target_path: str = "dataset/sdwpf_eval_target.parquet"
    raw_aligned_path: str = "dataset/sdwpf_raw_aligned.parquet"
    location_path: str = "dataset/sdwpf_turb_location_elevation.csv"
    column_description_file: str = "dataset/列名含义.csv"

    turbine_id_col: str = "TurbID"
    timestamp_col: str = "Tmstamp"
    target_col: str = "Patv_raw"
    target_mask_col: str = "valid_target_mask"
    input_patv_col: str = "Patv_clean_for_input"
    num_nodes: int = 134
    feature_cols: list[str] = field(default_factory=lambda: list(DEFAULT_16_FEATURES))

    lookback: int = 144
    max_pred_len: int = 10
    eval_horizons: list[int] = field(default_factory=lambda: [3, 6, 10])
    split_ratios: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])

    batch_size: int = 32
    eval_batch_size: int = 64
    train_batch_size: int | None = None
    val_batch_size: int | None = None
    test_batch_size: int | None = None
    epochs: int = 20
    patience: int = 10
    early_stopping_min_delta: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 64
    dropout: float = 0.1
    seed: int = 2026
    gradient_clip_val: float = 5.0

    train_sample_stride: int = 6
    val_sample_stride: int = 3
    test_sample_stride: int = 1

    loss_function: str = "masked_score_aligned_hybrid"
    result_protocol: str = "max_pred_len_prefix_eval"
    decoder_input_strategy: str = "direct_multi_output_prompt_query"
    enable_physical_clip_eval: bool = True
    physical_power_min_kw: float = 0.0
    physical_power_max_kw: float = 1500.0
    physical_clip_protocol: str = "uniform_eval_clip_all_models"
    checkpoint_selection_metric: str = "val_official_score_h10"
    checkpoint_selection_mode: str = "min"
    primary_val_horizon: int = 10

    use_vadsp: bool = False
    volatility_source_cols: list[str] = field(default_factory=lambda: ["Wspd", "Patv_clean_for_input"])
    volatility_mode: str = "per_node"
    vol_window: int = 6
    fine_kernel_size: int = 3
    coarse_windows: list[int] = field(default_factory=lambda: [6, 18, 36])
    use_train_robust_volatility: bool = True
    volatility_scale_eps: float = 1e-6
    vadsp_tau_min: float = 0.5
    vadsp_tau_max: float = 5.0
    vadsp_residual_scale: float = 0.1
    vadsp_branch_beta_init: float = 0.05
    vadsp_gate_mode: str = "dynamic"
    vadsp_random_seed: int = 2026
    use_trend_prior_graph: bool = False
    use_adaptive_graph: bool = False
    graph_tag: str = "trend_prior_v4_2"
    graph_output_root: str = "custom_models/graphs"
    macro_graph_source: str = "causal_trend_similarity"
    micro_graph_source: str = "distance_delta_wspd"
    macro_top_k: int = 10
    micro_top_k: int = 5
    macro_alpha: float = 0.7
    micro_alpha: float = 0.3
    macro_trend_window: int = 36
    micro_delta_window: int = 1
    macro_similarity_metric: str = "pearson"
    micro_similarity_metric: str = "cosine"
    trend_source_col: str = "Patv_raw"
    trend_method: str = "causal_moving_average"
    trend_window: int = 36
    trend_ema_alpha: float = 0.1
    graph_similarity_metric: str = "pearson"
    graph_top_k: int = 5
    graph_alpha: float = 0.7
    graph_coordinate_cols: list[str] = field(default_factory=lambda: ["x", "y"])
    graph_use_elevation: bool = False
    graph_missing_location_policy: str = "error"
    graph_self_loop: bool = False
    graph_distance_sigma: float | None = None
    graph_fusion_mode: str = "multiply"
    graph_operator: str = "simple"
    diffusion_order_micro: int = 2
    diffusion_order_macro: int = 2
    diffusion_use_bidirectional: bool = True
    diffusion_beta_init: float = 0.05
    node_embed_dim: int = 10
    adaptive_graph_temperature: float = 1.0
    use_graph_temporal_encoder: bool = False
    use_graph_in_temporal_encoder: bool = True
    use_temporal_attention: bool = False
    temporal_attention_heads: int = 4
    fine_tcn_kernel_size: int = 3
    coarse_tcn_kernel_size: int = 5
    fine_tcn_dilations: list[int] = field(default_factory=lambda: [1, 2])
    coarse_tcn_dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    use_stmg_coupling_block: bool = False
    num_coupling_layers: int = 1
    coupling_mode: str = "graph_then_cross"
    use_macro_prompt: bool = False
    use_cross_fusion: bool = True
    use_st_prompt: bool = False
    decoder_context_mode: str = "last_state"
    decoder_history_len: int | None = None
    macro_prompt_len: int = 4
    cross_attention_heads: int = 4
    cross_fusion_recent_len: int = 24
    fusion_mode: str = "cross"
    disable_reverse_cross: bool = False
    use_msmg_dwu: bool = False
    loss_protocol: str = "fair_main"
    msmg_base_loss: str = "smooth_l1"
    msmg_lambda_site: float = 0.2
    msmg_ema_alpha: float = 0.9
    msmg_node_weight_clip: tuple[float, float] = (0.5, 3.0)
    granularity_weight_mode: str = "uncertainty_precision"
    site_weight_mode: str = "dynamic"
    difficulty_gamma: float = 1.0
    difficulty_rate_gamma: float = 1.0
    difficulty_temperature: float = 1.0
    granularity_weight_clip: tuple[float, float] = (0.5, 3.0)

    component_ablation: str | None = None

    # Formal Batch4 provenance fields.  They are intentionally data fields in
    # the config so the runner can write the exact contract it executed; they
    # do not change model construction or the historical protocol.
    variant: str | None = None
    model_id: str | None = None
    definition: str | None = None
    training_role: str | None = None
    training_batch_profile_id: str | None = None
    checkpoint_copied: bool = False
    metrics_copied: bool = False
    warm_started_from_historical_a8: bool = False
    precision_policy: str | None = None

    model_name: str = "STMGPrompt_TemporalOnly"
    run_id: str | None = None
    output_root: str = "custom_models/results/st_mgprompt"
    device: str = "auto"
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    amp_enabled: bool = False
    amp_dtype: str = "float16"
    diagnostics_level: str = "standard"
    prediction_accumulation: str = "full"
    windows_safe_mode: bool = False
    smoke: bool = False
    smoke_use_synthetic: bool = True
    smoke_num_time_steps: int = 240

    input_scaler_type: str = "standard"
    target_scaler_type: str = "standard"

    @property
    def eligible_for_fair_main_table(self) -> bool:
        ineligible_models = {
            "STMGPrompt_DynamicPatching",
            "STMGPrompt_TrendPriorGraph",
            "STMGPrompt_GraphTemporalSmoke",
            "STMGPrompt_SerialGraphThenFusion",
            "STMGPrompt_CouplingCrossFusion",
            "STMGPrompt_Full_MSMGDWU",
            "STMGPrompt_Full_MSMGDWU_DiffusionHistory",
            "STMGPrompt_ComponentAblation",
        }
        return (
            self.model_name not in ineligible_models
            and not self.use_msmg_dwu
            and self.loss_protocol == "fair_main"
        )

    def validate(self) -> None:
        if self.target_col != "Patv_raw":
            raise ValueError("ST-MGPrompt v4.2 Step 0 target_col must be Patv_raw.")
        if self.target_mask_col != "valid_target_mask":
            raise ValueError("target_mask_col must be valid_target_mask.")
        if self.input_patv_col not in self.feature_cols:
            raise ValueError("Patv_clean_for_input must be included as a historical input feature.")
        if len(self.feature_cols) != len(set(self.feature_cols)):
            raise ValueError("feature_cols contains duplicates.")
        forbidden = [c for c in self.feature_cols if is_forbidden_input_col(c)]
        if forbidden:
            raise ValueError(f"Forbidden model input columns: {forbidden}")
        if self.lookback <= 0 or self.max_pred_len <= 0:
            raise ValueError("lookback and max_pred_len must be positive.")
        if any(h <= 0 for h in self.eval_horizons) or max(self.eval_horizons) > self.max_pred_len:
            raise ValueError("eval_horizons must be positive and <= max_pred_len.")
        if len(self.split_ratios) != 3 or any(r <= 0 for r in self.split_ratios):
            raise ValueError("split_ratios must contain three positive values.")
        if abs(sum(self.split_ratios) - 1.0) > 1e-6:
            raise ValueError("split_ratios must sum to 1.0.")
        if min(self.train_sample_stride, self.val_sample_stride, self.test_sample_stride) <= 0:
            raise ValueError("sample strides must be positive prediction-start strides.")
        if self.loss_protocol == "fair_main" and self.loss_function != "masked_score_aligned_hybrid":
            raise ValueError("fair_main protocol must use masked_score_aligned_hybrid.")
        if self.enable_physical_clip_eval:
            if self.physical_power_min_kw is None or self.physical_power_max_kw is None:
                raise ValueError("Physical eval clipping requires explicit min/max kW values.")
            if self.physical_power_min_kw > self.physical_power_max_kw:
                raise ValueError("physical_power_min_kw must be <= physical_power_max_kw.")
            if self.physical_clip_protocol != "uniform_eval_clip_all_models":
                raise ValueError("physical_clip_protocol must be uniform_eval_clip_all_models.")
        if self.primary_val_horizon not in self.eval_horizons:
            raise ValueError("primary_val_horizon must be included in eval_horizons.")
        expected_metric = f"val_official_score_h{self.primary_val_horizon}"
        if self.checkpoint_selection_metric != expected_metric:
            raise ValueError(f"checkpoint_selection_metric must be {expected_metric}.")
        if self.checkpoint_selection_mode != "min":
            raise ValueError("checkpoint_selection_mode must be min.")
        if self.use_msmg_dwu and self.loss_protocol == "fair_main":
            raise ValueError("MS-MG-DWU cannot be enabled in fair_main protocol.")
        if self.use_msmg_dwu and self.loss_function != "msmg_dwu_loss":
            raise ValueError("MS-MG-DWU must use loss_function=msmg_dwu_loss.")
        if self.loss_function == "msmg_dwu_loss" and not self.use_msmg_dwu:
            raise ValueError("loss_function=msmg_dwu_loss requires use_msmg_dwu=True.")
        if self.component_ablation is not None and self.component_ablation not in COMPONENT_ABLATION_IDS:
            raise ValueError(f"component_ablation must be one of {COMPONENT_ABLATION_IDS}.")
        if self.component_ablation is not None and self.model_name != "STMGPrompt_ComponentAblation":
            raise ValueError("Component ablations must use model_name=STMGPrompt_ComponentAblation.")
        if self.model_name in {"STMGPrompt_Full_MSMGDWU", "STMGPrompt_Full_MSMGDWU_DiffusionHistory"}:
            if not self.use_msmg_dwu or self.loss_function != "msmg_dwu_loss":
                raise ValueError(f"{self.model_name} must enable MS-MG-DWU loss.")
            if self.loss_protocol != "method_full":
                raise ValueError(f"{self.model_name} must use loss_protocol=method_full.")
        if self.msmg_base_loss not in {"mae", "mse", "smooth_l1"}:
            raise ValueError("msmg_base_loss must be mae, mse, or smooth_l1.")
        if self.granularity_weight_mode not in {"static", "uncertainty_precision", "difficulty_rate"}:
            raise ValueError("granularity_weight_mode must be static, uncertainty_precision, or difficulty_rate.")
        if self.site_weight_mode not in {"static", "dynamic"}:
            raise ValueError("site_weight_mode must be static or dynamic.")
        if self.difficulty_temperature <= 0:
            raise ValueError("difficulty_temperature must be positive.")
        if len(self.granularity_weight_clip) != 2:
            raise ValueError("granularity_weight_clip must have two values.")
        if self.granularity_weight_clip[0] <= 0 or self.granularity_weight_clip[1] < self.granularity_weight_clip[0]:
            raise ValueError("granularity_weight_clip must satisfy 0 < min <= max.")
        if self.msmg_lambda_site < 0:
            raise ValueError("msmg_lambda_site must be non-negative.")
        if not (0.0 <= self.msmg_ema_alpha < 1.0):
            raise ValueError("msmg_ema_alpha must be in [0, 1).")
        if len(self.msmg_node_weight_clip) != 2:
            raise ValueError("msmg_node_weight_clip must have two values.")
        if self.msmg_node_weight_clip[0] <= 0 or self.msmg_node_weight_clip[1] < self.msmg_node_weight_clip[0]:
            raise ValueError("msmg_node_weight_clip must satisfy 0 < min <= max.")
        if self.volatility_mode not in {"per_node", "global"}:
            raise ValueError("volatility_mode must be per_node or global.")
        if self.vol_window <= 0:
            raise ValueError("vol_window must be positive.")
        if self.fine_kernel_size <= 0:
            raise ValueError("fine_kernel_size must be positive.")
        if any(w <= 0 for w in self.coarse_windows):
            raise ValueError("coarse_windows must contain positive values.")
        if self.volatility_scale_eps <= 0:
            raise ValueError("volatility_scale_eps must be positive.")
        if not (0.0 < self.vadsp_tau_min <= self.vadsp_tau_max):
            raise ValueError("vadsp_tau_min/tau_max must satisfy 0 < min <= max.")
        if self.vadsp_branch_beta_init < 0:
            raise ValueError("vadsp_branch_beta_init must be non-negative.")
        if self.vadsp_gate_mode not in {
            "dynamic",
            "fine_only",
            "coarse_only",
            "fixed_dual",
            "random_gate",
            "shuffled_volatility",
        }:
            raise ValueError(
                "vadsp_gate_mode must be dynamic, fine_only, coarse_only, fixed_dual, "
                "random_gate, or shuffled_volatility."
            )
        if self.macro_graph_source != "causal_trend_similarity":
            raise ValueError("macro_graph_source must be causal_trend_similarity.")
        if self.micro_graph_source not in {"distance_delta_wspd", "distance_high_frequency"}:
            raise ValueError("micro_graph_source must be distance_delta_wspd or distance_high_frequency.")
        if self.macro_top_k <= 0 or self.micro_top_k <= 0:
            raise ValueError("macro_top_k and micro_top_k must be positive.")
        if not (0.0 <= self.macro_alpha <= 1.0 and 0.0 <= self.micro_alpha <= 1.0):
            raise ValueError("macro_alpha and micro_alpha must be in [0, 1].")
        if self.macro_similarity_metric not in {"pearson", "cosine"}:
            raise ValueError("macro_similarity_metric must be pearson or cosine.")
        if self.micro_similarity_metric not in {"pearson", "cosine"}:
            raise ValueError("micro_similarity_metric must be pearson or cosine.")
        if not self.graph_coordinate_cols:
            raise ValueError("graph_coordinate_cols must not be empty.")
        if self.graph_missing_location_policy not in {"error", "identity_for_synthetic_smoke"}:
            raise ValueError("graph_missing_location_policy must be error or identity_for_synthetic_smoke.")
        if self.trend_source_col not in {self.target_col, *self.feature_cols}:
            raise ValueError("trend_source_col must be Patv_raw or one of the configured input features.")
        if self.trend_method not in {"causal_moving_average", "causal_ema", "almon_smooth"}:
            raise ValueError("trend_method must be causal_moving_average, causal_ema, or almon_smooth.")
        if self.trend_window <= 0:
            raise ValueError("trend_window must be positive.")
        if not (0.0 < self.trend_ema_alpha <= 1.0):
            raise ValueError("trend_ema_alpha must be in (0, 1].")
        if self.graph_similarity_metric not in {"pearson", "cosine"}:
            raise ValueError("graph_similarity_metric must be pearson or cosine.")
        if self.graph_top_k <= 0:
            raise ValueError("graph_top_k must be positive.")
        if not (0.0 <= self.graph_alpha <= 1.0):
            raise ValueError("graph_alpha must be in [0, 1].")
        if self.graph_fusion_mode != "multiply":
            raise ValueError("Step 2 main graph fusion mode must be multiply.")
        if self.graph_operator not in {"simple", "bidirectional_diffusion"}:
            raise ValueError("graph_operator must be simple or bidirectional_diffusion.")
        if not (1 <= self.diffusion_order_micro <= 3 and 1 <= self.diffusion_order_macro <= 3):
            raise ValueError("diffusion_order_micro/macro must be in [1, 3].")
        if self.diffusion_beta_init < 0:
            raise ValueError("diffusion_beta_init must be non-negative.")
        if self.node_embed_dim <= 0:
            raise ValueError("node_embed_dim must be positive.")
        if self.adaptive_graph_temperature <= 0:
            raise ValueError("adaptive_graph_temperature must be positive.")
        if self.temporal_attention_heads <= 0:
            raise ValueError("temporal_attention_heads must be positive.")
        if self.fine_tcn_kernel_size <= 0 or self.coarse_tcn_kernel_size <= 0:
            raise ValueError("TCN kernel sizes must be positive.")
        if not self.fine_tcn_dilations or not self.coarse_tcn_dilations:
            raise ValueError("TCN dilation lists must not be empty.")
        if any(d <= 0 for d in [*self.fine_tcn_dilations, *self.coarse_tcn_dilations]):
            raise ValueError("TCN dilations must be positive.")
        bad_vol_cols = [c for c in self.volatility_source_cols if c not in self.feature_cols]
        if bad_vol_cols:
            raise ValueError(f"volatility_source_cols must be in feature_cols: {bad_vol_cols}")
        forbidden_vol_cols = [c for c in self.volatility_source_cols if is_forbidden_input_col(c)]
        if forbidden_vol_cols:
            raise ValueError(f"Forbidden volatility source columns: {forbidden_vol_cols}")
        if self.model_name == "STMGPrompt_TemporalOnly_VADSP" and not self.use_vadsp:
            raise ValueError("STMGPrompt_TemporalOnly_VADSP must enable VADSP.")
        if self.model_name == "STMGPrompt_DynamicPatching" and not self.use_vadsp:
            raise ValueError("STMGPrompt_DynamicPatching must enable VADSP.")
        coupled_model_names = {
            "STMGPrompt_FairFull",
            "STMGPrompt_FairFull_Diffusion",
            "STMGPrompt_FairFull_HistoryDecoder",
            "STMGPrompt_FairFull_DiffusionHistory",
            "STMGPrompt_CouplingCrossFusion",
            "STMGPrompt_Full_MSMGDWU",
            "STMGPrompt_Full_MSMGDWU_DiffusionHistory",
            "STMGPrompt_ComponentAblation",
        }
        if self.model_name in coupled_model_names and not self.use_stmg_coupling_block:
            raise ValueError(f"{self.model_name} must enable ST-MG Coupling Block.")
        if self.num_coupling_layers <= 0:
            raise ValueError("num_coupling_layers must be positive.")
        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("batch_size and eval_batch_size must be positive.")
        for name in ["train_batch_size", "val_batch_size", "test_batch_size"]:
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None.")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative.")
        if self.persistent_workers and self.num_workers <= 0:
            raise ValueError("persistent_workers requires num_workers > 0.")
        if self.amp_dtype not in {"float16"}:
            raise ValueError("amp_dtype must be float16.")
        if self.diagnostics_level not in {"none", "minimal", "standard", "full"}:
            raise ValueError("diagnostics_level must be none, minimal, standard, or full.")
        if self.prediction_accumulation not in {"full", "streaming"}:
            raise ValueError("prediction_accumulation must be full or streaming.")
        if self.coupling_mode != "graph_then_cross":
            raise ValueError("Step 4 coupling_mode must be graph_then_cross.")
        if self.macro_prompt_len <= 0:
            raise ValueError("macro_prompt_len must be positive.")
        if self.cross_attention_heads <= 0:
            raise ValueError("cross_attention_heads must be positive.")
        if self.cross_fusion_recent_len <= 0:
            raise ValueError("cross_fusion_recent_len must be positive.")
        if self.fusion_mode not in {"cross", "add", "concat"}:
            raise ValueError("fusion_mode must be cross, add, or concat.")
        if self.decoder_context_mode not in {"last_state", "full_history_cross_attention"}:
            raise ValueError("decoder_context_mode must be last_state or full_history_cross_attention.")
        if self.decoder_input_strategy not in {
            "direct_multi_output_prompt_query",
            "direct_multi_output_horizon_head",
        }:
            raise ValueError("Unsupported decoder_input_strategy.")
        if self.decoder_history_len is not None and self.decoder_history_len <= 0:
            raise ValueError("decoder_history_len must be positive or None.")
        fair_full_models = {
            "STMGPrompt_FairFull",
            "STMGPrompt_FairFull_Diffusion",
            "STMGPrompt_FairFull_HistoryDecoder",
            "STMGPrompt_FairFull_DiffusionHistory",
        }
        if self.model_name in fair_full_models and self.loss_function != "masked_score_aligned_hybrid":
            raise ValueError(f"{self.model_name} must use masked_score_aligned_hybrid.")
        if self.model_name in {"STMGPrompt_Full_MSMGDWU", "STMGPrompt_Full_MSMGDWU_DiffusionHistory"} and self.eligible_for_fair_main_table:
            raise ValueError(f"{self.model_name} must be ineligible for fair main table.")
        if self.model_name == "STMGPrompt_FairFull_Diffusion" and self.graph_operator != "bidirectional_diffusion":
            raise ValueError("STMGPrompt_FairFull_Diffusion must use graph_operator=bidirectional_diffusion.")
        if self.model_name == "STMGPrompt_FairFull_HistoryDecoder" and self.decoder_context_mode != "full_history_cross_attention":
            raise ValueError("STMGPrompt_FairFull_HistoryDecoder must use full_history_cross_attention.")
        if self.model_name in {"STMGPrompt_FairFull_DiffusionHistory", "STMGPrompt_Full_MSMGDWU_DiffusionHistory"}:
            if self.graph_operator != "bidirectional_diffusion":
                raise ValueError(f"{self.model_name} must use graph_operator=bidirectional_diffusion.")
            if self.decoder_context_mode != "full_history_cross_attention":
                raise ValueError(f"{self.model_name} must use full_history_cross_attention.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["eligible_for_fair_main_table"] = self.eligible_for_fair_main_table
        return data

    def save_json(self, path: str | Path) -> None:
        path = self.resolve_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def resolve_path(self, path: str | Path) -> Path:
        return resolve_project_path(path)

    @classmethod
    def from_json(cls, path: str | Path) -> "STMGPromptConfig":
        data = json.loads(resolve_project_path(path).read_text(encoding="utf-8"))
        data.pop("eligible_for_fair_main_table", None)
        return cls(**data)


def is_forbidden_input_col(col: str) -> bool:
    if col in FORBIDDEN_INPUT_COLS:
        return True
    lower = col.lower()
    return any(token in lower for token in FORBIDDEN_INPUT_SUBSTRINGS)


def apply_component_ablation(config: STMGPromptConfig, variant: str) -> STMGPromptConfig:
    """Apply the canonical Fixed-Dual A0-A8 component protocol from one source."""
    from .experiment_protocol import apply_variant

    resolved = apply_variant(config, variant, family="component_ablation")
    for key, value in asdict(resolved).items():
        setattr(config, key, value)
    return config
