from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DOC_ROOT = PROJECT_ROOT / "custom_models" / "docs" / "benchmark_v2" / "E8"
ANALYSIS_ROOT = (
    PROJECT_ROOT
    / "custom_models"
    / "results"
    / "benchmark_v2"
    / "e8_prompt_cross_fusion_analysis_seed2026"
)
CURRENT_SCOPE_MANIFEST = (
    PROJECT_ROOT / "custom_models" / "docs" / "benchmark_v2" / "BATCH4" / "CURRENT_BATCH4_SCOPE26_MANIFEST.json"
)
BASE_PROTOCOL = PROJECT_ROOT / "custom_models" / "src" / "benchmark_v2" / "protocol" / "benchmark_protocol_v1.json"
BATCH4_PROFILE = (
    PROJECT_ROOT / "custom_models" / "src" / "benchmark_v2" / "training_profiles" / "uniform_train_batch4_v1.json"
)
INTERNAL_ROOT = (
    PROJECT_ROOT
    / "custom_models"
    / "results"
    / "st_mgprompt_component_ablation"
    / "component_ablation_fixed_dual_seed2026"
)
CANONICAL_ROOT = (
    PROJECT_ROOT
    / "custom_models"
    / "results"
    / "st_mgprompt_canonical"
    / "full_fixed_dual_keep_msmgdwu_seed2026"
)

INTERNAL_VARIANTS = ("A0", "A4", "A5", "A6", "A7")
INTERNAL_DISPLAY_NAMES = {
    "A0": "ST-MGPrompt A0",
    "A4": "A4 w/o Macro Prompt",
    "A5": "A5 w/o Fine-to-Coarse Reverse Cross",
    "A6": "A6 w/o Cross Fusion",
    "A7": "A7 w/o ST Prompt (Horizon Head)",
}
INTERNAL_TARGETS = {
    "A4": ("Macro Prompt", ("use_macro_prompt",)),
    "A5": ("Fine-to-Coarse / Reverse Cross", ("disable_reverse_cross",)),
    "A6": ("Bidirectional Cross Fusion", ("use_cross_fusion",)),
    "A7": ("ST Prompt and prompt-query prediction head", ("use_st_prompt", "decoder_input_strategy")),
}

EXTERNAL_MODEL_IDS = ("patchtst", "itransformer", "timexer", "multipatchformer", "timemixer", "timefilter")
EXTERNAL_DISPLAY_NAMES = {
    "patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "timexer": "TimeXer",
    "multipatchformer": "MultiPatchFormer",
    "timemixer": "TimeMixer",
    "timefilter": "TimeFilter",
}
EXTERNAL_ROLES = {
    "patchtst": "single-scale patch representation control",
    "itransformer": "cross-variable token interaction control",
    "timexer": "endogenous/exogenous interaction control",
    "multipatchformer": "multi-patch / multi-granularity representation control",
    "timemixer": "multi-scale temporal mixing control",
    "timefilter": "adaptive temporal/spatiotemporal patch filtering control",
}
EXCLUDED_MODEL_IDS = (
    "persistence", "gru", "dlinear", "crossformer", "graph_wavenet", "agcrn", "segrnn", "msgnet"
)
FORBIDDEN_SOURCE_TOKENS = (
    "common_loss", "common-loss", "commonloss", "/e5/", "\\e5\\", "results_smoke", "smoke",
    "batch32", "_archived_attempts", ".original_scope26_quarantine",
)
HORIZONS = (3, 6, 10)
METRICS = ("Score", "MAE", "RMSE", "R2")
LOWER_IS_BETTER = ("Score", "MAE", "RMSE")

EXPECTED_PROTOCOL = {
    "dataset_id": "SDWPF",
    "target_column": "Patv_raw",
    "input_power_column": "Patv_clean_for_input",
    "mask_column": "valid_target_mask",
    "lookback": 144,
    "max_pred_len": 10,
    "eval_horizons": [3, 6, 10],
    "split_ratio": [0.8, 0.1, 0.1],
    "split_mode": "strict_chronological",
    "train_sample_stride": 6,
    "val_sample_stride": 3,
    "test_sample_stride": 1,
    "default_seed": 2026,
    "physical_power_min_kw": 0.0,
    "physical_power_max_kw": 1500.0,
    "checkpoint_horizon": 10,
    "checkpoint_direction": "lower_is_better",
}
EXPECTED_BATCH = {
    "profile_id": "uniform_train_batch4_v1",
    "train_batch_size": 4,
    "val_batch_size": 4,
    "test_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "effective_train_batch_size": 4,
}

CAPABILITY_VALUES = ("YES", "NO", "NOT_APPLICABLE", "NOT_EXTRACTABLE", "NOT_VERIFIED")
WORKBOOK_SHEETS = (
    "README", "Evidence_Readiness", "Protocol_Audit", "Internal_Ablation",
    "Cross_Direction_Decomposition", "Original26_External_Context", "Mechanism_Capability_Matrix",
    "Representation_Similarity", "Representation_Shift", "Cross_Fusion_Statistics",
    "Macro_Prompt_Statistics", "ST_Prompt_Horizon", "Paired_Window_Analysis", "Conditional_Groups",
    "Diagnostic_SideEffect_Audit", "Artifact_Manifest",
)
