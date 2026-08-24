from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DOC_ROOT = PROJECT_ROOT / "custom_models" / "docs" / "benchmark_v2" / "E7"
ANALYSIS_ROOT = (
    PROJECT_ROOT
    / "custom_models"
    / "results"
    / "benchmark_v2"
    / "e7_graph_mechanism_analysis_seed2026"
)
CURRENT_SCOPE_MANIFEST = (
    PROJECT_ROOT
    / "custom_models"
    / "docs"
    / "benchmark_v2"
    / "BATCH4"
    / "CURRENT_BATCH4_SCOPE26_MANIFEST.json"
)
BASE_PROTOCOL = (
    PROJECT_ROOT
    / "custom_models"
    / "src"
    / "benchmark_v2"
    / "protocol"
    / "benchmark_protocol_v1.json"
)
BATCH4_PROFILE = (
    PROJECT_ROOT
    / "custom_models"
    / "src"
    / "benchmark_v2"
    / "training_profiles"
    / "uniform_train_batch4_v1.json"
)
GRAPH_PROTOCOL = (
    PROJECT_ROOT
    / "custom_models"
    / "src"
    / "benchmark_v2"
    / "protocol"
    / "graph_protocol_v1.json"
)
NODE_ORDER = GRAPH_PROTOCOL.parent / "graph_v1" / "node_order_v1.json"
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

INTERNAL_VARIANTS = ("A0", "A1", "A2", "A3")
INTERNAL_DISPLAY_NAMES = {
    "A0": "ST-MGPrompt A0",
    "A1": "A1 w/o Spatial Graph",
    "A2": "A2 w/o Adaptive Graph",
    "A3": "A3 w/o Diffusion",
}
INTERNAL_TARGETS = {
    "A1": ("Spatial Graph", ("use_graph_in_temporal_encoder",)),
    "A2": ("Adaptive Graph", ("use_adaptive_graph",)),
    "A3": ("Bi-Diffusion", ("graph_operator",)),
}

EXTERNAL_MODEL_IDS = ("gcn", "stgcn", "dcrnn", "mtgnn", "stid", "tsmixer")
EXTERNAL_DISPLAY_NAMES = {
    "gcn": "GCN",
    "stgcn": "STGCN",
    "dcrnn": "DCRNN",
    "mtgnn": "MTGNN",
    "stid": "STID",
    "tsmixer": "TSMixer",
}
EXCLUDED_MODEL_IDS = (
    "persistence",
    "gru",
    "dlinear",
    "crossformer",
    "graph_wavenet",
    "agcrn",
    "segrnn",
    "msgnet",
)
FORBIDDEN_SOURCE_TOKENS = (
    "common_loss",
    "common-loss",
    "e5",
    "results_smoke",
    "smoke",
    "batch32",
    "_archived_attempts",
    ".original_scope26_quarantine",
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
EXPECTED_GRAPH = {
    "graph_id": "sdwpf_physical_knn_v1",
    "node_count": 134,
    "selected_k": 4,
}

WORKBOOK_SHEETS = (
    "README",
    "Evidence_Readiness",
    "Protocol_Audit",
    "Internal_Ablation",
    "Original26_External_Context",
    "Graph_Capability_Matrix",
    "Graph_Structure_Stats",
    "Prior_Learned_Overlap",
    "Distance_Weight_Analysis",
    "Node_Level_Analysis",
    "Conditional_Groups",
    "Diagnostic_SideEffect_Audit",
    "Artifact_Manifest",
)
