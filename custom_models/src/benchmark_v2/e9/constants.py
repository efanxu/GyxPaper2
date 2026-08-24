from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DOC_ROOT = PROJECT_ROOT / "custom_models" / "docs" / "benchmark_v2" / "E9"
TRANSFER_ROOT = PROJECT_ROOT / "custom_models" / "results" / "benchmark_v2" / "msmg_dwu_transfer_seed2026"
CONTROL_ROOT = PROJECT_ROOT / "custom_models" / "results" / "benchmark_v2_uniform_bs4"
SMOKE_ROOT = PROJECT_ROOT / "custom_models" / "results_smoke" / "benchmark_v2" / "e9_msmg_dwu_transfer"
PREFLIGHT_ROOT = PROJECT_ROOT / "custom_models" / "logs" / "benchmark_v2" / "e9" / "preflight"
SCOPE_MANIFEST = PROJECT_ROOT / "custom_models" / "docs" / "benchmark_v2" / "BATCH4" / "CURRENT_BATCH4_SCOPE26_MANIFEST.json"
PROTOCOL_PATH = PROJECT_ROOT / "custom_models" / "src" / "benchmark_v2" / "protocol" / "benchmark_protocol_v1.json"
BATCH4_PROFILE_PATH = PROJECT_ROOT / "custom_models" / "src" / "benchmark_v2" / "training_profiles" / "uniform_train_batch4_v1.json"

MODEL_IDS = ("lightts", "tide", "patchtst", "itransformer", "dcrnn", "mtgnn")
DISPLAY_NAMES = {
    "lightts": "LightTS", "tide": "TiDE", "patchtst": "PatchTST",
    "itransformer": "iTransformer", "dcrnn": "DCRNN", "mtgnn": "MTGNN",
}
ARCHITECTURE_FAMILIES = {
    "lightts": "lightweight", "tide": "MLP", "patchtst": "patch-transformer",
    "itransformer": "cross-variable-transformer", "dcrnn": "recurrent-diffusion-graph",
    "mtgnn": "adaptive-graph-learning",
}
EXCLUDED_MODEL_IDS = (
    "persistence", "gru", "dlinear", "crossformer", "graph_wavenet", "agcrn", "segrnn", "msgnet",
)
FORBIDDEN_SOURCE_TOKENS = (
    "common_loss", "common-loss", "commonloss", "/e5/", "\\e5\\", "results_smoke",
    "_archived_attempts", ".original_scope26_quarantine",
)
HORIZONS = (3, 6, 10)
METRICS = ("Score", "MAE", "RMSE", "R2")
LOSS_ID = "msmg_dwu_loss"
CONTROL_LOSS_ID = "masked_mse"
TRAINING_PROFILE_ID = "uniform_train_batch4_v1"
E9_SCOPE_ID = "e9_msmg_dwu_transfer_seed2026"

LOSS_PROFILE = {
    "loss_id": LOSS_ID,
    "source_symbol": "st_mgprompt.losses.MSMGDWULoss",
    "adapter_symbol": "benchmark_v2.losses.BenchmarkMSMGDWULoss",
    "base_loss": "smooth_l1",
    "lambda_site": 0.2,
    "ema_alpha": 0.9,
    "node_weight_clip": [0.5, 3.0],
    "granularity_weight_mode": "difficulty_rate",
    "site_weight_mode": "dynamic",
    "difficulty_gamma": 1.0,
    "difficulty_rate_gamma": 1.0,
    "difficulty_temperature": 1.0,
    "granularity_weight_clip": [0.5, 3.0],
    "eval_horizons": [3, 6, 10],
    "num_nodes": 134,
}

REQUIRED_TRAIN_ARTIFACTS = (
    "resolved_config.json", "effective_config.json", "protocol_check.json", "model_summary.json",
    "best_checkpoint.pt", "last_checkpoint.pt", "train_log.csv", "metrics_eval_h3.json",
    "metrics_eval_h6.json", "metrics_eval_h10.json", "metrics.csv", "prediction_metadata.json",
    "run_status.json",
)

WORKBOOK_SHEETS = (
    "README", "Portability_Audit", "Loss_Input_Contract", "Evidence_Readiness",
    "Original_Control_Audit", "Pairing_Audit", "Config_Diff", "Main_Transfer_Results",
    "H3_Comparison", "H6_Comparison", "H10_Comparison", "Family_Summary",
    "Training_Dynamics", "Loss_Components", "Efficiency_Overhead",
    "Optional_A0_A8_Reference", "Preflight_Inventory", "Failures", "Artifact_Manifest",
)
