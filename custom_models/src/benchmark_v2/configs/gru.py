from __future__ import annotations

from typing import Any, Mapping


GRU_CONFIG = {
    "model_id": "gru",
    "input_dim": 16,
    "hidden_dim": 64,
    "num_layers": 1,
    "bidirectional": False,
    "dropout": 0.0,
    "batch_first": True,
    "readout": "last_hidden_state",
    "prediction_head": "Linear(64,10)",
    "output_mode": "direct_multi_horizon",
    "uses_graph": False,
    "uses_node_embedding": False,
    "uses_time_mark": False,
    "uses_future_covariates": False,
    "uses_decoder_target": False,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "scheduler": None,
    "gradient_clip": None,
    "initialization_policy": "PyTorch default initialization after benchmark_v2 seed",
}


def resolve_gru_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    return {
        **GRU_CONFIG,
        "run_mode": run_mode,
        "horizon": int(protocol["max_pred_len"]),
        "lookback": int(protocol["lookback"]),
        "epochs": int(protocol["epochs"]),
        "patience": int(protocol["early_stopping_patience"]),
        "min_delta": float(protocol["early_stopping_min_delta"]),
        "seed": int(protocol["default_seed"]),
        "amp_enabled": bool(protocol["amp_enabled"]),
        "checkpoint_metric": "validation_official_score_h10",
        "checkpoint_direction": "lower_is_better",
        "loss": "masked_mse",
        "loss_space": "normalized_target_space",
        "protocol_hash": protocol["protocol_hash"],
    }
