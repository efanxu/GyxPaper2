from __future__ import annotations

from typing import Any, Mapping

from ..errors import ContractError


def resolve_common(
    model_id: str, protocol: Mapping[str, Any], *, run_mode: str
) -> dict[str, Any]:
    if int(protocol["lookback"]) != 144:
        raise ContractError("E1-B TSLib configs require frozen lookback=144.")
    if int(protocol["max_pred_len"]) != 10:
        raise ContractError("E1-B TSLib configs require frozen pred_len=10.")
    if int(protocol["feature_count"]) != 16:
        raise ContractError("E1-B TSLib configs require frozen enc_in=16.")
    if protocol["future_observed_covariates"] != "disabled":
        raise ContractError("Future observed covariates must remain disabled.")
    if protocol["future_calendar_covariates"] != "disabled":
        raise ContractError("Future calendar covariates must remain disabled.")
    return {
        "model_id": model_id,
        "task_name": "long_term_forecast",
        "run_mode": run_mode,
        "seq_len": 144,
        "pred_len": 10,
        "enc_in": 16,
        "input_mode": "multivariate",
        "output_mode": "single_target",
        "target_column": "Patv_raw",
        "selected_power_feature": "Patv_clean_for_input",
        "output_policy": "select_power_associated_channel",
        "node_semantics": "node_shared",
        "cross_node_interaction": False,
        "uses_graph": False,
        "uses_node_embedding": False,
        "uses_future_target": False,
        "uses_future_observed_covariates": False,
        "uses_future_calendar_covariates": False,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "scheduler": None,
        "gradient_clip": None,
        "epochs": int(protocol["epochs"]),
        "patience": int(protocol["early_stopping_patience"]),
        "min_delta": float(protocol["early_stopping_min_delta"]),
        "seed": int(protocol["default_seed"]),
        "amp_enabled": bool(protocol["amp_enabled"]),
        "loss": "masked_mse",
        "loss_space": "normalized_target_space",
        "checkpoint_metric": "validation_official_score_h10",
        "checkpoint_direction": "lower_is_better",
        "protocol_id": protocol["protocol_id"],
        "upstream_project": "THUML/Time-Series-Library",
        "upstream_license_path": "Time-Series-Library/LICENSE",
        "source_modified": False,
        "validation_search_performed": False,
        "test_result_used": False,
        "initialization_policy": (
            "Upstream TSLib initialization after benchmark_v2 seed"
        ),
    }
