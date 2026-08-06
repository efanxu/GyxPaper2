from __future__ import annotations

from typing import Any, Mapping

from ..errors import ContractError


def resolve_graph_common(
    model_id: str, protocol: Mapping[str, Any], *, run_mode: str
) -> dict[str, Any]:
    if int(protocol["lookback"]) != 144:
        raise ContractError("E3-B graph models require lookback=144.")
    if int(protocol["max_pred_len"]) != 10:
        raise ContractError("E3-B graph models require pred_len=10.")
    if int(protocol["feature_count"]) != 16:
        raise ContractError("E3-B graph models require 16 input features.")
    if protocol["future_observed_covariates"] != "disabled":
        raise ContractError("Future observed covariates must remain disabled.")
    if protocol["future_calendar_covariates"] != "disabled":
        raise ContractError("Future calendar covariates must remain disabled.")
    return {
        "model_id": model_id,
        "run_mode": run_mode,
        "lookback": 144,
        "horizon": 10,
        "node_count": 134,
        "input_dim": 16,
        "input_contract": "(B,144,134,16)",
        "output_contract": "(B,134,10)",
        "node_semantics": "native_134_turbine_graph",
        "cross_node_interaction": True,
        "uses_graph": True,
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
        "clean_room_implementation": True,
        "validation_search_performed": False,
        "test_result_used": False,
        "smoke_result_used_for_selection": False,
        "oom_driven_capacity_reduction": False,
        "initialization_policy": (
            "PyTorch Xavier/standard initialization after benchmark_v2 seed"
        ),
    }
