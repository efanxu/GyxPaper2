from __future__ import annotations

from typing import Any, Mapping

from ..errors import ContractError
from ..models.graph_models.adaptive_common import E3_C_POLICIES


def resolve_e3_c_common(
    model_id: str, protocol: Mapping[str, Any], *, run_mode: str
) -> dict[str, Any]:
    if model_id not in E3_C_POLICIES:
        raise ContractError(f"Unknown E3-C model: {model_id}")
    if int(protocol["lookback"]) != 144:
        raise ContractError("E3-C models require lookback=144.")
    if int(protocol["max_pred_len"]) != 10:
        raise ContractError("E3-C models require pred_len=10.")
    if int(protocol["feature_count"]) != 16:
        raise ContractError("E3-C models require 16 input features.")
    if protocol["future_observed_covariates"] != "disabled":
        raise ContractError("Future observed covariates must remain disabled.")
    if protocol["future_calendar_covariates"] != "disabled":
        raise ContractError("Future calendar covariates must remain disabled.")
    policy = E3_C_POLICIES[model_id]
    graph_propagation = model_id != "stid"
    return {
        "model_id": model_id,
        "run_mode": run_mode,
        "lookback": 144,
        "horizon": 10,
        "node_count": 134,
        "num_nodes": 134,
        "input_dim": 16,
        "input_contract": "(B,144,134,16)",
        "output_contract": "(B,134,10)",
        "node_semantics": "native_134_turbine_node_axis",
        "cross_node_interaction": graph_propagation,
        "uses_graph": graph_propagation,
        "uses_node_embedding": model_id in {
            "graph_wavenet",
            "mtgnn",
            "agcrn",
            "stid",
        },
        "uses_physical_support": policy["uses_physical_support"],
        "physical_support_names": list(policy["physical_support_names"]),
        "graph_support_names": list(policy["physical_support_names"]),
        "adaptive_graph_policy": policy["adaptive_graph_policy"],
        "node_identity_policy": policy["node_identity_policy"],
        "temporal_identity_policy": policy["temporal_identity_policy"],
        "adaptive_initialization_policy": policy[
            "adaptive_initialization_policy"
        ],
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
        "protocol_hash": protocol["protocol_hash"],
        "clean_room_implementation": True,
        "validation_search_performed": False,
        "test_result_used": False,
        "smoke_result_used_for_selection": False,
        "oom_driven_capacity_reduction": False,
        "initialization_policy": (
            "PyTorch Xavier/standard initialization after benchmark_v2 seed"
        ),
    }
