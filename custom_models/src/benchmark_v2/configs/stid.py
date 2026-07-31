from __future__ import annotations

from typing import Any, Mapping

from .e3_c_common import resolve_e3_c_common


STID_CONFIG = {
    "num_nodes": 134,
    "input_len": 144,
    "input_dim": 16,
    "output_len": 10,
    "embed_dim": 32,
    "num_layers": 3,
    "if_node": True,
    "node_dim": 32,
    "if_time_in_day": True,
    "time_of_day_size": 144,
    "time_of_day_dim": 32,
    "if_day_in_week": True,
    "day_of_week_size": 7,
    "day_of_week_dim": 32,
    "mlp_dropout": 0.15,
    "hidden_dim": 128,
    "timezone_policy": "SOURCE_NAIVE_UNCHANGED",
    "graph_propagation": False,
}


def resolve_stid_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    return {
        **resolve_e3_c_common("stid", protocol, run_mode=run_mode),
        **STID_CONFIG,
        "wrapper_path": (
            "custom_models/src/benchmark_v2/models/graph_models/stid.py"
        ),
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e3_c.py",
        "primary_architecture_source": (
            "Shao et al., Spatial-Temporal Identity: A Simple yet Effective "
            "Baseline for Multivariate Time Series Forecasting, CIKM 2022"
        ),
        "official_repository_identity": (
            "https://github.com/GestaltCogTeam/STID"
        ),
        "official_repository_license": "Apache-2.0",
        "config_resolution_reason": (
            "Prompt-frozen 32-dimensional series/node/time/day identities, "
            "three residual MLP layers and direct horizon regression; only "
            "the last observed historical timestamp is consumed."
        ),
    }
