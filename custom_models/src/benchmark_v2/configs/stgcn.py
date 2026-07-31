from __future__ import annotations

from typing import Any, Mapping

from .graph_common import resolve_graph_common


STGCN_CONFIG = {
    "input_dim": 16,
    "num_st_blocks": 2,
    "temporal_kernel_size": 3,
    "graph_chebyshev_order": 3,
    "temporal_channels": 64,
    "graph_channels": 16,
    "output_channels": 64,
    "output_hidden": 128,
    "dropout": 0.3,
    "temporal_gate": "GLU",
    "graph_activation": "ReLU",
    "remaining_temporal_length": 136,
    "lambda_max_policy": "frozen_2.0_graph_protocol_v1",
    "graph_support_names": ["L_tilde"],
}


def resolve_stgcn_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    return {
        **resolve_graph_common("stgcn", protocol, run_mode=run_mode),
        **STGCN_CONFIG,
        "wrapper_path": (
            "custom_models/src/benchmark_v2/models/graph_models/stgcn.py"
        ),
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e3_b.py",
        "primary_architecture_source": (
            "Yu, Yin, and Zhu, Spatio-Temporal Graph Convolutional Networks: "
            "A Deep Learning Framework for Traffic Forecasting, IJCAI 2018"
        ),
        "official_repository_identity": (
            "https://github.com/VeritasYin/STGCN_IJCAI-18"
        ),
        "config_resolution_reason": (
            "Classic two-block temporal-GLU -> Chebyshev graph convolution "
            "-> temporal-GLU identity, with the prompt-frozen shared output "
            "collapse for lookback 144 and horizon 10."
        ),
    }
