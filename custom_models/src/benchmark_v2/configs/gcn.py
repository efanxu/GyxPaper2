from __future__ import annotations

from typing import Any, Mapping

from .graph_common import resolve_graph_common


GCN_CONFIG = {
    "input_dim": 16,
    "hidden_dim": 64,
    "num_graph_layers": 2,
    "dropout": 0.1,
    "activation": "ReLU",
    "graph_support_names": ["A_gcn"],
    "temporal_projection": "shared Linear(144,10)",
    "output_projection": "shared Linear(64,1)",
}


def resolve_gcn_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    return {
        **resolve_graph_common("gcn", protocol, run_mode=run_mode),
        **GCN_CONFIG,
        "wrapper_path": (
            "custom_models/src/benchmark_v2/models/graph_models/gcn.py"
        ),
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e3_b.py",
        "primary_architecture_source": (
            "Kipf and Welling, Semi-Supervised Classification with Graph "
            "Convolutional Networks, ICLR 2017"
        ),
        "official_repository_identity": "https://github.com/tkipf/gcn",
        "config_resolution_reason": (
            "Prompt-frozen two-layer first-order GCN and minimal shared "
            "direct multi-horizon forecast adaptation; no recurrent encoder."
        ),
    }
