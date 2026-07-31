from __future__ import annotations

from typing import Any, Mapping

from .e3_c_common import resolve_e3_c_common


MTGNN_CONFIG = {
    "num_nodes": 134,
    "input_dim": 16,
    "horizon": 10,
    "seq_length": 144,
    "gcn_true": True,
    "buildA_true": True,
    "predefined_A": "not_consumed",
    "static_feat": None,
    "gcn_depth": 2,
    "dropout": 0.3,
    "subgraph_size": 20,
    "node_dim": 40,
    "dilation_exponential": 1,
    "layers": 3,
    "conv_channels": 32,
    "residual_channels": 32,
    "skip_channels": 64,
    "end_channels": 128,
    "propalpha": 0.05,
    "tanhalpha": 3.0,
    "layer_norm_affine": True,
    "num_split": 1,
    "node_sampling": False,
    "curriculum_learning": False,
    "inception_kernels": [2, 3, 6, 7],
    "receptive_field": 19,
}


def resolve_mtgnn_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    return {
        **resolve_e3_c_common("mtgnn", protocol, run_mode=run_mode),
        **MTGNN_CONFIG,
        "wrapper_path": (
            "custom_models/src/benchmark_v2/models/graph_models/mtgnn.py"
        ),
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e3_c.py",
        "primary_architecture_source": (
            "Wu et al., Connecting the Dots: Multivariate Time Series "
            "Forecasting with Graph Neural Networks, KDD 2020"
        ),
        "official_repository_identity": "https://github.com/nnzhan/MTGNN",
        "official_repository_license": "MIT",
        "config_resolution_reason": (
            "Prompt-frozen learned directed top-k graph, three dilated "
            "inception layers and depth-two bidirectional mix-hop; canonical "
            "buildA_true consumes no predefined physical graph."
        ),
    }
