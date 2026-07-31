from __future__ import annotations

from typing import Any, Mapping

from .e3_c_common import resolve_e3_c_common


GRAPH_WAVENET_CONFIG = {
    "num_nodes": 134,
    "input_dim": 16,
    "horizon": 10,
    "dropout": 0.3,
    "residual_channels": 32,
    "dilation_channels": 32,
    "skip_channels": 256,
    "end_channels": 512,
    "kernel_size": 2,
    "blocks": 4,
    "layers_per_block": 2,
    "diffusion_order": 2,
    "gcn_enabled": True,
    "adaptive_adjacency": True,
    "adaptive_embedding_dim": 10,
    "adaptive_initialization": "random_seeded",
    "receptive_field": 13,
    "temporal_alignment": "last_valid_causal_forecast_anchor",
}


def resolve_graph_wavenet_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    return {
        **resolve_e3_c_common(
            "graph_wavenet", protocol, run_mode=run_mode
        ),
        **GRAPH_WAVENET_CONFIG,
        "wrapper_path": (
            "custom_models/src/benchmark_v2/models/graph_models/"
            "graph_wavenet.py"
        ),
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e3_c.py",
        "primary_architecture_source": (
            "Wu et al., Graph WaveNet for Deep Spatial-Temporal Graph "
            "Modeling, IJCAI 2019"
        ),
        "official_repository_identity": (
            "https://github.com/nnzhan/Graph-WaveNet"
        ),
        "official_repository_license": "MIT",
        "config_resolution_reason": (
            "Prompt-frozen author-standard 4x2 gated dilated architecture, "
            "dual frozen random-walk supports, random low-rank adaptive "
            "adjacency, and explicit last valid causal anchor for T=144."
        ),
    }
