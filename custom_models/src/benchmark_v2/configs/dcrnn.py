from __future__ import annotations

from typing import Any, Mapping

from .graph_common import resolve_graph_common


DCRNN_CONFIG = {
    "encoder_input_dim": 16,
    "decoder_input_dim": 1,
    "output_dim": 1,
    "hidden_dim": 64,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "max_diffusion_step": 2,
    "filter_type": "dual_random_walk",
    "dropout": 0.0,
    "use_curriculum_learning": False,
    "teacher_forcing": False,
    "scheduled_sampling": False,
    "decoder_go_token": "all_zero_normalized_scalar",
    "graph_support_names": ["P_forward", "P_reverse"],
}


def resolve_dcrnn_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    return {
        **resolve_graph_common("dcrnn", protocol, run_mode=run_mode),
        **DCRNN_CONFIG,
        "wrapper_path": (
            "custom_models/src/benchmark_v2/models/graph_models/dcrnn.py"
        ),
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e3_b.py",
        "primary_architecture_source": (
            "Li, Yu, Shahabi, and Liu, Diffusion Convolutional Recurrent "
            "Neural Network: Data-Driven Traffic Forecasting, ICLR 2018"
        ),
        "official_repository_identity": "https://github.com/liyaguang/DCRNN",
        "config_resolution_reason": (
            "Prompt-frozen dual-random-walk K=2 DCGRU encoder-decoder. "
            "Scheduled sampling is disabled solely to preserve the unified "
            "target-never-enters-forward benchmark contract."
        ),
    }
