from __future__ import annotations

from typing import Any, Mapping

from .e3_c_common import resolve_e3_c_common


AGCRN_CONFIG = {
    "num_nodes": 134,
    "input_dim": 16,
    "output_dim": 1,
    "horizon": 10,
    "embed_dim": 10,
    "rnn_units": 64,
    "num_layers": 2,
    "cheb_order": 2,
    "predefined_graph": "disabled",
    "adaptive_basis": ["T0_identity", "T1_DAGG"],
    "teacher_forcing": False,
}


def resolve_agcrn_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    return {
        **resolve_e3_c_common("agcrn", protocol, run_mode=run_mode),
        **AGCRN_CONFIG,
        "wrapper_path": (
            "custom_models/src/benchmark_v2/models/graph_models/agcrn.py"
        ),
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e3_c.py",
        "primary_architecture_source": (
            "Bai et al., Adaptive Graph Convolutional Recurrent Network for "
            "Traffic Forecasting, NeurIPS 2020"
        ),
        "official_repository_identity": "https://github.com/LeiBAI/AGCRN",
        "official_repository_license": "MIT",
        "config_resolution_reason": (
            "Prompt-frozen DAGG/NAPL embed_dim=10, two order bases, two "
            "64-unit recurrent layers, and a direct target-free horizon head."
        ),
    }
