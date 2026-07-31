from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


TRANSFORMER_CONFIG = {
    "label_len": 48,
    "dec_in": 16,
    "c_out": 1,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 2048,
    "factor": 1,
    "dropout": 0.1,
    "activation": "gelu",
    "embed": "timeF",
    "freq": "h",
}


def resolve_transformer_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    return {
        **resolve_common("transformer", protocol, run_mode=run_mode),
        **TRANSFORMER_CONFIG,
        "raw_output_channels": 1,
        "decoder_history_source": "observed_x_last_label_len",
        "decoder_future_source": "all_zero",
        "time_mark_policy": "all_zero_placeholder_required_by_DataEmbedding",
        "wrapper_path": "custom_models/src/benchmark_v2/models/transformer.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e2_a.py",
        "model_specific_parameters": dict(TRANSFORMER_CONFIG),
        "config_resolution_reason": "frozen Time-Series-Library run.py architecture defaults",
    }
