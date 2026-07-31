from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


TIDE_CONFIG = {
    "label_len": 48,
    "c_out": 1,
    "d_model": 512,
    "d_ff": 2048,
    "e_layers": 2,
    "d_layers": 1,
    "dropout": 0.1,
    "freq": "h",
    "bias": True,
    "feature_encode_dim": 2,
}


def resolve_tide_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    return {
        **resolve_common("tide", protocol, run_mode=run_mode),
        **TIDE_CONFIG,
        "decode_dim": 1,
        "temporal_decoder_hidden": 2048,
        "raw_output_channels": 16,
        "wrapper_path": "custom_models/src/benchmark_v2/models/tide.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/tide.py",
        "model_specific_parameters": dict(TIDE_CONFIG),
        "config_resolution_reason": (
            "run.py architecture defaults; c_out=1 is the minimal M-to-S "
            "decoder width, while upstream still emits one forecast per input channel"
        ),
        "uses_zero_placeholders": True,
        "zero_placeholder_covariates": True,
        "effective_future_covariates": "none",
    }
