from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


ITRANSFORMER_CONFIG = {
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_ff": 2048,
    "factor": 1,
    "dropout": 0.1,
    "activation": "gelu",
    "embed": "timeF",
    "freq": "h",
}


def resolve_itransformer_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    return {
        **resolve_common("itransformer", protocol, run_mode=run_mode),
        **ITRANSFORMER_CONFIG,
        "raw_output_channels": 16,
        "variable_token_count": 16,
        "variable_token_semantics": "features_within_one_turbine",
        "time_mark_policy": "none",
        "wrapper_path": "custom_models/src/benchmark_v2/models/itransformer.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e2_a.py",
        "model_specific_parameters": dict(ITRANSFORMER_CONFIG),
        "config_resolution_reason": "frozen Time-Series-Library run.py architecture defaults",
    }
