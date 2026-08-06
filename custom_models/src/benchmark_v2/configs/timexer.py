from __future__ import annotations

from typing import Any, Mapping

from ..errors import ContractError
from .tslib_common import resolve_common


TIMEXER_CONFIG = {
    "features": "MS",
    "use_norm": True,
    "patch_len": 16,
    "patch_num": 9,
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


def resolve_timexer_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    features = tuple(protocol["ordered_input_features"])
    power = str(protocol["input_power_column"])
    if not features or features[-1] != power:
        raise ContractError(
            "TimeXer requires Patv_clean_for_input as the final endogenous channel."
        )
    return {
        **resolve_common("timexer", protocol, run_mode=run_mode),
        **TIMEXER_CONFIG,
        "raw_output_channels": 1,
        "endogenous_feature": power,
        "endogenous_feature_index": len(features) - 1,
        "exogenous_features": list(features[:-1]),
        "exogenous_feature_count": len(features) - 1,
        "future_exogenous_tensor": None,
        "time_mark_policy": "none",
        "wrapper_path": "custom_models/src/benchmark_v2/models/timexer.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e2_a.py",
        "model_specific_parameters": dict(TIMEXER_CONFIG),
        "config_resolution_reason": "upstream TimeXer MS path; final channel is frozen endogenous power",
    }
