from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


DLINEAR_CONFIG = {
    "moving_avg": 25,
    "individual": False,
    "c_out": 16,
}


def resolve_dlinear_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    return {
        **resolve_common("dlinear", protocol, run_mode=run_mode),
        **DLINEAR_CONFIG,
        "raw_output_channels": 16,
        "wrapper_path": "custom_models/src/benchmark_v2/models/dlinear.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/dlinear.py",
        "model_specific_parameters": {
            "moving_avg": 25,
            "individual": False,
        },
        "config_resolution_reason": "upstream_run_py_defaults_and_frozen_protocol",
        "uses_zero_placeholders": False,
    }
