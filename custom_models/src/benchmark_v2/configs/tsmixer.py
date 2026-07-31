from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


# Only fields read by Time-Series-Library/models/TSMixer.py.
TSMIXER_CONFIG = {
    "e_layers": 2,
    "d_model": 32,
    "dropout": 0.1,
}


def resolve_tsmixer_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    values = dict(TSMIXER_CONFIG)
    return {
        **resolve_common("tsmixer", protocol, run_mode=run_mode),
        **values,
        "raw_output_channels": 16,
        "normalization": "none_in_local_upstream_source",
        "time_mark_policy": "none",
        "wrapper_path": "custom_models/src/benchmark_v2/models/tsmixer.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e2_c.py",
        "model_specific_parameters": dict(values),
        "config_resolution_reason": (
            "local TSMixer source fields plus local official Weather long-term "
            "forecasting e_layers=2,d_model=32; unused unified CLI fields excluded"
        ),
        "validation_search_performed": False,
        "test_result_used": False,
        "smoke_result_used_for_selection": False,
        "oom_driven_capacity_reduction": False,
    }
