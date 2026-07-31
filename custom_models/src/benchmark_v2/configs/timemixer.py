from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


TIMEMIXER_CONFIG = {
    "label_len": 0,
    "enc_in": 16,
    "dec_in": 16,
    "c_out": 16,
    "d_model": 16,
    "d_ff": 32,
    "e_layers": 3,
    "dropout": 0.1,
    "down_sampling_layers": 3,
    "down_sampling_window": 2,
    "down_sampling_method": "avg",
    "decomp_method": "moving_avg",
    "moving_avg": 25,
    "channel_independence": 1,
    "use_norm": 1,
    "embed": "timeF",
    "freq": "h",
}


def resolve_timemixer_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    values = dict(TIMEMIXER_CONFIG)
    lengths = [
        int(protocol["lookback"]) // (values["down_sampling_window"] ** index)
        for index in range(values["down_sampling_layers"] + 1)
    ]
    if lengths != [144, 72, 36, 18]:
        raise ValueError(f"TimeMixer frozen scale lengths changed: {lengths}")
    return {
        **resolve_common("timemixer", protocol, run_mode=run_mode),
        **values,
        "raw_output_channels": 16,
        "scale_lengths": lengths,
        "time_mark_policy": "none",
        "wrapper_path": "custom_models/src/benchmark_v2/models/timemixer.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e2_c.py",
        "model_specific_parameters": dict(values),
        "config_resolution_reason": (
            "local TimeMixer source shape requirements and local official "
            "Weather long-term forecasting script; benchmark dimensions override "
            "only dataset-dependent seq_len/pred_len/enc_in"
        ),
        "validation_search_performed": False,
        "test_result_used": False,
        "smoke_result_used_for_selection": False,
        "oom_driven_capacity_reduction": False,
    }
