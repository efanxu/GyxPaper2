from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


TIMEFILTER_CONFIG = {
    "c_out": 16,
    "d_model": 512,
    "d_ff": 2048,
    "e_layers": 2,
    "n_heads": 8,
    "dropout": 0.1,
    "patch_len": 16,
    "alpha": 0.1,
    "top_p": 0.5,
    "pos": 1,
}


def resolve_timefilter_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    values = dict(TIMEFILTER_CONFIG)
    seq_len = int(protocol["lookback"])
    features = int(protocol["feature_count"])
    if seq_len % values["patch_len"] != 0:
        raise ValueError("TimeFilter forbids silent patch truncation.")
    patches_per_variable = seq_len // values["patch_len"]
    token_count = features * patches_per_variable
    if values["d_model"] % values["n_heads"] != 0:
        raise ValueError("TimeFilter d_model must be divisible by n_heads.")
    return {
        **resolve_common("timefilter", protocol, run_mode=run_mode),
        **values,
        "raw_output_channels": 16,
        "patches_per_variable": patches_per_variable,
        "token_count": token_count,
        "mask_regions": ["S", "T", "ST"],
        "moe_auxiliary_loss_used_for_training": False,
        "time_mark_policy": "none",
        "wrapper_path": "custom_models/src/benchmark_v2/models/timefilter.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e2_d.py",
        "model_specific_parameters": dict(values),
        "config_resolution_reason": (
            "no local TimeFilter task script exists; local run.py defaults "
            "and source patch/filter fields frozen before smoke"
        ),
        "smoke_result_used_for_selection": False,
        "oom_driven_capacity_reduction": False,
    }
