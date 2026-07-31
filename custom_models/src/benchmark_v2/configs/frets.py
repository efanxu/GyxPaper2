from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


FRETS_CONFIG = {
    "channel_independence": "0",
    "embed_size": 128,
    "hidden_size": 256,
    "sparsity_threshold": 0.01,
    "scale": 0.02,
}


def resolve_frets_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    values = dict(FRETS_CONFIG)
    if values["channel_independence"] != "0":
        raise ValueError("FreTS full multivariate identity requires string '0'.")
    constructor_values = {"channel_independence": values["channel_independence"]}
    return {
        **resolve_common("frets", protocol, run_mode=run_mode),
        **constructor_values,
        "raw_output_channels": 16,
        "embed_size": values["embed_size"],
        "hidden_size": values["hidden_size"],
        "sparsity_threshold": values["sparsity_threshold"],
        "scale": values["scale"],
        "channel_frequency_branch": True,
        "temporal_frequency_branch": True,
        "time_mark_policy": "none",
        "amp_forward_policy": (
            "native Trainer autocast; isolated raw-source CUDA AMP "
            "forward/backward passed without a precision-boundary repair"
        ),
        "wrapper_path": "custom_models/src/benchmark_v2/models/frets.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e2_c.py",
        "model_specific_parameters": dict(values),
        "config_resolution_reason": (
            "local FreTS source hard-coded capacity and explicit correction of "
            "source string comparison channel_independence == '0'"
        ),
        "validation_search_performed": False,
        "test_result_used": False,
        "smoke_result_used_for_selection": False,
        "oom_driven_capacity_reduction": False,
    }
