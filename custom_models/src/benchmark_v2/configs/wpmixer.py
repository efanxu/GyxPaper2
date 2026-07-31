from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


WPMIXER_CONFIG = {
    "label_len": 0,
    "c_out": 16,
    "d_model": 256,
    "dropout": 0.1,
    "patch_len": 16,
    "batch_size": 32,
    "use_amp": False,
    "wavelet": "db2",
    "level": 1,
    "tfactor": 5,
    "dfactor": 5,
    "stride": 8,
    "no_decomposition": False,
}


def resolve_wpmixer_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    return {
        **resolve_common("wpmixer", protocol, run_mode=run_mode),
        **WPMIXER_CONFIG,
        "raw_output_channels": 16,
        "time_mark_policy": "none",
        "amp_policy": (
            "benchmark_v2 Trainer owns AMP; DWT/IDWT fixed FP32 while "
            "resolution mixers remain under Trainer AMP"
        ),
        "resolution_branch_count": 2,
        "wrapper_path": "custom_models/src/benchmark_v2/models/wpmixer.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e2_b.py",
        "model_specific_parameters": dict(WPMIXER_CONFIG),
        "config_resolution_reason": (
            "source constructor defaults for wavelet/mixer factors/stride; "
            "official patch_len=16 and Weather first-horizon width"
        ),
        "oom_driven_capacity_reduction": False,
    }
