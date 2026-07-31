from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


TIMESNET_CONFIG = {
    "label_len": 48,
    "c_out": 16,
    "d_model": 32,
    "d_ff": 32,
    "e_layers": 2,
    "top_k": 5,
    "num_kernels": 6,
    "dropout": 0.1,
    "embed": "timeF",
    "freq": "h",
}


def resolve_timesnet_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    frequency_bins = int(protocol["lookback"]) // 2 + 1
    if TIMESNET_CONFIG["top_k"] > frequency_bins - 1:
        raise ValueError("TimesNet top_k exceeds non-DC history frequencies.")
    return {
        **resolve_common("timesnet", protocol, run_mode=run_mode),
        **TIMESNET_CONFIG,
        "raw_output_channels": 16,
        "frequency_bins": frequency_bins,
        "period_source": "dynamic_fft_of_history_only",
        "time_mark_policy": "none",
        "amp_fft_policy": "FFT FP32; convolution and backward under AMP",
        "wrapper_path": "custom_models/src/benchmark_v2/models/timesnet.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e2_b.py",
        "model_specific_parameters": dict(TIMESNET_CONFIG),
        "config_resolution_reason": (
            "local official Weather long-term forecasting script; fixed before smoke"
        ),
        "oom_driven_capacity_reduction": False,
    }
