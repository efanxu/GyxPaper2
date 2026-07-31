from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


MSGNET_CONFIG = {
    "label_len": 48,
    "c_out": 16,
    "d_model": 512,
    "d_ff": 2048,
    "e_layers": 2,
    "n_heads": 8,
    "top_k": 5,
    "dropout": 0.1,
    "embed": "timeF",
    "freq": "h",
    "conv_channel": 32,
    "skip_channel": 32,
    "gcn_depth": 2,
    "propalpha": 0.3,
    "node_dim": 10,
    "individual": False,
}


def resolve_msgnet_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    values = dict(MSGNET_CONFIG)
    if values["d_model"] < values["c_out"]:
        raise ValueError("MSGNet GraphBlock requires d_model >= c_out.")
    if not 0 < values["top_k"] <= int(protocol["lookback"]) // 2:
        raise ValueError("MSGNet top_k exceeds available non-DC FFT bins.")
    return {
        **resolve_common("msgnet", protocol, run_mode=run_mode),
        **values,
        "raw_output_channels": 16,
        "adaptive_adjacency_shape": [16, 16],
        "fft_batch_isolation": "one shared model call per flattened prediction item",
        "amp_fft_policy": (
            "FFT and frequency statistics FP32; learned branches retain AMP"
        ),
        "time_mark_policy": "none",
        "wrapper_path": "custom_models/src/benchmark_v2/models/msgnet.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e2_d.py",
        "model_specific_parameters": dict(values),
        "config_resolution_reason": (
            "no local MSGNet task script exists; fields use local run.py "
            "defaults plus source-required graph fields, frozen before smoke"
        ),
        "smoke_result_used_for_selection": False,
        "oom_driven_capacity_reduction": False,
    }
