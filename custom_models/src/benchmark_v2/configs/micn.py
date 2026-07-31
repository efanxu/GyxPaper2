from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


MICN_CONFIG = {
    "label_len": 144,
    "c_out": 16,
    "d_model": 32,
    "n_heads": 8,
    "d_layers": 1,
    "dropout": 0.1,
    "embed": "timeF",
    "freq": "h",
    "conv_kernel": [12, 16],
    "decomp_kernel": [13, 17],
    "isometric_kernel": [13, 10],
}


def resolve_micn_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    config = {
        **resolve_common("micn", protocol, run_mode=run_mode),
        **MICN_CONFIG,
        "raw_output_channels": 16,
        "decoder_length": 154,
        "decoder_history_length": 144,
        "decoder_future_length": 10,
        "decoder_future_source": "all_zero",
        "time_mark_policy": "all_zero_length_seq_len_plus_pred_len",
        "wrapper_path": "custom_models/src/benchmark_v2/models/micn.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e2_b.py",
        "model_specific_parameters": dict(MICN_CONFIG),
        "config_resolution_reason": (
            "local official Weather long-term forecasting width; full-history "
            "decoder mark length is required by MICN source semantics"
        ),
        "oom_driven_capacity_reduction": False,
    }
    expected_decomp = [
        value + 1 if value % 2 == 0 else value
        for value in config["conv_kernel"]
    ]
    expected_iso = [
        (config["seq_len"] + config["pred_len"] + value) // value
        if value % 2 == 0
        else (config["seq_len"] + config["pred_len"] + value - 1) // value
        for value in config["conv_kernel"]
    ]
    if expected_decomp != config["decomp_kernel"]:
        raise ValueError("MICN frozen decomposition kernels disagree with source.")
    if expected_iso != config["isometric_kernel"]:
        raise ValueError("MICN frozen isometric kernels disagree with source.")
    return config
