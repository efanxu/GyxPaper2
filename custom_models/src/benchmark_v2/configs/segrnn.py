from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


SEGRNN_CONFIG = {
    "seg_len": 2,
    "d_model": 512,
    "dropout": 0.1,
    "c_out": 16,
}


def resolve_segrnn_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    config = {
        **resolve_common("segrnn", protocol, run_mode=run_mode),
        **SEGRNN_CONFIG,
        "raw_output_channels": 16,
        "wrapper_path": "custom_models/src/benchmark_v2/models/segrnn.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/segrnn.py",
        "model_specific_parameters": {
            "seg_len": 2,
            "seg_num_x": 72,
            "seg_num_y": 5,
            "d_model": 512,
            "dropout": 0.1,
        },
        "config_resolution_reason": "protocol_divisibility_constraint",
        "uses_zero_placeholders": False,
        "known_original_seg_len": 96,
        "known_original_smoke_status": "FAIL",
        "known_original_smoke_error": "dimension mismatch",
        "effective_seg_len": 2,
    }
    seg_len = int(config["seg_len"])
    if config["seq_len"] % seg_len or config["pred_len"] % seg_len:
        raise ValueError("SegRNN seg_len must divide seq_len and pred_len exactly.")
    config["seg_num_x"] = config["seq_len"] // seg_len
    config["seg_num_y"] = config["pred_len"] // seg_len
    return config
