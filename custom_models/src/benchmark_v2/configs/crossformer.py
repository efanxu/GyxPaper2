from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


CROSSFORMER_CONFIG = {
    "d_model": 32,
    "d_ff": 32,
    "e_layers": 2,
    "n_heads": 8,
    "factor": 3,
    "dropout": 0.1,
}


def resolve_crossformer_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    values = dict(CROSSFORMER_CONFIG)
    return {
        **resolve_common("crossformer", protocol, run_mode=run_mode),
        **values,
        "c_out": 16,
        "raw_output_channels": 16,
        "seg_len": 12,
        "win_size": 2,
        "pad_in_len": 144,
        "pad_out_len": 12,
        "in_seg_num": 12,
        "encoder_segment_counts": [12, 6],
        "decoder_segment_count": 1,
        "time_mark_policy": "none",
        "wrapper_path": "custom_models/src/benchmark_v2/models/crossformer.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e2_d.py",
        "model_specific_parameters": dict(values),
        "config_resolution_reason": (
            "local Crossformer Weather script fixes e_layers=2,factor=3,"
            "d_model=32,d_ff=32; n_heads/dropout use local run.py defaults"
        ),
        "smoke_result_used_for_selection": False,
        "oom_driven_capacity_reduction": False,
    }
