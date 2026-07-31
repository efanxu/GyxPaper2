from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


MULTIPATCHFORMER_CONFIG = {
    "label_len": 48,
    "c_out": 16,
    "d_model": 256,
    "d_ff": 512,
    "e_layers": 1,
    "n_heads": 8,
    "dropout": 0.1,
    "patch_lengths": [8, 16, 24, 32],
    "patch_strides": [8, 8, 7, 7],
    "patch_paddings": [0, 8, 0, 8],
    "patch_counts": [18, 18, 18, 18],
}


def _patch_count(length: int, patch: int, stride: int, padding: int) -> int:
    return (length + padding - patch) // stride + 1


def resolve_multipatchformer_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    config = {
        **resolve_common("multipatchformer", protocol, run_mode=run_mode),
        **MULTIPATCHFORMER_CONFIG,
        "raw_output_channels": 16,
        "time_mark_policy": "none",
        "branch_count": 4,
        "compatibility_adjustments": {
            "branch_3_padding": "7_to_0",
            "branch_4_stride": "6_to_7",
            "branch_4_padding": "6_to_8",
            "semi_autoregressive_head": (
                "layers_5_to_8_use_exact_cumulative_chunk_widths_for_pred_len_10"
            ),
            "silent_crop": False,
            "disabled_branches": [],
        },
        "wrapper_path": (
            "custom_models/src/benchmark_v2/models/multipatchformer.py"
        ),
        "adapter_path": (
            "custom_models/src/benchmark_v2/adapters/e2_b.py"
        ),
        "model_specific_parameters": dict(MULTIPATCHFORMER_CONFIG),
        "config_resolution_reason": (
            "official Weather first-horizon capacity plus minimum explicit "
            "source-compatible patch geometry for fixed seq_len=144"
        ),
        "oom_driven_capacity_reduction": False,
    }
    counts = [
        _patch_count(144, patch, stride, padding)
        for patch, stride, padding in zip(
            config["patch_lengths"],
            config["patch_strides"],
            config["patch_paddings"],
        )
    ]
    if counts != config["patch_counts"] or len(set(counts)) != 1:
        raise ValueError("MultiPatchFormer frozen branch geometry is inconsistent.")
    if len(set(config["patch_lengths"])) != 4:
        raise ValueError("MultiPatchFormer must retain four distinct patch scales.")
    return config
