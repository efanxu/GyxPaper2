from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


PATCHTST_CONFIG = {
    "patch_len": 16,
    "stride": 8,
    "padding_patch": "end",
    "padding_len": 8,
    "patch_num": 18,
    "individual": False,
    "decomposition": False,
    "kernel_size": 25,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_ff": 2048,
    "factor": 1,
    "dropout": 0.1,
    "head_dropout": 0.1,
    "activation": "gelu",
}


def resolve_patchtst_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    config = {
        **resolve_common("patchtst", protocol, run_mode=run_mode),
        **PATCHTST_CONFIG,
        "raw_output_channels": 16,
        "original_length": 144,
        "padded_length": 152,
        "final_patch_shape": "(B*N*16,18,512)",
        "wrapper_path": "custom_models/src/benchmark_v2/models/patchtst.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/e2_a.py",
        "model_specific_parameters": dict(PATCHTST_CONFIG),
        "config_resolution_reason": "upstream constructor defaults patch_len=16,stride=8 with explicit end padding",
    }
    expected = int((config["seq_len"] - config["patch_len"]) / config["stride"] + 2)
    if expected != config["patch_num"]:
        raise ValueError("PatchTST frozen patch_num does not match upstream formula.")
    return config
