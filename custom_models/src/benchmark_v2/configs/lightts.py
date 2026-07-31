from __future__ import annotations

from typing import Any, Mapping

from .tslib_common import resolve_common


LIGHTTS_CONFIG = {
    "chunk_size": 8,
    "d_model": 512,
    "dropout": 0.1,
    "c_out": 16,
}


def resolve_lightts_config(
    protocol: Mapping[str, Any], *, run_mode: str = "formal"
) -> dict[str, Any]:
    config = {
        **resolve_common("lightts", protocol, run_mode=run_mode),
        **LIGHTTS_CONFIG,
        "raw_output_channels": 16,
        "wrapper_path": "custom_models/src/benchmark_v2/models/lightts.py",
        "adapter_path": "custom_models/src/benchmark_v2/adapters/lightts.py",
        "model_specific_parameters": {
            "chunk_size": 8,
            "num_chunks": 18,
            "d_model": 512,
            "dropout": 0.1,
        },
        "config_resolution_reason": (
            "largest divisor of seq_len=144 not exceeding pred_len=10; "
            "avoids upstream padding while preserving run.py d_model/dropout"
        ),
        "uses_zero_placeholders": False,
    }
    if config["seq_len"] % config["chunk_size"]:
        raise ValueError("LightTS E1-B chunk_size must divide seq_len exactly.")
    config["num_chunks"] = config["seq_len"] // config["chunk_size"]
    return config
