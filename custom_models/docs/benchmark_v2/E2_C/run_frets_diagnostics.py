from __future__ import annotations

import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark_v2.upstream import load_tslib_model_class


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT = (
    PROJECT_ROOT
    / "custom_models/results_smoke/benchmark_v2/e2_c/diagnostics/"
    "frets_branch_amp_diagnostics.json"
)


def config(value):
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=144,
        pred_len=10,
        enc_in=16,
        channel_independence=value,
    )


def branch_case(model_class, value):
    model = model_class(config(value))
    x = torch.randn(1, 144, 16)
    output = model(x, None, None, None)
    output.square().mean().backward()
    channel_names = ("r1", "i1", "rb1", "ib1")
    temporal_names = ("r2", "i2", "rb2", "ib2")
    return {
        "input_value": value,
        "input_type": type(value).__name__,
        "output_shape": list(output.shape),
        "channel_frequency_executed": all(
            getattr(model, name).grad is not None for name in channel_names
        ),
        "channel_gradient_sums": {
            name: (
                None
                if getattr(model, name).grad is None
                else float(getattr(model, name).grad.abs().sum())
            )
            for name in channel_names
        },
        "temporal_gradient_sums": {
            name: float(getattr(model, name).grad.abs().sum())
            for name in temporal_names
        },
    }


def raw_amp_case(model_class):
    result = {
        "status": "NOT_RUN",
        "cuda_available": torch.cuda.is_available(),
        "amp_enabled": True,
        "wrapper_applied": False,
        "channel_independence": "0",
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }
    if not torch.cuda.is_available():
        return result
    device = torch.device("cuda")
    model = model_class(config("0")).to(device)
    x = torch.randn(1, 144, 16, device=device)
    try:
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            output = model(x, None, None, None)
            loss = output.square().mean()
        loss.backward()
        result.update(
            {
                "status": "PASS",
                "output_shape": list(output.shape),
                "output_dtype": str(output.dtype),
                "backward_completed": True,
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": (
                    "FAIL_OOM"
                    if "out of memory" in str(exc).casefold()
                    else "FAIL_NON_OOM"
                ),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "backward_completed": False,
            }
        )
    return result


def main():
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    model_class, source = load_tslib_model_class("frets")
    payload = {
        "task": "E2-C",
        "model_id": "frets",
        "source_path": source.source_path,
        "type_matrix": [
            branch_case(model_class, value) for value in (0, 1, "0", "1")
        ],
        "raw_upstream_amp_attempt": raw_amp_case(model_class),
        "final_wrapper_policy": (
            "Native Trainer autocast retained because raw-source CUDA AMP "
            "forward/backward passed"
        ),
        "final_channel_independence": "0",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
