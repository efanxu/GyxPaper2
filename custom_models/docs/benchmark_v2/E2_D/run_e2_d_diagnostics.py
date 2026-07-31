from __future__ import annotations

import json
from pathlib import Path

import torch

from benchmark_v2.contracts import BenchmarkBatch
from benchmark_v2.model_runtime import build_model_runtime
from benchmark_v2.protocol import load_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT = (
    PROJECT_ROOT
    / "custom_models/results_smoke/benchmark_v2/e2_d/diagnostics/"
    "e2_d_model_diagnostics.json"
)


def main() -> None:
    torch.set_num_threads(1)
    protocol = load_protocol()
    generator = torch.Generator().manual_seed(2026)
    batch = BenchmarkBatch(
        x=torch.randn(1, 144, 1, 16, generator=generator),
        target=torch.randn(1, 1, 10, generator=generator),
        mask=torch.ones(1, 1, 10, dtype=torch.bool),
        sample_ids=[0],
        window_end_indices=[0],
        node_ids=[0],
        metadata={"contains_future_target": False},
    )
    results = {}
    for model_id in ("crossformer", "msgnet", "timefilter"):
        runtime = build_model_runtime(model_id, protocol, run_mode="smoke")
        runtime.model.eval()
        output = runtime.adapter(runtime.model, batch)
        output.prediction.square().mean().backward()
        nonzero = [
            name
            for name, parameter in runtime.model.named_parameters()
            if parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and float(parameter.grad.abs().sum()) > 0
        ]
        results[model_id] = {
            "prediction_shape": list(output.prediction.shape),
            "raw_output_shape": list(output.raw_output_shape),
            "parameter_count": runtime.parameter_count,
            "aux": output.aux,
            "finite_nonzero_gradient_parameters": nonzero,
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "task": "E2-D",
                "run_mode": "smoke_diagnostic",
                "seed": 2026,
                "formal_training": False,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT.relative_to(PROJECT_ROOT).as_posix())


if __name__ == "__main__":
    main()
