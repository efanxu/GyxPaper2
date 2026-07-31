from __future__ import annotations

from ..contracts import BenchmarkAdapter, BenchmarkBatch, BenchmarkOutput


def adapter_forward(adapter: BenchmarkAdapter, model, batch: BenchmarkBatch, **kwargs) -> BenchmarkOutput:
    return adapter(model, batch, **kwargs)

