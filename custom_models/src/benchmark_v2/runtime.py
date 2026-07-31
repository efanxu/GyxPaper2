from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAL_ROOT = PROJECT_ROOT / "custom_models" / "results" / "benchmark_v2"
SMOKE_ROOT = PROJECT_ROOT / "custom_models" / "results_smoke" / "benchmark_v2"


def environment_snapshot() -> dict[str, str | bool | None]:
    result = {"python": sys.executable, "python_version": sys.version, "platform": platform.platform(), "device": "cpu", "cuda_available": False, "torch_version": None}
    try:
        import torch
        result.update({"torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "device": "cuda" if torch.cuda.is_available() else "cpu"})
    except ImportError:
        pass
    return result


class ProviderBatchIterable:
    """Lazy adapter over SDWPFDataProvider.batches without duplicating window logic."""

    def __init__(self, provider, split: str, batch_size: int, limit_batches: int | None = None):
        self.provider = provider
        self.split = str(split)
        self.batch_size = int(batch_size)
        self.limit_batches = None if limit_batches is None else int(limit_batches)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.limit_batches is not None and self.limit_batches <= 0:
            raise ValueError("limit_batches must be positive when provided")

    def __iter__(self) -> Iterator:
        original = list(self.provider.starts[self.split])
        try:
            for batch_index, offset in enumerate(range(0, len(original), self.batch_size)):
                if self.limit_batches is not None and batch_index >= self.limit_batches:
                    break
                self.provider.starts[self.split] = original[offset : offset + self.batch_size]
                batches = self.provider.batches(self.split, self.batch_size)
                if len(batches) != 1:
                    raise RuntimeError(
                        f"Expected one lazy provider batch, got {len(batches)}"
                    )
                yield batches[0]
        finally:
            self.provider.starts[self.split] = original
