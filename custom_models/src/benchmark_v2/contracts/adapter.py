from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .batch import BenchmarkBatch
from .output import BenchmarkOutput


class BenchmarkAdapter(ABC):
    """The only model boundary used by benchmark_v2."""

    @abstractmethod
    def build_model(self, config: dict[str, Any], protocol: Any):
        raise NotImplementedError

    @abstractmethod
    def prepare_model_inputs(self, batch: BenchmarkBatch, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def forward_model(self, model, model_inputs: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def normalize_output(self, raw_output: Any, batch: BenchmarkBatch, **kwargs: Any) -> BenchmarkOutput:
        raise NotImplementedError

    def __call__(self, model, batch: BenchmarkBatch, **kwargs: Any) -> BenchmarkOutput:
        batch.validate(expected_horizon=int(kwargs.get("expected_horizon", 10)), expected_features=int(kwargs.get("expected_features", 16)))
        inputs = self.prepare_model_inputs(batch, **kwargs)
        raw = self.forward_model(model, inputs, **kwargs)
        output = self.normalize_output(raw, batch, **kwargs)
        output.validate(batch, expected_horizon=int(kwargs.get("expected_horizon", 10)))
        return output

    def summarize_effective_config(self, config: dict[str, Any]) -> dict[str, Any]:
        return dict(config)

