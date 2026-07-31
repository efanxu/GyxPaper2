from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import ContractError


@dataclass
class BenchmarkOutput:
    prediction: Any
    aux: dict[str, Any] = field(default_factory=dict)
    raw_output_shape: tuple[int, ...] | None = None
    semantic_trace: dict[str, Any] = field(default_factory=dict)

    def validate(self, batch, *, expected_horizon: int = 10) -> None:
        expected = (batch.batch_size, batch.node_count, expected_horizon)
        got = tuple(int(x) for x in self.prediction.shape)
        if got != expected:
            raise ContractError(f"prediction must be {expected}, got {got}")
        import numpy as np
        values = self.prediction.detach().cpu().numpy() if hasattr(self.prediction, "detach") else np.asarray(self.prediction)
        if not np.isfinite(values).all():
            raise ContractError("prediction contains NaN/Inf")
        if self.semantic_trace.get("target_column") != "Patv_raw":
            raise ContractError("prediction semantic_trace must explicitly identify Patv_raw")
        if not self.semantic_trace.get("output_semantics"):
            raise ContractError("prediction semantic_trace must declare output semantics")

