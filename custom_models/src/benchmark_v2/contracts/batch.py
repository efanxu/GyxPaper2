from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import ContractError


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in value.shape)


@dataclass
class BenchmarkBatch:
    x: Any
    target: Any
    mask: Any
    sample_ids: Any
    window_end_indices: Any
    node_ids: Any
    x_mark: Any = None
    y_mark: Any = None
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)
    target_raw_or_inverse_transform: Any = None

    def validate(self, *, expected_horizon: int = 10, expected_features: int = 16, expected_nodes: int | None = None) -> None:
        x_shape = _shape(self.x)
        target_shape = _shape(self.target)
        mask_shape = _shape(self.mask)
        if len(x_shape) != 4 or x_shape[-1] != expected_features:
            raise ContractError(f"x must be (B,T,N,{expected_features}), got {x_shape}")
        expected = (x_shape[0], x_shape[2], expected_horizon)
        if target_shape != expected or mask_shape != expected:
            raise ContractError(f"target/mask must be {expected}, got target={target_shape}, mask={mask_shape}")
        if expected_nodes is not None and x_shape[2] != expected_nodes:
            raise ContractError(f"Expected N={expected_nodes}, got {x_shape[2]}")
        if getattr(self.mask, "dtype", None) not in (None, bool):
            try:
                if str(self.mask.dtype) not in {"torch.bool", "torch.float32", "torch.float64", "float32", "float64", "bool"}:
                    raise ContractError("mask must be bool or an explicit 0/1 numeric tensor")
            except AttributeError:
                pass
        if self.x_mark is not None and _shape(self.x_mark)[:3] != x_shape[:3]:
            raise ContractError("x_mark must preserve B,T,N ordering")
        if self.y_mark is not None and _shape(self.y_mark)[:3] != target_shape[:3]:
            raise ContractError("y_mark must preserve B,N,H ordering")
        if self.target_raw_or_inverse_transform is not None and _shape(self.target_raw_or_inverse_transform) != target_shape:
            raise ContractError("target_raw_or_inverse_transform must align exactly with target and mask")
        if not isinstance(self.metadata, dict):
            raise ContractError("metadata must be a dictionary")
        if self.metadata.get("contains_future_target"):
            raise ContractError("Batch metadata cannot contain future target information")
        mask_values = self.mask.detach().cpu().numpy() if hasattr(self.mask, "detach") else self.mask
        import numpy as np
        if not np.isin(np.asarray(mask_values), [0, 1, False, True]).all():
            raise ContractError("mask must contain only bool/0/1 values")

    @property
    def batch_size(self) -> int:
        return int(self.x.shape[0])

    @property
    def node_count(self) -> int:
        return int(self.x.shape[2])

    @property
    def horizon(self) -> int:
        return int(self.target.shape[-1])

