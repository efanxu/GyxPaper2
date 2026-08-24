from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..errors import ContractError


@dataclass(frozen=True)
class LossInputBundle:
    """Model-independent loss boundary for benchmark_v2."""

    prediction: Any
    target: Any
    mask: Any
    site_id: tuple[Any, ...]
    horizon_index: tuple[int, ...]
    normalization_identity: str
    protocol_identity: str
    split: str

    def validate(self) -> None:
        shapes = tuple(tuple(int(item) for item in value.shape) for value in (self.prediction, self.target, self.mask))
        if len(set(shapes)) != 1 or len(shapes[0]) != 3:
            raise ContractError(f"LossInputBundle tensors must align as (B,N,H), got {shapes}")
        if len(self.site_id) != shapes[0][1]:
            raise ContractError("LossInputBundle site_id must match the N axis.")
        if self.horizon_index != tuple(range(1, shapes[0][2] + 1)):
            raise ContractError("LossInputBundle horizon_index must be the public one-based horizon axis.")
        if self.split not in {"train", "val", "test", "synthetic"}:
            raise ContractError(f"Unsupported loss split identity: {self.split}")


def make_loss_input_bundle(prediction: Any, batch: Any, protocol: Mapping[str, Any]) -> LossInputBundle:
    node_count = int(prediction.shape[1])
    node_ids = tuple(batch.node_ids) if batch.node_ids is not None else tuple(range(node_count))
    bundle = LossInputBundle(
        prediction=prediction,
        target=batch.target,
        mask=batch.mask,
        site_id=node_ids,
        horizon_index=tuple(range(1, int(prediction.shape[2]) + 1)),
        normalization_identity=str(protocol.get("target_loss_space", "normalized_target_space")),
        protocol_identity=str(protocol.get("protocol_id", protocol.get("protocol_version", "benchmark_v2"))),
        split=str(getattr(batch, "split", "synthetic")),
    )
    bundle.validate()
    return bundle


def call_loss(loss_fn: Any, prediction: Any, batch: Any, protocol: Mapping[str, Any]):
    if bool(getattr(loss_fn, "uses_loss_input_bundle", False)):
        return loss_fn(make_loss_input_bundle(prediction, batch, protocol))
    return loss_fn(prediction, batch.target, batch.mask)


__all__ = ["LossInputBundle", "call_loss", "make_loss_input_bundle"]
