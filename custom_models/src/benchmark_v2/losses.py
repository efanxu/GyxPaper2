from __future__ import annotations

from typing import Callable

import torch

from .errors import ContractError


def _validate(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or target.shape != mask.shape or pred.ndim != 3:
        raise ContractError(f"pred/target/mask must align as (B,N,H), got {pred.shape}, {target.shape}, {mask.shape}")
    return mask.to(dtype=pred.dtype)


def _reduce(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    valid = mask.sum()
    if int(valid.detach().item()) == 0:
        return None
    return (values * mask).sum() / valid.clamp_min(1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    mask = _validate(pred, target, mask)
    return _reduce((pred - target).square(), mask)


def masked_score_aligned_hybrid(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor | None:
    """Fair-main H3/H6/H10 aligned hybrid in normalized target space."""
    mask = _validate(pred, target, mask)
    if int(mask.sum().detach().item()) == 0:
        return None
    err = pred - target
    mae = (err.abs() * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
    rmse = torch.sqrt((err.square() * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0) + eps)
    valid_node = (mask.sum(dim=2) > 0).to(pred.dtype)
    node_loss = 0.5 * mae + 0.5 * rmse
    return (node_loss * valid_node).sum() / valid_node.sum().clamp_min(1.0)


LOSS_REGISTRY: dict[str, Callable] = {
    "masked_mse": masked_mse,
    "masked_score_aligned_hybrid": masked_score_aligned_hybrid,
}


def get_loss(name: str) -> Callable:
    try:
        return LOSS_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark_v2 loss {name}; Dynamic MS-MG-DWU is not part of E0-B") from exc

