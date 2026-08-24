from __future__ import annotations

from typing import Callable

import torch

from .contracts.loss import LossInputBundle
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


class BenchmarkMSMGDWULoss(torch.nn.Module):
    """Exact public-tensor adapter around the audited ST-MGPrompt loss.

    benchmark_v2 uses (B,N,H), while the original implementation uses
    (B,H,N). The transpose is the only semantic adapter.
    """

    uses_loss_input_bundle = True

    def __init__(self, *, eval_horizons=(3, 6, 10), num_nodes: int = 134, **profile):
        super().__init__()
        from st_mgprompt.losses import MSMGDWULoss

        self.impl = MSMGDWULoss(
            eval_horizons=list(eval_horizons),
            num_nodes=int(num_nodes),
            base_loss=str(profile.get("base_loss", "smooth_l1")),
            lambda_site=float(profile.get("lambda_site", 0.2)),
            ema_alpha=float(profile.get("ema_alpha", 0.9)),
            node_weight_clip=tuple(profile.get("node_weight_clip", (0.5, 3.0))),
            granularity_weight_mode=str(profile.get("granularity_weight_mode", "difficulty_rate")),
            site_weight_mode=str(profile.get("site_weight_mode", "dynamic")),
            difficulty_gamma=float(profile.get("difficulty_gamma", 1.0)),
            difficulty_rate_gamma=float(profile.get("difficulty_rate_gamma", 1.0)),
            difficulty_temperature=float(profile.get("difficulty_temperature", 1.0)),
            granularity_weight_clip=tuple(profile.get("granularity_weight_clip", (0.5, 3.0))),
        )

    @property
    def last_details(self):
        return self.impl.last_details

    def diagnostics_state(self):
        return self.impl.diagnostics_state()

    def forward(self, bundle: LossInputBundle):
        bundle.validate()
        return self.impl(
            bundle.prediction.transpose(1, 2),
            bundle.target.transpose(1, 2),
            bundle.mask.transpose(1, 2),
        )


def build_msmg_dwu_loss(*, eval_horizons=(3, 6, 10), num_nodes: int = 134, **profile):
    return BenchmarkMSMGDWULoss(eval_horizons=eval_horizons, num_nodes=num_nodes, **profile)


def get_loss(name: str) -> Callable:
    try:
        return LOSS_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark_v2 loss {name}; Dynamic MS-MG-DWU is not part of E0-B") from exc
