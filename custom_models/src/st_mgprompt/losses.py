from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _validate_shapes(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or target.shape != mask.shape:
        raise ValueError(
            f"pred/target/mask shapes must match, got {pred.shape}, {target.shape}, {mask.shape}."
        )
    if pred.ndim != 3:
        raise ValueError(f"pred/target/mask must be [B,H,N], got pred ndim={pred.ndim}.")
    return mask.float()


def _masked_reduce(loss_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    mask = mask.float()
    if mask.sum().item() <= 0:
        return None
    denom = mask.sum().clamp_min(1.0)
    return (loss_values * mask).sum() / denom


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    """Masked MAE in normalized target space; shapes are [B,H,N]."""

    mask = _validate_shapes(pred, target, mask)
    return _masked_reduce((pred - target).abs(), mask)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    """Masked MSE in normalized target space; shapes are [B,H,N]."""

    mask = _validate_shapes(pred, target, mask)
    return _masked_reduce((pred - target).square(), mask)


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    """Masked SmoothL1 in normalized target space; shapes are [B,H,N]."""

    mask = _validate_shapes(pred, target, mask)
    return _masked_reduce(F.smooth_l1_loss(pred, target, reduction="none"), mask)


def masked_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    """Masked RMSE in normalized target space; shapes are [B,H,N]."""

    mse = masked_mse(pred, target, mask)
    if mse is None:
        return None
    return torch.sqrt(mse + eps)


def masked_score_aligned_hybrid_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    """Fair-main loss in normalized target space.

    Shapes are [B,H,N]. The mask is used only for loss weighting.
    """

    mask = _validate_shapes(pred, target, mask)
    if mask.sum().item() <= 0:
        return None
    err = pred - target
    abs_err = err.abs() * mask
    sq_err = err.square() * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    mae = abs_err.sum(dim=1) / denom
    rmse = torch.sqrt(sq_err.sum(dim=1) / denom + eps)
    valid_node = (mask.sum(dim=1) > 0).float()
    node_loss = 0.5 * rmse + 0.5 * mae
    valid_node_count = valid_node.sum().clamp_min(1.0)
    return (node_loss * valid_node).sum() / valid_node_count


def _node_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    base_loss: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if base_loss == "mae":
        values = (pred - target).abs()
    elif base_loss == "mse":
        values = (pred - target).square()
    elif base_loss == "smooth_l1":
        values = F.smooth_l1_loss(pred, target, reduction="none")
    else:
        raise ValueError(f"Unsupported base_loss: {base_loss}")
    denom = mask.sum(dim=(0, 1)).clamp_min(1.0)
    losses = (values * mask).sum(dim=(0, 1)) / denom
    valid = mask.sum(dim=(0, 1)) > 0
    return losses, valid


class MSMGDWULoss(nn.Module):
    """Multi-site multi-granularity dynamic weighted uncertainty loss.

    This loss is for the method-full model and loss ablations only. It must not
    be mixed into the strict fair-main table.
    """

    def __init__(
        self,
        eval_horizons: list[int],
        num_nodes: int,
        base_loss: str = "smooth_l1",
        lambda_site: float = 0.2,
        ema_alpha: float = 0.9,
        node_weight_clip: tuple[float, float] = (0.5, 3.0),
        granularity_weight_mode: str = "uncertainty_precision",
        site_weight_mode: str = "dynamic",
        difficulty_gamma: float = 1.0,
        difficulty_rate_gamma: float = 1.0,
        difficulty_temperature: float = 1.0,
        granularity_weight_clip: tuple[float, float] = (0.5, 3.0),
    ) -> None:
        super().__init__()
        if not eval_horizons:
            raise ValueError("eval_horizons must not be empty.")
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")
        if base_loss not in {"mae", "mse", "smooth_l1"}:
            raise ValueError("base_loss must be mae, mse, or smooth_l1.")
        clip_min, clip_max = node_weight_clip
        if clip_min <= 0 or clip_max < clip_min:
            raise ValueError("node_weight_clip must satisfy 0 < min <= max.")
        self.eval_horizons = [int(h) for h in eval_horizons]
        self.num_nodes = int(num_nodes)
        self.base_loss = base_loss
        self.lambda_site = float(lambda_site)
        self.ema_alpha = float(ema_alpha)
        self.node_weight_clip = (float(clip_min), float(clip_max))
        if granularity_weight_mode not in {"static", "uncertainty_precision", "difficulty_rate"}:
            raise ValueError("granularity_weight_mode must be static, uncertainty_precision, or difficulty_rate.")
        self.granularity_weight_mode = granularity_weight_mode
        if site_weight_mode not in {"static", "dynamic"}:
            raise ValueError("site_weight_mode must be static or dynamic.")
        self.site_weight_mode = site_weight_mode
        self.difficulty_gamma = float(difficulty_gamma)
        self.difficulty_rate_gamma = float(difficulty_rate_gamma)
        self.difficulty_temperature = float(difficulty_temperature)
        self.granularity_weight_clip = (float(granularity_weight_clip[0]), float(granularity_weight_clip[1]))
        self.log_sigma_g = nn.Parameter(
            torch.zeros(len(self.eval_horizons)),
            requires_grad=granularity_weight_mode == "uncertainty_precision",
        )
        self.register_buffer("ema_granularity_loss", torch.ones(len(self.eval_horizons)))
        self.register_buffer("initial_granularity_loss", torch.ones(len(self.eval_horizons)))
        self.register_buffer("initial_granularity_fitted", torch.zeros(len(self.eval_horizons), dtype=torch.bool))
        self.register_buffer("ema_node_loss", torch.ones(self.num_nodes))
        self.register_buffer("node_weight", torch.ones(self.num_nodes))
        self.last_details: dict[str, float | list[float]] = {}

    def _base_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        if self.base_loss == "mae":
            return masked_mae(pred, target, mask)
        if self.base_loss == "mse":
            return masked_mse(pred, target, mask)
        return masked_smooth_l1(pred, target, mask)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        mask = _validate_shapes(pred, target, mask)
        if mask.sum().item() <= 0:
            self.last_details = {"skipped_all_invalid_batch": 1.0}
            return None
        if pred.shape[-1] != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got {pred.shape[-1]}.")

        granularity_terms: list[torch.Tensor] = []
        granularity_losses: dict[int, torch.Tensor] = {}
        loss_vector = pred.new_full((len(self.eval_horizons),), float("nan"))
        valid_granularity = torch.zeros(len(self.eval_horizons), dtype=torch.bool, device=pred.device)
        for idx, horizon in enumerate(self.eval_horizons):
            h = min(int(horizon), pred.shape[1])
            loss_h = self._base_loss(pred[:, :h, :], target[:, :h, :], mask[:, :h, :])
            if loss_h is None:
                continue
            granularity_losses[horizon] = loss_h
            loss_vector[idx] = loss_h
            valid_granularity[idx] = True
            if self.granularity_weight_mode == "uncertainty_precision":
                granularity_terms.append(torch.exp(-self.log_sigma_g[idx]) * loss_h + self.log_sigma_g[idx])
        if not granularity_terms:
            if not valid_granularity.any().item():
                self.last_details = {"skipped_all_invalid_batch": 1.0}
                return None
        if self.granularity_weight_mode == "difficulty_rate":
            with torch.no_grad():
                detached_losses = torch.where(valid_granularity, loss_vector.detach(), self.ema_granularity_loss.to(pred.device))
                if self.training:
                    ema_current = self.ema_granularity_loss.to(pred.device)
                    updated = self.ema_alpha * ema_current + (1.0 - self.ema_alpha) * detached_losses
                    updated = torch.where(valid_granularity, updated, ema_current)
                    self.ema_granularity_loss.copy_(updated.detach().cpu() if updated.device.type != "cpu" else updated.detach())
                    fitted = self.initial_granularity_fitted.to(pred.device)
                    initial = self.initial_granularity_loss.to(pred.device)
                    new_initial = torch.where(valid_granularity & ~fitted, detached_losses.clamp_min(1e-12), initial)
                    new_fitted = fitted | valid_granularity
                    self.initial_granularity_loss.copy_(
                        new_initial.detach().cpu() if new_initial.device.type != "cpu" else new_initial.detach()
                    )
                    self.initial_granularity_fitted.copy_(
                        new_fitted.detach().cpu() if new_fitted.device.type != "cpu" else new_fitted.detach()
                    )
                ema_losses = self.ema_granularity_loss.to(pred.device).clamp_min(1e-12)
                initial_losses = self.initial_granularity_loss.to(pred.device).clamp_min(1e-12)
                difficulty_level = ema_losses / ema_losses[valid_granularity].mean().clamp_min(1e-12)
                relative_rate = ema_losses / initial_losses
                relative_rate = relative_rate / relative_rate[valid_granularity].mean().clamp_min(1e-12)
                logits = (
                    self.difficulty_gamma * torch.log(difficulty_level.clamp_min(1e-12))
                    + self.difficulty_rate_gamma * torch.log(relative_rate.clamp_min(1e-12))
                )
                weights = len(self.eval_horizons) * torch.softmax(logits / self.difficulty_temperature, dim=0)
                weights = weights.clamp(*self.granularity_weight_clip)
                weights = weights / weights.mean().clamp_min(1e-12)
            granularity_loss = (weights.detach()[valid_granularity] * loss_vector[valid_granularity]).mean()
        elif self.granularity_weight_mode == "uncertainty_precision":
            granularity_loss = torch.stack(granularity_terms).mean()
            weights = torch.exp(-self.log_sigma_g.detach())
            ema_losses = self.ema_granularity_loss.to(pred.device)
            initial_losses = self.initial_granularity_loss.to(pred.device)
            difficulty_level = ema_losses / ema_losses.mean().clamp_min(1e-12)
            relative_rate = ema_losses / initial_losses.clamp_min(1e-12)
        else:
            granularity_loss = loss_vector[valid_granularity].mean()
            weights = torch.ones_like(loss_vector)
            ema_losses = self.ema_granularity_loss.to(pred.device)
            initial_losses = self.initial_granularity_loss.to(pred.device)
            difficulty_level = torch.ones_like(ema_losses)
            relative_rate = torch.ones_like(ema_losses)

        node_losses, valid_node = _node_loss(pred, target, mask, self.base_loss)
        with torch.no_grad():
            if self.site_weight_mode == "dynamic" and self.training and valid_node.any().item():
                current = node_losses.detach()
                updated = self.ema_alpha * self.ema_node_loss + (1.0 - self.ema_alpha) * current
                self.ema_node_loss.copy_(torch.where(valid_node, updated, self.ema_node_loss))
                mean_loss = self.ema_node_loss[valid_node].mean().clamp_min(1e-6)
                node_weights = (self.ema_node_loss / mean_loss).clamp(*self.node_weight_clip)
                self.node_weight.copy_(node_weights)
        detached_weight = self.node_weight.detach()
        if valid_node.any().item():
            site_loss = (detached_weight[valid_node] * node_losses[valid_node]).mean()
        else:
            site_loss = pred.new_tensor(0.0)
        total = granularity_loss + self.lambda_site * site_loss

        details: dict[str, float | list[float]] = {
            "total_loss": float(total.detach().cpu()),
            "site_loss": float(site_loss.detach().cpu()),
            "site_weight_mean": float(detached_weight.mean().detach().cpu()),
            "site_weight_max": float(detached_weight.max().detach().cpu()),
            "site_weight_min": float(detached_weight.min().detach().cpu()),
            "skipped_all_invalid_batch": 0.0,
        }
        for idx, horizon in enumerate(self.eval_horizons):
            key_h = int(horizon)
            raw_loss = (
                float(granularity_losses[horizon].detach().cpu()) if horizon in granularity_losses else float("nan")
            )
            weight_value = float(weights[idx].detach().cpu())
            details[f"granularity_loss_h{key_h}"] = raw_loss
            details[f"raw_loss_h{key_h}"] = raw_loss
            details[f"ema_loss_h{key_h}"] = float(ema_losses[idx].detach().cpu())
            details[f"initial_loss_h{key_h}"] = float(initial_losses[idx].detach().cpu())
            details[f"difficulty_level_h{key_h}"] = float(difficulty_level[idx].detach().cpu())
            details[f"relative_training_rate_h{key_h}"] = float(relative_rate[idx].detach().cpu())
            details[f"granularity_weight_h{key_h}"] = weight_value
            details[f"weighted_contribution_h{key_h}"] = raw_loss * weight_value if raw_loss == raw_loss else float("nan")
        self.last_details = details
        return total

    def diagnostics_state(self) -> dict[str, torch.Tensor | list[int] | str | float | tuple[float, float]]:
        return {
            "eval_horizons": list(self.eval_horizons),
            "base_loss": self.base_loss,
            "lambda_site": self.lambda_site,
            "ema_alpha": self.ema_alpha,
            "node_weight_clip": self.node_weight_clip,
            "granularity_weight_mode": self.granularity_weight_mode,
            "site_weight_mode": self.site_weight_mode,
            "granularity_weight": (
                torch.exp(-self.log_sigma_g.detach()).cpu()
                if self.granularity_weight_mode == "uncertainty_precision"
                else torch.as_tensor([
                    self.last_details.get(f"granularity_weight_h{int(h)}", 1.0) for h in self.eval_horizons
                ])
            ),
            "log_sigma_g": self.log_sigma_g.detach().cpu(),
            "ema_granularity_loss": self.ema_granularity_loss.detach().cpu(),
            "initial_granularity_loss": self.initial_granularity_loss.detach().cpu(),
            "ema_node_loss": self.ema_node_loss.detach().cpu(),
            "node_weight": self.node_weight.detach().cpu(),
        }


def get_loss_metadata(config) -> dict[str, bool | str]:
    if getattr(config, "use_msmg_dwu", False) or config.loss_function == "msmg_dwu_loss":
        return {
            "model": "STMGPrompt_Full_MSMGDWU",
            "loss_function": "msmg_dwu_loss",
            "loss_changed_from_fair_protocol": True,
            "eligible_for_fair_main_table": False,
            "eligible_for_method_full_table": True,
            "fair_loss_comparison": False,
        }
    return {
        "loss_function": "masked_score_aligned_hybrid",
        "loss_changed_from_fair_protocol": False,
        "eligible_for_fair_main_table": bool(config.eligible_for_fair_main_table),
        "eligible_for_method_full_table": False,
        "fair_loss_comparison": True,
    }


def get_loss_fn(name: str, config=None, num_nodes: int | None = None):
    if name == "masked_score_aligned_hybrid":
        return masked_score_aligned_hybrid_loss
    if name == "msmg_dwu_loss":
        if config is None:
            raise ValueError("config is required to build msmg_dwu_loss.")
        return MSMGDWULoss(
            eval_horizons=list(config.eval_horizons),
            num_nodes=int(num_nodes if num_nodes is not None else config.num_nodes),
            base_loss=getattr(config, "msmg_base_loss", "smooth_l1"),
            lambda_site=float(getattr(config, "msmg_lambda_site", 0.2)),
            ema_alpha=float(getattr(config, "msmg_ema_alpha", 0.9)),
            node_weight_clip=tuple(getattr(config, "msmg_node_weight_clip", (0.5, 3.0))),
            granularity_weight_mode=getattr(config, "granularity_weight_mode", "uncertainty_precision"),
            site_weight_mode=getattr(config, "site_weight_mode", "dynamic"),
            difficulty_gamma=float(getattr(config, "difficulty_gamma", 1.0)),
            difficulty_rate_gamma=float(getattr(config, "difficulty_rate_gamma", 1.0)),
            difficulty_temperature=float(getattr(config, "difficulty_temperature", 1.0)),
            granularity_weight_clip=tuple(getattr(config, "granularity_weight_clip", (0.5, 3.0))),
        )
    raise ValueError(f"Unknown loss function: {name}")
