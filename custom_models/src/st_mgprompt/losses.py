from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


LOSS_STATE_SCHEMA_VERSION = "msmg_dwu_loss_state_v2"


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
        static_granularity_weights: tuple[float, ...] = (1.0, 2.0, 3.0),
        static_granularity_weight_source: str = "preset_arithmetic_progression",
        dwa_temperature: float = 2.0,
        dwa_update_timing: str = "epoch_end",
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
        if granularity_weight_mode not in {
            "static",
            "static_increasing",
            "uncertainty_precision",
            "difficulty_rate",
            "dynamic_weight_average",
        }:
            raise ValueError(
                "granularity_weight_mode must be static, static_increasing, uncertainty_precision, "
                "difficulty_rate, or dynamic_weight_average."
            )
        self.granularity_weight_mode = granularity_weight_mode
        if site_weight_mode not in {"static", "dynamic"}:
            raise ValueError("site_weight_mode must be static or dynamic.")
        self.site_weight_mode = site_weight_mode
        self.difficulty_gamma = float(difficulty_gamma)
        self.difficulty_rate_gamma = float(difficulty_rate_gamma)
        self.difficulty_temperature = float(difficulty_temperature)
        self.granularity_weight_clip = (float(granularity_weight_clip[0]), float(granularity_weight_clip[1]))
        if len(static_granularity_weights) != len(self.eval_horizons):
            raise ValueError("static_granularity_weights must match eval_horizons.")
        if any(float(value) <= 0 for value in static_granularity_weights):
            raise ValueError("static_granularity_weights must be positive.")
        if dwa_temperature <= 0:
            raise ValueError("dwa_temperature must be positive.")
        if dwa_update_timing != "epoch_end":
            raise ValueError("dwa_update_timing must be epoch_end.")
        self.static_granularity_weight_source = str(static_granularity_weight_source)
        self.dwa_temperature = float(dwa_temperature)
        self.dwa_update_timing = str(dwa_update_timing)
        self.loss_state_schema_version = LOSS_STATE_SCHEMA_VERSION
        self.log_sigma_g = nn.Parameter(
            torch.zeros(len(self.eval_horizons)),
            requires_grad=granularity_weight_mode == "uncertainty_precision",
        )
        static_weights = torch.as_tensor(static_granularity_weights, dtype=torch.float32)
        self.register_buffer("static_granularity_weight_raw", static_weights)
        self.register_buffer("static_granularity_weight", static_weights / static_weights.mean())
        self.register_buffer("ema_granularity_loss", torch.ones(len(self.eval_horizons)))
        self.register_buffer("initial_granularity_loss", torch.ones(len(self.eval_horizons)))
        self.register_buffer("initial_granularity_fitted", torch.zeros(len(self.eval_horizons), dtype=torch.bool))
        self.register_buffer("ema_node_loss", torch.ones(self.num_nodes))
        self.register_buffer("node_weight", torch.ones(self.num_nodes))
        self.register_buffer("dwa_weight", torch.ones(len(self.eval_horizons)))
        self.register_buffer("dwa_previous_epoch_loss", torch.ones(len(self.eval_horizons)))
        self.register_buffer("dwa_previous_fitted", torch.zeros(len(self.eval_horizons), dtype=torch.bool))
        self.register_buffer("dwa_epoch_loss_sum", torch.zeros(len(self.eval_horizons)))
        self.register_buffer("dwa_epoch_loss_count", torch.zeros(len(self.eval_horizons)))
        self.register_buffer("dwa_update_count", torch.zeros((), dtype=torch.long))
        self.last_details: dict[str, float | list[float]] = {}

    def _base_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        if self.base_loss == "mae":
            return masked_mae(pred, target, mask)
        if self.base_loss == "mse":
            return masked_mse(pred, target, mask)
        return masked_smooth_l1(pred, target, mask)

    def begin_training_epoch(self) -> None:
        if self.granularity_weight_mode == "dynamic_weight_average":
            self.dwa_epoch_loss_sum.zero_()
            self.dwa_epoch_loss_count.zero_()

    @torch.no_grad()
    def end_training_epoch(self) -> dict[str, float]:
        if self.granularity_weight_mode != "dynamic_weight_average":
            return {}
        valid = self.dwa_epoch_loss_count > 0
        if not valid.any().item():
            return {
                f"granularity_weight_h{int(horizon)}": float(self.dwa_weight[index].detach().cpu())
                for index, horizon in enumerate(self.eval_horizons)
            }
        current = torch.where(
            valid,
            self.dwa_epoch_loss_sum / self.dwa_epoch_loss_count.clamp_min(1.0),
            self.dwa_previous_epoch_loss,
        )
        ratios = torch.ones_like(current)
        comparable = valid & self.dwa_previous_fitted
        if comparable.any().item():
            ratios[comparable] = current[comparable] / self.dwa_previous_epoch_loss[comparable].clamp_min(1e-12)
            next_weight = self.dwa_weight.clone()
            raw_weight = comparable.sum() * torch.softmax(ratios[comparable] / self.dwa_temperature, dim=0)
            clipped = raw_weight.clamp(*self.granularity_weight_clip)
            next_weight[comparable] = clipped / clipped.mean().clamp_min(1e-12)
            self.dwa_weight.copy_(next_weight)
        self.dwa_previous_epoch_loss.copy_(torch.where(valid, current, self.dwa_previous_epoch_loss))
        self.dwa_previous_fitted.logical_or_(valid)
        self.dwa_update_count.add_(1)
        details: dict[str, float] = {"dwa_update_count": float(self.dwa_update_count.item())}
        for index, horizon in enumerate(self.eval_horizons):
            details[f"dwa_epoch_loss_h{int(horizon)}"] = float(current[index].detach().cpu())
            details[f"dwa_loss_ratio_h{int(horizon)}"] = float(ratios[index].detach().cpu())
            details[f"granularity_weight_h{int(horizon)}"] = float(self.dwa_weight[index].detach().cpu())
        self.last_details.update(details)
        return details

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
        if not valid_granularity.any().item():
            self.last_details = {"skipped_all_invalid_batch": 1.0}
            return None
        preclip_weights = torch.ones_like(loss_vector)
        if self.granularity_weight_mode == "difficulty_rate":
            with torch.no_grad():
                detached_losses = torch.where(valid_granularity, loss_vector.detach(), self.ema_granularity_loss.to(pred.device))
                if self.training:
                    ema_current = self.ema_granularity_loss.to(pred.device)
                    updated = self.ema_alpha * ema_current + (1.0 - self.ema_alpha) * detached_losses
                    updated = torch.where(valid_granularity, updated, ema_current)
                    self.ema_granularity_loss.copy_(updated.detach())
                    fitted = self.initial_granularity_fitted.to(pred.device)
                    initial = self.initial_granularity_loss.to(pred.device)
                    new_initial = torch.where(valid_granularity & ~fitted, detached_losses.clamp_min(1e-12), initial)
                    new_fitted = fitted | valid_granularity
                    self.initial_granularity_loss.copy_(new_initial.detach())
                    self.initial_granularity_fitted.copy_(new_fitted.detach())
                ema_losses = self.ema_granularity_loss.to(pred.device).clamp_min(1e-12)
                initial_losses = self.initial_granularity_loss.to(pred.device).clamp_min(1e-12)
                difficulty_level = ema_losses / ema_losses[valid_granularity].mean().clamp_min(1e-12)
                relative_rate = ema_losses / initial_losses
                relative_rate = relative_rate / relative_rate[valid_granularity].mean().clamp_min(1e-12)
                logits = (
                    self.difficulty_gamma * torch.log(difficulty_level.clamp_min(1e-12))
                    + self.difficulty_rate_gamma * torch.log(relative_rate.clamp_min(1e-12))
                )
                weights = torch.ones_like(loss_vector)
                preclip_valid = valid_granularity.sum() * torch.softmax(
                    logits[valid_granularity] / self.difficulty_temperature, dim=0
                )
                preclip_weights[valid_granularity] = preclip_valid
                clipped = preclip_valid.clamp(*self.granularity_weight_clip)
                weights[valid_granularity] = clipped / clipped.mean().clamp_min(1e-12)
            granularity_loss = (weights.detach()[valid_granularity] * loss_vector[valid_granularity]).mean()
        elif self.granularity_weight_mode == "uncertainty_precision":
            granularity_loss = torch.stack(granularity_terms).mean()
            weights = torch.exp(-self.log_sigma_g.detach())
            preclip_weights = weights
            ema_losses = self.ema_granularity_loss.to(pred.device)
            initial_losses = self.initial_granularity_loss.to(pred.device)
            difficulty_level = ema_losses / ema_losses.mean().clamp_min(1e-12)
            relative_rate = ema_losses / initial_losses.clamp_min(1e-12)
        elif self.granularity_weight_mode == "static_increasing":
            weights = self.static_granularity_weight.to(pred.device)
            preclip_weights = self.static_granularity_weight_raw.to(pred.device)
            granularity_loss = (weights[valid_granularity] * loss_vector[valid_granularity]).mean()
            ema_losses = self.ema_granularity_loss.to(pred.device)
            initial_losses = self.initial_granularity_loss.to(pred.device)
            difficulty_level = torch.ones_like(ema_losses)
            relative_rate = torch.ones_like(ema_losses)
        elif self.granularity_weight_mode == "dynamic_weight_average":
            weights = self.dwa_weight.to(pred.device)
            preclip_weights = weights
            if self.training:
                with torch.no_grad():
                    self.dwa_epoch_loss_sum[valid_granularity] += loss_vector.detach()[valid_granularity]
                    self.dwa_epoch_loss_count[valid_granularity] += 1
            granularity_loss = (weights.detach()[valid_granularity] * loss_vector[valid_granularity]).mean()
            ema_losses = self.dwa_previous_epoch_loss.to(pred.device)
            initial_losses = self.initial_granularity_loss.to(pred.device)
            difficulty_level = torch.ones_like(ema_losses)
            relative_rate = torch.ones_like(ema_losses)
        else:
            granularity_loss = self._base_loss(pred, target, mask)
            if granularity_loss is None:
                self.last_details = {"skipped_all_invalid_batch": 1.0}
                return None
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
                self.node_weight.copy_(torch.where(valid_node, node_weights, self.node_weight))
        detached_weight = self.node_weight.detach()
        if valid_node.any().item():
            site_loss = (detached_weight[valid_node] * node_losses[valid_node]).mean()
        else:
            site_loss = pred.new_tensor(0.0)
        total = (
            granularity_loss + self.lambda_site * site_loss
            if self.site_weight_mode == "dynamic"
            else granularity_loss
        )

        details: dict[str, float | list[float]] = {
            "total_loss": float(total.detach().cpu()),
            "site_loss": float(site_loss.detach().cpu()),
            "site_weight_mean": float(detached_weight.mean().detach().cpu()),
            "site_weight_max": float(detached_weight.max().detach().cpu()),
            "site_weight_min": float(detached_weight.min().detach().cpu()),
            "valid_horizon_count": float(valid_granularity.sum().item()),
            "valid_node_count": float(valid_node.sum().item()),
            "granularity_weight_clip_low_fraction": float(
                (preclip_weights[valid_granularity] < self.granularity_weight_clip[0]).float().mean().detach().cpu()
            ),
            "granularity_weight_clip_high_fraction": float(
                (preclip_weights[valid_granularity] > self.granularity_weight_clip[1]).float().mean().detach().cpu()
            ),
            "site_weight_clip_low_fraction": float(
                (detached_weight[valid_node] <= self.node_weight_clip[0]).float().mean().detach().cpu()
            ),
            "site_weight_clip_high_fraction": float(
                (detached_weight[valid_node] >= self.node_weight_clip[1]).float().mean().detach().cpu()
            ),
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
            details[f"granularity_weight_preclip_h{key_h}"] = float(preclip_weights[idx].detach().cpu())
            details[f"granularity_weight_h{key_h}"] = weight_value
            details[f"weighted_contribution_h{key_h}"] = raw_loss * weight_value if raw_loss == raw_loss else float("nan")
        self.last_details = details
        return total

    def diagnostics_state(self) -> dict[str, torch.Tensor | list[int] | str | float | tuple[float, float]]:
        return {
            "loss_state_schema_version": self.loss_state_schema_version,
            "eval_horizons": list(self.eval_horizons),
            "base_loss": self.base_loss,
            "lambda_site": self.lambda_site,
            "ema_alpha": self.ema_alpha,
            "node_weight_clip": self.node_weight_clip,
            "granularity_weight_mode": self.granularity_weight_mode,
            "site_weight_mode": self.site_weight_mode,
            "static_granularity_weight_source": self.static_granularity_weight_source,
            "static_granularity_weight_raw": self.static_granularity_weight_raw.detach().cpu(),
            "static_granularity_weight": self.static_granularity_weight.detach().cpu(),
            "dwa_temperature": self.dwa_temperature,
            "dwa_update_timing": self.dwa_update_timing,
            "dwa_weight": self.dwa_weight.detach().cpu(),
            "dwa_previous_epoch_loss": self.dwa_previous_epoch_loss.detach().cpu(),
            "dwa_previous_fitted": self.dwa_previous_fitted.detach().cpu(),
            "dwa_update_count": int(self.dwa_update_count.item()),
            "granularity_weight": (
                torch.exp(-self.log_sigma_g.detach()).cpu()
                if self.granularity_weight_mode == "uncertainty_precision"
                else self.static_granularity_weight.detach().cpu()
                if self.granularity_weight_mode == "static_increasing"
                else self.dwa_weight.detach().cpu()
                if self.granularity_weight_mode == "dynamic_weight_average"
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
            static_granularity_weights=tuple(getattr(config, "static_granularity_weights", (1.0, 2.0, 3.0))),
            static_granularity_weight_source=getattr(
                config, "static_granularity_weight_source", "preset_arithmetic_progression"
            ),
            dwa_temperature=float(getattr(config, "dwa_temperature", 2.0)),
            dwa_update_timing=getattr(config, "dwa_update_timing", "epoch_end"),
        )
    raise ValueError(f"Unknown loss function: {name}")
