from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from ..errors import ContractError


class MovingAverage(nn.Module):
    """Repeat the physical-power mean over the frozen trailing window."""

    def __init__(self, horizon: int = 10, ma_window: int = 144):
        super().__init__()
        self.horizon = int(horizon)
        self.ma_window = int(ma_window)
        if self.horizon <= 0:
            raise ContractError("MovingAverage horizon must be positive.")
        if self.ma_window <= 0:
            raise ContractError("MovingAverage ma_window must be positive.")

    def forward(self, physical_power_history: torch.Tensor) -> torch.Tensor:
        if physical_power_history.ndim != 2:
            raise ContractError(
                "MovingAverage expects physical power history shaped (B*N,T)."
            )
        time_steps = int(physical_power_history.shape[1])
        if time_steps < self.ma_window:
            raise ContractError(
                f"MovingAverage requires T >= ma_window; got T={time_steps}, "
                f"ma_window={self.ma_window}. No silent truncation is allowed."
            )
        window = physical_power_history[:, -self.ma_window :]
        if window.numel() == 0:
            raise ContractError("MovingAverage averaging window cannot be empty.")
        if not torch.isfinite(window).all():
            raise ContractError(
                "MovingAverage requires every Patv_clean_for_input value in the "
                "averaging window to be finite; NaN/Inf cannot be filled."
            )
        mean_power = window.mean(dim=1)
        return mean_power.unsqueeze(1).expand(-1, self.horizon)


def create_model(
    config: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> MovingAverage:
    config = dict(config or {})
    protocol = protocol or {}
    run_mode = str(config.get("run_mode", "formal"))
    ma_window = int(config.get("ma_window", protocol.get("lookback", 144)))
    formal_window = int(protocol.get("lookback", 144))
    if run_mode == "formal" and ma_window != formal_window:
        raise ContractError(
            f"Formal MovingAverage ma_window is frozen at {formal_window}; got "
            f"{ma_window}. It cannot be tuned from validation or test data."
        )
    if run_mode not in {"formal", "smoke"}:
        raise ContractError(f"Unknown MovingAverage run_mode: {run_mode}")
    horizon = int(config.get("horizon", protocol.get("max_pred_len", 10)))
    return MovingAverage(horizon=horizon, ma_window=ma_window)
