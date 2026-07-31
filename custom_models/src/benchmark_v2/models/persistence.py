from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from ..errors import ContractError


class Persistence(nn.Module):
    """Repeat the last visible physical Patv_clean_for_input value."""

    def __init__(self, horizon: int = 10):
        super().__init__()
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ContractError("Persistence horizon must be positive.")

    def forward(self, physical_power_history: torch.Tensor) -> torch.Tensor:
        if physical_power_history.ndim != 2 or physical_power_history.shape[1] == 0:
            raise ContractError(
                "Persistence expects physical power history shaped (B*N,T) with T>0."
            )
        last_visible = physical_power_history[:, -1]
        if not torch.isfinite(last_visible).all():
            raise ContractError(
                "Persistence requires the last visible Patv_clean_for_input value "
                "for every node to be finite; NaN/Inf cannot be filled or searched backward."
            )
        return last_visible.unsqueeze(1).expand(-1, self.horizon)


def create_model(
    config: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> Persistence:
    config = dict(config or {})
    horizon = int(config.get("horizon", (protocol or {}).get("max_pred_len", 10)))
    return Persistence(horizon=horizon)
