from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from ..errors import ContractError


class NodeSharedGRU(nn.Module):
    """Independent standard GRU with one parameter set shared by all nodes."""

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 1,
        horizon: int = 10,
        *,
        bidirectional: bool = False,
        dropout: float = 0.0,
        batch_first: bool = True,
    ):
        super().__init__()
        if bidirectional:
            raise ContractError("The E1-A NodeSharedGRU must be unidirectional.")
        if not batch_first:
            raise ContractError("The E1-A NodeSharedGRU requires batch_first=True.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.horizon = int(horizon)
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=float(dropout),
        )
        self.prediction_head = nn.Linear(self.hidden_dim, self.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or int(x.shape[-1]) != self.input_dim:
            raise ContractError(
                f"NodeSharedGRU expects (B*N,T,{self.input_dim}), got {tuple(x.shape)}"
            )
        if int(x.shape[1]) == 0:
            raise ContractError("NodeSharedGRU history cannot be empty.")
        if not torch.isfinite(x).all():
            raise ContractError("NodeSharedGRU input contains NaN/Inf.")
        _, final_hidden = self.gru(x)
        return self.prediction_head(final_hidden[-1])


def create_model(
    config: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> NodeSharedGRU:
    config = dict(config or {})
    protocol = protocol or {}
    return NodeSharedGRU(
        input_dim=int(config.get("input_dim", protocol.get("feature_count", 16))),
        hidden_dim=int(config.get("hidden_dim", 64)),
        num_layers=int(config.get("num_layers", 1)),
        horizon=int(config.get("horizon", protocol.get("max_pred_len", 10))),
        bidirectional=bool(config.get("bidirectional", False)),
        dropout=float(config.get("dropout", 0.0)),
        batch_first=bool(config.get("batch_first", True)),
    )
