from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from ...graph import GraphBundle
from .common import graph_identity_dict, runtime_support


class GCNLayer(nn.Module):
    """Kipf-Welling first-order propagation: support @ X @ W."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        propagated = torch.einsum("nm,...mc->...nc", support, x)
        return self.linear(propagated)


class GCNForecast(nn.Module):
    """Two spatial GCN layers plus the frozen shared forecast adaptation."""

    def __init__(
        self,
        *,
        bundle: GraphBundle,
        input_dim: int = 16,
        hidden_dim: int = 64,
        lookback: int = 144,
        horizon: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.graph1 = GCNLayer(self.input_dim, self.hidden_dim)
        self.graph2 = GCNLayer(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.temporal_projection = nn.Linear(self.lookback, self.horizon)
        self.output_projection = nn.Linear(self.hidden_dim, 1)
        self.register_buffer(
            "A_gcn", runtime_support(bundle, "A_gcn"), persistent=False
        )
        self.graph_identity = graph_identity_dict(bundle, ("A_gcn",))
        self.last_shape_trace: dict[str, list[int]] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != (
            self.lookback,
            self.A_gcn.shape[0],
            self.input_dim,
        ):
            raise ValueError(
                "GCNForecast requires "
                f"(B,{self.lookback},{self.A_gcn.shape[0]},{self.input_dim}), "
                f"got {tuple(x.shape)}."
            )
        h1 = torch.relu(self.graph1(x, self.A_gcn))
        h1_drop = self.dropout(h1)
        h2 = torch.relu(self.graph2(h1_drop, self.A_gcn))
        temporal = self.temporal_projection(
            h2.permute(0, 2, 3, 1).contiguous()
        )
        prediction = self.output_projection(
            temporal.permute(0, 1, 3, 2).contiguous()
        ).squeeze(-1)
        self.last_shape_trace = {
            "input": list(x.shape),
            "graph_layer_1": list(h1.shape),
            "graph_layer_2": list(h2.shape),
            "temporal_projection": list(temporal.shape),
            "prediction": list(prediction.shape),
        }
        return prediction


def create_model(
    config: Mapping[str, Any] | None = None,
    protocol: Any = None,
    *,
    graph_bundle: GraphBundle,
) -> GCNForecast:
    values = dict(config or {})
    return GCNForecast(
        bundle=graph_bundle,
        input_dim=int(values.get("input_dim", 16)),
        hidden_dim=int(values.get("hidden_dim", 64)),
        lookback=int(values.get("lookback", 144)),
        horizon=int(values.get("horizon", 10)),
        dropout=float(values.get("dropout", 0.1)),
    )
