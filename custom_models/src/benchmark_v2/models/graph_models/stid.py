from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ...graph import GraphBundle
from .adaptive_common import e3_c_graph_identity


class ResidualMLP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.dropout(self.linear2(x))
        return self.activation(x + residual)


class STIDForecast(nn.Module):
    def __init__(self, config: dict[str, Any], bundle: GraphBundle):
        super().__init__()
        self.num_nodes = int(config["num_nodes"])
        self.horizon = int(config["output_len"])
        embed_dim = int(config["embed_dim"])
        node_dim = int(config["node_dim"])
        tod_dim = int(config["time_of_day_dim"])
        dow_dim = int(config["day_of_week_dim"])
        hidden_dim = embed_dim + node_dim + tod_dim + dow_dim
        self.graph_identity = e3_c_graph_identity(bundle, "stid")
        self.time_series_embedding = nn.Linear(
            int(config["input_len"]) * int(config["input_dim"]), embed_dim
        )
        self.node_embedding = nn.Parameter(
            torch.empty(self.num_nodes, node_dim)
        )
        nn.init.xavier_uniform_(self.node_embedding)
        self.time_of_day_embedding = nn.Embedding(
            int(config["time_of_day_size"]), tod_dim
        )
        self.day_of_week_embedding = nn.Embedding(
            int(config["day_of_week_size"]), dow_dim
        )
        self.encoder = nn.ModuleList(
            [
                ResidualMLP(hidden_dim, float(config["mlp_dropout"]))
                for _ in range(int(config["num_layers"]))
            ]
        )
        self.regression_head = nn.Linear(hidden_dim, self.horizon)
        self.last_shape_trace: dict[str, Any] = {}

    def forward(
        self,
        x: torch.Tensor,
        time_of_day_id: torch.Tensor,
        day_of_week_id: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(x.shape[1:]) != (144, self.num_nodes, 16):
            raise ValueError(
                f"STID requires (B,144,{self.num_nodes},16), got {tuple(x.shape)}"
            )
        if tuple(time_of_day_id.shape) != (x.shape[0],):
            raise ValueError("STID time_of_day_id must be shape (B,).")
        if tuple(day_of_week_id.shape) != (x.shape[0],):
            raise ValueError("STID day_of_week_id must be shape (B,).")
        if not bool(
            ((time_of_day_id >= 0) & (time_of_day_id < 144)).all()
        ):
            raise ValueError("STID time_of_day_id outside [0,143].")
        if not bool(((day_of_week_id >= 0) & (day_of_week_id < 7)).all()):
            raise ValueError("STID day_of_week_id outside [0,6].")
        history = x.permute(0, 2, 1, 3).reshape(x.shape[0], self.num_nodes, -1)
        series = self.time_series_embedding(history)
        nodes = self.node_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
        tod = (
            self.time_of_day_embedding(time_of_day_id)
            .unsqueeze(1)
            .expand(-1, self.num_nodes, -1)
        )
        dow = (
            self.day_of_week_embedding(day_of_week_id)
            .unsqueeze(1)
            .expand(-1, self.num_nodes, -1)
        )
        hidden = torch.cat((series, nodes, tod, dow), dim=-1)
        for layer in self.encoder:
            hidden = layer(hidden)
        prediction = self.regression_head(hidden)
        self.last_shape_trace = {
            "time_series_embedding": list(series.shape),
            "node_embedding": list(nodes.shape),
            "time_of_day_embedding": list(tod.shape),
            "day_of_week_embedding": list(dow.shape),
            "hidden": list(hidden.shape),
            "residual_mlp_layers": len(self.encoder),
            "graph_propagation": False,
        }
        return prediction


def create_model(
    *, config: dict[str, Any], graph_bundle: GraphBundle, **_: Any
) -> STIDForecast:
    return STIDForecast(config, graph_bundle)
