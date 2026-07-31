from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ...graph import GraphBundle
from .adaptive_common import e3_c_graph_identity


class AdaptiveGraphConv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        cheb_order: int,
        embed_dim: int,
    ):
        super().__init__()
        if int(cheb_order) != 2:
            raise ValueError("Canonical AGCRN freezes cheb_order=2 (T0,T1).")
        self.cheb_order = 2
        self.weights_pool = nn.Parameter(
            torch.empty(embed_dim, self.cheb_order, input_dim, output_dim)
        )
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, output_dim))
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)
        self.last_node_weight_variation = 0.0

    def forward(
        self,
        x: torch.Tensor,
        node_embeddings: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        identity = torch.eye(
            adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device
        )
        supports = torch.stack((identity, adjacency), dim=0)
        propagated = torch.einsum("knm,bmi->bnki", supports, x)
        weights = torch.einsum(
            "ne,ekio->nkio", node_embeddings, self.weights_pool
        )
        bias = node_embeddings @ self.bias_pool
        self.last_node_weight_variation = float(
            weights.detach().float().var(dim=0).mean().cpu()
        )
        return torch.einsum("bnki,nkio->bno", propagated, weights) + bias


class AGCRNCell(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, *, cheb_order: int, embed_dim: int
    ):
        super().__init__()
        combined = input_dim + hidden_dim
        self.hidden_dim = int(hidden_dim)
        self.gate = AdaptiveGraphConv(
            combined,
            2 * hidden_dim,
            cheb_order=cheb_order,
            embed_dim=embed_dim,
        )
        self.candidate = AdaptiveGraphConv(
            combined,
            hidden_dim,
            cheb_order=cheb_order,
            embed_dim=embed_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        node_embeddings: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        gates = torch.sigmoid(
            self.gate(
                torch.cat((x, state), dim=-1), node_embeddings, adjacency
            )
        )
        update, reset = gates.chunk(2, dim=-1)
        candidate = torch.tanh(
            self.candidate(
                torch.cat((x, reset * state), dim=-1),
                node_embeddings,
                adjacency,
            )
        )
        return update * state + (1.0 - update) * candidate


class AGCRNForecast(nn.Module):
    def __init__(self, config: dict[str, Any], bundle: GraphBundle):
        super().__init__()
        self.num_nodes = int(config["num_nodes"])
        self.horizon = int(config["horizon"])
        self.hidden_dim = int(config["rnn_units"])
        embed_dim = int(config["embed_dim"])
        self.graph_identity = e3_c_graph_identity(bundle, "agcrn")
        self.node_embeddings = nn.Parameter(
            torch.empty(self.num_nodes, embed_dim)
        )
        nn.init.xavier_uniform_(self.node_embeddings)
        dims = [int(config["input_dim"])] + [
            self.hidden_dim
        ] * int(config["num_layers"])
        self.cells = nn.ModuleList(
            [
                AGCRNCell(
                    dims[index],
                    self.hidden_dim,
                    cheb_order=int(config["cheb_order"]),
                    embed_dim=embed_dim,
                )
                for index in range(int(config["num_layers"]))
            ]
        )
        self.horizon_head = nn.Linear(self.hidden_dim, self.horizon)
        self.last_shape_trace: dict[str, Any] = {}

    def adaptive_adjacency(self) -> torch.Tensor:
        scores = torch.relu(self.node_embeddings @ self.node_embeddings.T)
        return torch.softmax(scores, dim=1)

    def forward(
        self, x: torch.Tensor, *, adjacency_override: torch.Tensor | None = None
    ) -> torch.Tensor:
        if tuple(x.shape[1:]) != (144, self.num_nodes, 16):
            raise ValueError(
                f"AGCRN requires (B,144,{self.num_nodes},16), got {tuple(x.shape)}"
            )
        adjacency = (
            self.adaptive_adjacency()
            if adjacency_override is None
            else adjacency_override
        )
        layer_input = x
        final_states: list[torch.Tensor] = []
        for cell in self.cells:
            state = torch.zeros(
                x.shape[0],
                self.num_nodes,
                self.hidden_dim,
                dtype=x.dtype,
                device=x.device,
            )
            outputs = []
            for time_index in range(x.shape[1]):
                state = cell(
                    layer_input[:, time_index],
                    state,
                    self.node_embeddings,
                    adjacency,
                )
                outputs.append(state)
            layer_input = torch.stack(outputs, dim=1)
            final_states.append(state)
        prediction = self.horizon_head(final_states[-1])
        self.last_shape_trace = {
            "adaptive_basis_count": 2,
            "adaptive_basis_names": ["T0_identity", "T1_adaptive"],
            "encoder_layers": len(self.cells),
            "encoder_output": list(layer_input.shape),
            "last_hidden": list(final_states[-1].shape),
            "direct_horizon_head": list(prediction.shape),
        }
        return prediction


def create_model(
    *, config: dict[str, Any], graph_bundle: GraphBundle, **_: Any
) -> AGCRNForecast:
    return AGCRNForecast(config, graph_bundle)
