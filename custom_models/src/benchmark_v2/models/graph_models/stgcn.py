from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from ...graph import GraphBundle
from .common import graph_identity_dict, runtime_support


def chebyshev_basis(support: torch.Tensor, order: int) -> torch.Tensor:
    if support.ndim != 2 or support.shape[0] != support.shape[1]:
        raise ValueError("Chebyshev support must be square.")
    if order < 1:
        raise ValueError("Chebyshev order must be positive.")
    values = [torch.eye(support.shape[0], dtype=support.dtype, device=support.device)]
    if order > 1:
        values.append(support)
    for _ in range(2, order):
        values.append(2.0 * support @ values[-1] - values[-2])
    return torch.stack(values, dim=0)


class TemporalGLUConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.kernel_size = int(kernel_size)
        self.convolution = nn.Conv2d(
            self.input_dim,
            2 * self.output_dim,
            kernel_size=(self.kernel_size, 1),
        )
        self.residual_projection = (
            nn.Conv2d(self.input_dim, self.output_dim, kernel_size=1)
            if self.input_dim != self.output_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.permute(0, 3, 1, 2).contiguous()
        residual = self.residual_projection(value)[
            :, :, self.kernel_size - 1 :, :
        ]
        filter_value, gate_value = self.convolution(value).chunk(2, dim=1)
        output = (filter_value + residual) * torch.sigmoid(gate_value)
        return output.permute(0, 2, 3, 1).contiguous()


class ChebyshevGraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, order: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.order = int(order)
        self.weight = nn.Parameter(
            torch.empty(self.order, self.input_dim, self.output_dim)
        )
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self, x: torch.Tensor, basis: torch.Tensor
    ) -> torch.Tensor:
        if tuple(basis.shape[:1]) != (self.order,):
            raise ValueError(
                f"Expected {self.order} Chebyshev terms, got {tuple(basis.shape)}."
            )
        aggregated = torch.einsum("knm,btmc->bktnc", basis, x)
        return (
            torch.einsum("bktnc,kco->btno", aggregated, self.weight)
            + self.bias
        )


class STConvBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        temporal_dim: int,
        graph_dim: int,
        output_dim: int,
        *,
        temporal_kernel: int,
        chebyshev_order: int,
        dropout: float,
    ):
        super().__init__()
        self.temporal1 = TemporalGLUConv(
            input_dim, temporal_dim, temporal_kernel
        )
        self.graph = ChebyshevGraphConv(
            temporal_dim, graph_dim, chebyshev_order
        )
        self.temporal2 = TemporalGLUConv(
            graph_dim, output_dim, temporal_kernel
        )
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self, x: torch.Tensor, basis: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, list[int]]]:
        temporal1 = self.temporal1(x)
        graph = torch.relu(self.graph(temporal1, basis))
        temporal2 = self.temporal2(graph)
        output = self.dropout(self.layer_norm(temporal2))
        return output, {
            "temporal1": list(temporal1.shape),
            "graph": list(graph.shape),
            "temporal2": list(temporal2.shape),
        }


class STGCNForecast(nn.Module):
    """Classic IJCAI-2018 T-G-T blocks with Chebyshev graph convolution."""

    def __init__(
        self,
        *,
        bundle: GraphBundle,
        input_dim: int = 16,
        temporal_channels: int = 64,
        graph_channels: int = 16,
        output_channels: int = 64,
        num_blocks: int = 2,
        temporal_kernel: int = 3,
        chebyshev_order: int = 3,
        lookback: int = 144,
        horizon: int = 10,
        output_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        if int(num_blocks) != 2:
            raise ValueError("E3-B freezes STGCN to two ST-Conv blocks.")
        self.input_dim = int(input_dim)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.temporal_kernel = int(temporal_kernel)
        self.chebyshev_order = int(chebyshev_order)
        self.remaining_temporal_length = self.lookback - (
            2 * int(num_blocks) * (self.temporal_kernel - 1)
        )
        l_tilde = runtime_support(bundle, "L_tilde")
        self.register_buffer("L_tilde", l_tilde, persistent=False)
        self.register_buffer(
            "chebyshev_support",
            chebyshev_basis(l_tilde, self.chebyshev_order),
            persistent=False,
        )
        self.block1 = STConvBlock(
            self.input_dim,
            temporal_channels,
            graph_channels,
            output_channels,
            temporal_kernel=self.temporal_kernel,
            chebyshev_order=self.chebyshev_order,
            dropout=dropout,
        )
        self.block2 = STConvBlock(
            output_channels,
            temporal_channels,
            graph_channels,
            output_channels,
            temporal_kernel=self.temporal_kernel,
            chebyshev_order=self.chebyshev_order,
            dropout=dropout,
        )
        self.temporal_collapse = nn.Conv2d(
            output_channels,
            output_hidden,
            kernel_size=(self.remaining_temporal_length, 1),
        )
        self.output_norm = nn.LayerNorm(output_hidden)
        self.horizon_projection = nn.Linear(output_hidden, self.horizon)
        self.graph_identity = graph_identity_dict(bundle, ("L_tilde",))
        self.last_shape_trace: dict[str, Any] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != (
            self.lookback,
            self.L_tilde.shape[0],
            self.input_dim,
        ):
            raise ValueError(
                "STGCNForecast input mismatch: "
                f"expected (B,{self.lookback},{self.L_tilde.shape[0]},"
                f"{self.input_dim}), got {tuple(x.shape)}."
            )
        block1, trace1 = self.block1(x, self.chebyshev_support)
        block2, trace2 = self.block2(block1, self.chebyshev_support)
        collapsed = self.temporal_collapse(
            block2.permute(0, 3, 1, 2).contiguous()
        )
        if collapsed.shape[2] != 1:
            raise RuntimeError(
                "STGCN temporal collapse must produce exactly one time step."
            )
        shared = collapsed.squeeze(2).permute(0, 2, 1).contiguous()
        shared = torch.relu(self.output_norm(shared))
        prediction = self.horizon_projection(shared)
        self.last_shape_trace = {
            "input": list(x.shape),
            "block1": trace1,
            "block2": trace2,
            "remaining_temporal_length": int(block2.shape[1]),
            "temporal_collapse": list(collapsed.shape),
            "prediction": list(prediction.shape),
        }
        return prediction


def create_model(
    config: Mapping[str, Any] | None = None,
    protocol: Any = None,
    *,
    graph_bundle: GraphBundle,
) -> STGCNForecast:
    values = dict(config or {})
    return STGCNForecast(
        bundle=graph_bundle,
        input_dim=int(values.get("input_dim", 16)),
        temporal_channels=int(values.get("temporal_channels", 64)),
        graph_channels=int(values.get("graph_channels", 16)),
        output_channels=int(values.get("output_channels", 64)),
        num_blocks=int(values.get("num_st_blocks", 2)),
        temporal_kernel=int(values.get("temporal_kernel_size", 3)),
        chebyshev_order=int(values.get("graph_chebyshev_order", 3)),
        lookback=int(values.get("lookback", 144)),
        horizon=int(values.get("horizon", 10)),
        output_hidden=int(values.get("output_hidden", 128)),
        dropout=float(values.get("dropout", 0.3)),
    )
