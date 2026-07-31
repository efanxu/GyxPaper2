from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ...graph import GraphBundle
from .adaptive_common import e3_c_graph_identity
from .common import runtime_support


def graph_propagate(x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nm,bcmt->bcnt", adjacency, x)


class DiffusionGraphConv(nn.Module):
    def __init__(
        self,
        channels: int,
        output_channels: int,
        *,
        support_count: int,
        order: int,
        dropout: float,
    ):
        super().__init__()
        self.order = int(order)
        self.support_count = int(support_count)
        total = (1 + self.support_count * self.order) * channels
        self.projection = nn.Conv2d(total, output_channels, kernel_size=(1, 1))
        self.dropout = float(dropout)
        self.last_branch_norms: list[float] = []

    def forward(
        self, x: torch.Tensor, supports: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        if len(supports) != self.support_count:
            raise ValueError(
                f"Expected {self.support_count} supports, got {len(supports)}"
            )
        terms = [x]
        norms: list[float] = []
        for support in supports:
            propagated = x
            for _ in range(self.order):
                propagated = graph_propagate(propagated, support)
                terms.append(propagated)
                norms.append(float(propagated.detach().float().norm().cpu()))
        self.last_branch_norms = norms
        output = self.projection(torch.cat(terms, dim=1))
        return F.dropout(output, self.dropout, training=self.training)


class GraphWaveNetForecast(nn.Module):
    def __init__(self, config: dict[str, Any], bundle: GraphBundle):
        super().__init__()
        self.num_nodes = int(config["num_nodes"])
        self.horizon = int(config["horizon"])
        self.blocks = int(config["blocks"])
        self.layers_per_block = int(config["layers_per_block"])
        self.diffusion_order = int(config["diffusion_order"])
        residual_channels = int(config["residual_channels"])
        dilation_channels = int(config["dilation_channels"])
        skip_channels = int(config["skip_channels"])
        end_channels = int(config["end_channels"])
        kernel_size = int(config["kernel_size"])
        dropout = float(config["dropout"])
        self.receptive_field = 1
        self.dilations: list[int] = []
        for _ in range(self.blocks):
            for layer in range(self.layers_per_block):
                dilation = 2**layer
                self.dilations.append(dilation)
                self.receptive_field += (kernel_size - 1) * dilation

        self.graph_identity = e3_c_graph_identity(bundle, "graph_wavenet")
        self.register_buffer(
            "P_forward", runtime_support(bundle, "P_forward"), persistent=False
        )
        self.register_buffer(
            "P_reverse", runtime_support(bundle, "P_reverse"), persistent=False
        )
        embedding_dim = int(config["adaptive_embedding_dim"])
        self.nodevec1 = nn.Parameter(torch.empty(self.num_nodes, embedding_dim))
        self.nodevec2 = nn.Parameter(torch.empty(embedding_dim, self.num_nodes))
        nn.init.xavier_uniform_(self.nodevec1)
        nn.init.xavier_uniform_(self.nodevec2)

        self.start_conv = nn.Conv2d(
            int(config["input_dim"]), residual_channels, kernel_size=(1, 1)
        )
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for dilation in self.dilations:
            temporal = (1, kernel_size)
            temporal_dilation = (1, dilation)
            self.filter_convs.append(
                nn.Conv2d(
                    residual_channels,
                    dilation_channels,
                    kernel_size=temporal,
                    dilation=temporal_dilation,
                )
            )
            self.gate_convs.append(
                nn.Conv2d(
                    residual_channels,
                    dilation_channels,
                    kernel_size=temporal,
                    dilation=temporal_dilation,
                )
            )
            self.residual_convs.append(
                nn.Conv2d(residual_channels, residual_channels, kernel_size=(1, 1))
            )
            self.skip_convs.append(
                nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1))
            )
            self.graph_convs.append(
                DiffusionGraphConv(
                    dilation_channels,
                    residual_channels,
                    support_count=3,
                    order=self.diffusion_order,
                    dropout=dropout,
                )
            )
            self.batch_norms.append(nn.BatchNorm2d(residual_channels))
        self.end_conv_1 = nn.Conv2d(
            skip_channels, end_channels, kernel_size=(1, 1)
        )
        self.end_conv_2 = nn.Conv2d(
            end_channels, self.horizon, kernel_size=(1, 1)
        )
        self.last_shape_trace: dict[str, Any] = {}

    def adaptive_adjacency(self) -> torch.Tensor:
        return torch.softmax(
            torch.relu(self.nodevec1 @ self.nodevec2), dim=1
        )

    def _supports(self, mode: str) -> tuple[torch.Tensor, ...]:
        adaptive = self.adaptive_adjacency()
        zero = torch.zeros_like(adaptive)
        if mode == "physical_adaptive":
            return self.P_forward, self.P_reverse, adaptive
        if mode == "physical_only":
            return self.P_forward, self.P_reverse, zero
        if mode == "adaptive_only":
            return zero, zero, adaptive
        raise ValueError(f"Unknown Graph WaveNet support mode: {mode}")

    def forward(
        self, x: torch.Tensor, *, support_mode: str = "physical_adaptive"
    ) -> torch.Tensor:
        if tuple(x.shape[1:]) != (144, self.num_nodes, 16):
            raise ValueError(
                f"Graph WaveNet requires (B,144,{self.num_nodes},16), got {tuple(x.shape)}"
            )
        value = x.permute(0, 3, 2, 1).contiguous()
        value = self.start_conv(value)
        supports = self._supports(support_mode)
        skip: torch.Tensor | None = None
        trace: list[dict[str, Any]] = []
        for index, dilation in enumerate(self.dilations):
            residual = value
            filtered = torch.tanh(self.filter_convs[index](residual))
            gated = torch.sigmoid(self.gate_convs[index](residual))
            temporal = filtered * gated
            contribution = self.skip_convs[index](temporal)
            skip = (
                contribution
                if skip is None
                else skip[..., -contribution.shape[-1] :] + contribution
            )
            value = self.graph_convs[index](temporal, supports)
            value = self.residual_convs[index](value)
            value = value + residual[..., -value.shape[-1] :]
            value = self.batch_norms[index](value)
            trace.append(
                {
                    "layer": index,
                    "dilation": dilation,
                    "temporal_width": int(temporal.shape[-1]),
                    "residual_width": int(value.shape[-1]),
                    "diffusion_branch_norms": list(
                        self.graph_convs[index].last_branch_norms
                    ),
                }
            )
        if skip is None:
            raise RuntimeError("Graph WaveNet produced no skip path.")
        output = self.end_conv_2(F.relu(self.end_conv_1(F.relu(skip))))
        self.last_shape_trace = {
            "receptive_field": self.receptive_field,
            "layers": trace,
            "final_temporal_width": int(output.shape[-1]),
            "forecast_anchor": "last_valid_causal_temporal_position",
            "support_mode": support_mode,
        }
        return output[..., -1].permute(0, 2, 1).contiguous()


def create_model(
    *, config: dict[str, Any], graph_bundle: GraphBundle, **_: Any
) -> GraphWaveNetForecast:
    return GraphWaveNetForecast(config, graph_bundle)
