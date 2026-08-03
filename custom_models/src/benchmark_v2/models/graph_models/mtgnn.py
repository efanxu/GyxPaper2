from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ...graph import GraphBundle
from .adaptive_common import e3_c_graph_identity


class DirectedGraphConstructor(nn.Module):
    def __init__(
        self, num_nodes: int, node_dim: int, top_k: int, tanh_alpha: float
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.top_k = int(top_k)
        self.tanh_alpha = float(tanh_alpha)
        self.emb1 = nn.Embedding(num_nodes, node_dim)
        self.emb2 = nn.Embedding(num_nodes, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)
        self.register_buffer(
            "node_index", torch.arange(num_nodes), persistent=False
        )
        self.last_topk_indices: torch.Tensor | None = None

    def forward(self) -> torch.Tensor:
        left = torch.tanh(
            self.tanh_alpha * self.lin1(self.emb1(self.node_index))
        )
        right = torch.tanh(
            self.tanh_alpha * self.lin2(self.emb2(self.node_index))
        )
        relation = left @ right.T - right @ left.T
        # Keep the directed top-k ordering while avoiding exact tanh
        # saturation.  Saturated constructor scores can make emb1/emb2
        # gradients identically zero for otherwise valid random initializers.
        # The detached row scale is a numerical normalization only; it does
        # not change the ranking used by the top-k mask.
        row_scale = relation.detach().abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        scores = torch.sigmoid(self.tanh_alpha * relation / row_scale)
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        indices = order[:, : self.top_k]
        mask = torch.zeros_like(scores)
        mask.scatter_(1, indices, 1.0)
        self.last_topk_indices = indices.detach()
        return scores * mask


class DilatedInception(nn.Module):
    KERNELS = (2, 3, 6, 7)

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        if out_channels % len(self.KERNELS):
            raise ValueError("Inception output channels must divide four kernels.")
        branch_channels = out_channels // len(self.KERNELS)
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=(1, kernel),
                    dilation=(1, dilation),
                )
                for kernel in self.KERNELS
            ]
        )
        self.last_branch_widths: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        self.last_branch_widths = [int(value.shape[-1]) for value in outputs]
        width = min(value.shape[-1] for value in outputs)
        return torch.cat([value[..., -width:] for value in outputs], dim=1)


class MixProp(nn.Module):
    def __init__(
        self,
        channels: int,
        output_channels: int,
        *,
        depth: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        self.depth = int(depth)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.projection = nn.Conv2d(
            (self.depth + 1) * channels, output_channels, kernel_size=(1, 1)
        )
        self.last_hop_norms: list[float] = []

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(
            adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype
        )
        support = adjacency + identity
        support = support / support.sum(dim=1, keepdim=True).clamp_min(1e-12)
        hidden = x
        terms = [x]
        norms: list[float] = []
        for _ in range(self.depth):
            propagated = torch.einsum("nm,bcmt->bcnt", support, hidden)
            hidden = self.alpha * x + (1.0 - self.alpha) * propagated
            terms.append(hidden)
            norms.append(float(hidden.detach().float().norm().cpu()))
        self.last_hop_norms = norms
        output = self.projection(torch.cat(terms, dim=1))
        return F.dropout(output, self.dropout, training=self.training)


class MTGNNForecast(nn.Module):
    def __init__(self, config: dict[str, Any], bundle: GraphBundle):
        super().__init__()
        self.num_nodes = int(config["num_nodes"])
        self.horizon = int(config["horizon"])
        self.layers = int(config["layers"])
        self.receptive_field = 1 + self.layers * (7 - 1)
        residual_channels = int(config["residual_channels"])
        conv_channels = int(config["conv_channels"])
        skip_channels = int(config["skip_channels"])
        end_channels = int(config["end_channels"])
        self.graph_identity = e3_c_graph_identity(bundle, "mtgnn")
        self.graph_constructor = DirectedGraphConstructor(
            self.num_nodes,
            int(config["node_dim"]),
            int(config["subgraph_size"]),
            float(config["tanhalpha"]),
        )
        self.start_conv = nn.Conv2d(
            int(config["input_dim"]), residual_channels, kernel_size=(1, 1)
        )
        self.skip0 = nn.Conv2d(
            int(config["input_dim"]),
            skip_channels,
            kernel_size=(1, int(config["seq_length"])),
        )
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.forward_props = nn.ModuleList()
        self.reverse_props = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        width = int(config["seq_length"])
        layer_widths: list[int] = []
        for _ in range(self.layers):
            self.filter_convs.append(
                DilatedInception(residual_channels, conv_channels, dilation=1)
            )
            self.gate_convs.append(
                DilatedInception(residual_channels, conv_channels, dilation=1)
            )
            self.forward_props.append(
                MixProp(
                    conv_channels,
                    residual_channels,
                    depth=int(config["gcn_depth"]),
                    alpha=float(config["propalpha"]),
                    dropout=float(config["dropout"]),
                )
            )
            self.reverse_props.append(
                MixProp(
                    conv_channels,
                    residual_channels,
                    depth=int(config["gcn_depth"]),
                    alpha=float(config["propalpha"]),
                    dropout=float(config["dropout"]),
                )
            )
            self.residual_convs.append(
                nn.Conv2d(residual_channels, residual_channels, kernel_size=(1, 1))
            )
            width -= 6
            layer_widths.append(width)
            self.skip_convs.append(
                nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, width))
            )
            self.layer_norms.append(
                nn.LayerNorm(
                    (residual_channels, self.num_nodes, width),
                    elementwise_affine=bool(config["layer_norm_affine"]),
                )
            )
        self.layer_widths = layer_widths
        self.skipE = nn.Conv2d(
            residual_channels, skip_channels, kernel_size=(1, width)
        )
        self.end_conv_1 = nn.Conv2d(
            skip_channels, end_channels, kernel_size=(1, 1)
        )
        self.end_conv_2 = nn.Conv2d(
            end_channels, self.horizon, kernel_size=(1, 1)
        )
        self.last_shape_trace: dict[str, Any] = {}

    def learned_adjacency(self) -> torch.Tensor:
        return self.graph_constructor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != (144, self.num_nodes, 16):
            raise ValueError(
                f"MTGNN requires (B,144,{self.num_nodes},16), got {tuple(x.shape)}"
            )
        raw = x.permute(0, 3, 2, 1).contiguous()
        value = self.start_conv(raw)
        skip = self.skip0(F.dropout(raw, 0.3, training=self.training))
        adjacency = self.learned_adjacency()
        trace: list[dict[str, Any]] = []
        for index in range(self.layers):
            residual = value
            filtered = torch.tanh(self.filter_convs[index](residual))
            gated = torch.sigmoid(self.gate_convs[index](residual))
            temporal = filtered * gated
            skip = skip + self.skip_convs[index](temporal)
            forward = self.forward_props[index](temporal, adjacency)
            reverse = self.reverse_props[index](temporal, adjacency.T)
            value = forward + reverse
            value = value + self.residual_convs[index](
                residual[..., -value.shape[-1] :]
            )
            value = self.layer_norms[index](value)
            trace.append(
                {
                    "layer": index,
                    "filter_branch_widths": list(
                        self.filter_convs[index].last_branch_widths
                    ),
                    "gate_branch_widths": list(
                        self.gate_convs[index].last_branch_widths
                    ),
                    "temporal_width": int(temporal.shape[-1]),
                    "forward_hop_norms": list(
                        self.forward_props[index].last_hop_norms
                    ),
                    "reverse_hop_norms": list(
                        self.reverse_props[index].last_hop_norms
                    ),
                }
            )
        skip = skip + self.skipE(value)
        output = self.end_conv_2(F.relu(self.end_conv_1(F.relu(skip))))
        self.last_shape_trace = {
            "receptive_field": self.receptive_field,
            "skip0_width": 1,
            "layers": trace,
            "skipE_width": int(self.skipE(value).shape[-1]),
            "final_temporal_width": int(output.shape[-1]),
        }
        return output[..., 0].permute(0, 2, 1).contiguous()


def create_model(
    *, config: dict[str, Any], graph_bundle: GraphBundle, **_: Any
) -> MTGNNForecast:
    return MTGNNForecast(config, graph_bundle)
