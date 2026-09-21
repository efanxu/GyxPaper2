from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .graph_layers import PriorConstrainedDiffusionGraphConv, SimpleGraphConv


class CausalConv1dBlock(nn.Module):
    """Shared causal temporal conv over each turbine sequence."""

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if dilation <= 0:
            raise ValueError("dilation must be positive.")
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.left_padding = (self.kernel_size - 1) * self.dilation
        self.conv = nn.Conv1d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
        )
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be [B,L,N,D], got {tuple(x.shape)}.")
        B, L, N, D = x.shape
        if D != self.hidden_dim:
            raise ValueError(f"Expected D={self.hidden_dim}, got {D}.")
        seq = x.permute(0, 2, 3, 1).reshape(B * N, D, L)
        seq = F.pad(seq, (self.left_padding, 0))
        out = self.conv(seq)
        out = self.dropout(self.activation(out))
        out = out.reshape(B, N, D, L).permute(0, 3, 1, 2).contiguous()
        return self.norm(out + x)


class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        num_heads = max(1, int(num_heads))
        while hidden_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be [B,L,N,D], got {tuple(x.shape)}.")
        B, L, N, D = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(B * N, L, D)
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        out, _ = self.attn(seq, seq, seq, attn_mask=causal_mask, need_weights=False)
        out = self.dropout(out).reshape(B, N, L, D).permute(0, 2, 1, 3).contiguous()
        return self.norm(out + x)


class DilatedTCNEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.0,
        use_temporal_attention: bool = False,
        temporal_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if not dilations:
            raise ValueError("dilations must not be empty.")
        self.blocks = nn.ModuleList(
            [
                CausalConv1dBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.attention = (
            TemporalSelfAttention(hidden_dim, num_heads=temporal_attention_heads, dropout=dropout)
            if use_temporal_attention
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for block in self.blocks:
            h = block(h)
        if self.attention is not None:
            h = self.attention(h)
        return h


class GraphTemporalBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        dilations: list[int] | tuple[int, ...],
        dropout: float = 0.0,
        use_graph: bool = True,
        graph_operator: str = "simple",
        diffusion_order: int = 2,
        diffusion_use_bidirectional: bool = True,
        diffusion_direction: str = "bidirectional",
        diffusion_projection_mode: str = "shared_output_dim",
        diffusion_beta_init: float = 0.05,
        use_temporal_attention: bool = False,
        temporal_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.use_graph = bool(use_graph)
        self.graph_operator = graph_operator
        if self.use_graph:
            if graph_operator == "simple":
                self.graph_conv = SimpleGraphConv(hidden_dim, dropout=dropout, beta_init=diffusion_beta_init)
            elif graph_operator == "bidirectional_diffusion":
                self.graph_conv = PriorConstrainedDiffusionGraphConv(
                    hidden_dim,
                    diffusion_order=diffusion_order,
                    dropout=dropout,
                    beta_init=diffusion_beta_init,
                    use_bidirectional=diffusion_use_bidirectional,
                    direction=diffusion_direction,
                    projection_mode=diffusion_projection_mode,
                )
            else:
                raise ValueError(f"Unsupported graph_operator: {graph_operator}")
        else:
            self.graph_conv = None
        self.causal_tcn = DilatedTCNEncoder(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            use_temporal_attention=use_temporal_attention,
            temporal_attention_heads=temporal_attention_heads,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, A: Tensor | None = None, return_aux: bool = False):
        h = x
        graph_aux = {}
        if self.graph_conv is not None:
            if A is None:
                raise ValueError("A is required when use_graph=True.")
            if isinstance(self.graph_conv, PriorConstrainedDiffusionGraphConv):
                h, graph_aux = self.graph_conv(h, A, return_aux=True)
            else:
                h = self.graph_conv(h, A)
                graph_aux = {
                    "graph_operator": "simple",
                    "beta_graph": self.graph_conv.beta_graph.detach(),
                    "graph_delta_norm": (h - x).norm(dim=-1).mean().detach(),
                    "graph_delta_ratio": ((h - x).norm(dim=-1).mean() / x.norm(dim=-1).mean().clamp_min(1e-8)).detach(),
                }
        h = self.causal_tcn(h)
        out = self.layer_norm(h + x)
        if return_aux:
            return out, graph_aux
        return out


class FineMicroGraphTemporalEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2),
        dropout: float = 0.0,
        use_graph: bool = True,
        graph_operator: str = "simple",
        diffusion_order: int = 2,
        diffusion_use_bidirectional: bool = True,
        diffusion_direction: str = "bidirectional",
        diffusion_projection_mode: str = "shared_output_dim",
        diffusion_beta_init: float = 0.05,
        use_temporal_attention: bool = False,
        temporal_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.block = GraphTemporalBlock(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            use_graph=use_graph,
            graph_operator=graph_operator,
            diffusion_order=diffusion_order,
            diffusion_use_bidirectional=diffusion_use_bidirectional,
            diffusion_direction=diffusion_direction,
            diffusion_projection_mode=diffusion_projection_mode,
            diffusion_beta_init=diffusion_beta_init,
            use_temporal_attention=use_temporal_attention,
            temporal_attention_heads=temporal_attention_heads,
        )

    def forward(self, x_fine: Tensor, A_micro: Tensor | None = None, return_aux: bool = False):
        return self.block(x_fine, A_micro, return_aux=return_aux)


class CoarseMacroGraphTemporalEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 5,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.0,
        use_graph: bool = True,
        graph_operator: str = "simple",
        diffusion_order: int = 2,
        diffusion_use_bidirectional: bool = True,
        diffusion_direction: str = "bidirectional",
        diffusion_projection_mode: str = "shared_output_dim",
        diffusion_beta_init: float = 0.05,
        use_temporal_attention: bool = False,
        temporal_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.block = GraphTemporalBlock(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            use_graph=use_graph,
            graph_operator=graph_operator,
            diffusion_order=diffusion_order,
            diffusion_use_bidirectional=diffusion_use_bidirectional,
            diffusion_direction=diffusion_direction,
            diffusion_projection_mode=diffusion_projection_mode,
            diffusion_beta_init=diffusion_beta_init,
            use_temporal_attention=use_temporal_attention,
            temporal_attention_heads=temporal_attention_heads,
        )

    def forward(self, x_coarse: Tensor, A_macro: Tensor | None = None, return_aux: bool = False):
        return self.block(x_coarse, A_macro, return_aux=return_aux)


class DualGraphTemporalEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        fine_kernel_size: int = 3,
        coarse_kernel_size: int = 5,
        fine_dilations: list[int] | tuple[int, ...] = (1, 2),
        coarse_dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.0,
        use_graph: bool = True,
        graph_operator: str = "simple",
        diffusion_order_micro: int = 2,
        diffusion_order_macro: int = 2,
        diffusion_use_bidirectional: bool = True,
        diffusion_direction: str = "bidirectional",
        diffusion_projection_mode: str = "shared_output_dim",
        diffusion_beta_init: float = 0.05,
        use_temporal_attention: bool = False,
        temporal_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.fine_encoder = FineMicroGraphTemporalEncoder(
            hidden_dim=hidden_dim,
            kernel_size=fine_kernel_size,
            dilations=fine_dilations,
            dropout=dropout,
            use_graph=use_graph,
            graph_operator=graph_operator,
            diffusion_order=diffusion_order_micro,
            diffusion_use_bidirectional=diffusion_use_bidirectional,
            diffusion_direction=diffusion_direction,
            diffusion_projection_mode=diffusion_projection_mode,
            diffusion_beta_init=diffusion_beta_init,
            use_temporal_attention=use_temporal_attention,
            temporal_attention_heads=temporal_attention_heads,
        )
        self.coarse_encoder = CoarseMacroGraphTemporalEncoder(
            hidden_dim=hidden_dim,
            kernel_size=coarse_kernel_size,
            dilations=coarse_dilations,
            dropout=dropout,
            use_graph=use_graph,
            graph_operator=graph_operator,
            diffusion_order=diffusion_order_macro,
            diffusion_use_bidirectional=diffusion_use_bidirectional,
            diffusion_direction=diffusion_direction,
            diffusion_projection_mode=diffusion_projection_mode,
            diffusion_beta_init=diffusion_beta_init,
            use_temporal_attention=use_temporal_attention,
            temporal_attention_heads=temporal_attention_heads,
        )

    def forward(self, x_fine: Tensor, x_coarse: Tensor, A_micro: Tensor | None, A_macro: Tensor | None) -> tuple[Tensor, Tensor]:
        h_fine = self.fine_encoder(x_fine, A_micro)
        h_coarse = self.coarse_encoder(x_coarse, A_macro)
        return h_fine, h_coarse
