from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def row_normalize(A: Tensor, eps: float = 1e-8) -> Tensor:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square [N,N], got {tuple(A.shape)}.")
    return A / A.sum(dim=-1, keepdim=True).clamp_min(eps)


class AdaptiveGraphBuilder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        node_embed_dim: int = 10,
        temperature: float = 1.0,
        fusion_mode: str = "multiply",
        use_prior: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if fusion_mode != "multiply":
            raise ValueError("Step 2 AdaptiveGraphBuilder supports the protocol default multiply mode.")
        self.num_nodes = int(num_nodes)
        self.node_embed_dim = int(node_embed_dim)
        self.temperature = float(temperature)
        self.fusion_mode = fusion_mode
        self.use_prior = bool(use_prior)
        self.eps = float(eps)
        self.E1 = nn.Parameter(torch.empty(self.num_nodes, self.node_embed_dim))
        self.E2 = nn.Parameter(torch.empty(self.num_nodes, self.node_embed_dim))
        nn.init.xavier_uniform_(self.E1)
        nn.init.xavier_uniform_(self.E2)

    def forward(self, A_prior: Tensor) -> Tensor:
        if A_prior.shape != (self.num_nodes, self.num_nodes):
            raise ValueError(f"A_prior must be [{self.num_nodes},{self.num_nodes}], got {tuple(A_prior.shape)}.")
        prior = A_prior.to(device=self.E1.device, dtype=self.E1.dtype)
        logits = torch.relu(self.E1 @ self.E2.T) / self.temperature
        A_adp = torch.softmax(logits, dim=-1)
        A_final = prior * A_adp if self.use_prior else A_adp
        A_final = A_final / A_final.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return A_final


class FixedPriorGraphBuilder(nn.Module):
    """Non-learnable graph path used by the w/o Adaptive Graph ablation."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)
        self.use_prior = True

    def forward(self, A_prior: Tensor) -> Tensor:
        return row_normalize(A_prior, eps=self.eps)


class SimpleGraphConv(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0, beta_init: float = 0.05) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.beta_graph = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, h: Tensor, A: Tensor) -> Tensor:
        if h.ndim != 4:
            raise ValueError(f"h must be [B,L,N,D], got {tuple(h.shape)}.")
        if A.shape != (h.shape[2], h.shape[2]):
            raise ValueError(f"A shape {tuple(A.shape)} must match N={h.shape[2]}.")
        A = A.to(device=h.device, dtype=h.dtype)
        h_msg = torch.einsum("ij,btjd->btid", A, h)
        graph_delta = self.linear(torch.cat([h, h_msg], dim=-1))
        out = h + self.beta_graph * graph_delta
        return self.dropout(out)


class PriorConstrainedDiffusionGraphConv(nn.Module):
    """Prior-constrained bidirectional K-hop diffusion graph convolution."""

    def __init__(
        self,
        hidden_dim: int,
        diffusion_order: int = 2,
        dropout: float = 0.1,
        beta_init: float = 0.05,
        fusion_mode: str = "concat_projection",
        use_bidirectional: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if diffusion_order <= 0:
            raise ValueError("diffusion_order must be positive.")
        if diffusion_order > 3:
            raise ValueError("diffusion_order > 3 is disabled by protocol to avoid over-smoothing.")
        if fusion_mode != "concat_projection":
            raise ValueError("Only concat_projection fusion is supported for diffusion graph conv.")
        self.hidden_dim = int(hidden_dim)
        self.diffusion_order = int(diffusion_order)
        self.fusion_mode = fusion_mode
        self.use_bidirectional = bool(use_bidirectional)
        self.eps = float(eps)
        num_streams = 1 + self.diffusion_order
        if self.use_bidirectional:
            num_streams += self.diffusion_order
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim * num_streams, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.beta_graph = nn.Parameter(torch.tensor(float(beta_init)))

    @staticmethod
    def _node_cosine_similarity(x: Tensor) -> Tensor:
        # Average pairwise node cosine similarity over batch and time.
        if x.shape[2] <= 1:
            return torch.ones((), device=x.device, dtype=x.dtype)
        normalized = F.normalize(x, dim=-1, eps=1e-8)
        sim = torch.einsum("btid,btjd->btij", normalized, normalized)
        N = x.shape[2]
        eye = torch.eye(N, device=x.device, dtype=torch.bool)
        return sim[..., ~eye].mean()

    def forward(self, h: Tensor, A: Tensor, return_aux: bool = False):
        if h.ndim != 4:
            raise ValueError(f"h must be [B,L,N,D], got {tuple(h.shape)}.")
        if A.shape != (h.shape[2], h.shape[2]):
            raise ValueError(f"A shape {tuple(A.shape)} must match N={h.shape[2]}.")
        A = A.to(device=h.device, dtype=h.dtype)
        A_forward = row_normalize(A, eps=self.eps)
        A_backward = row_normalize(A.transpose(-1, -2), eps=self.eps)

        features = [h]
        forward_hops: list[Tensor] = []
        backward_hops: list[Tensor] = []
        h_forward = h
        h_backward = h
        for _ in range(self.diffusion_order):
            h_forward = torch.einsum("ij,btjd->btid", A_forward, h_forward)
            forward_hops.append(h_forward)
            features.append(h_forward)
        if self.use_bidirectional:
            for _ in range(self.diffusion_order):
                h_backward = torch.einsum("ij,btjd->btid", A_backward, h_backward)
                backward_hops.append(h_backward)
                features.append(h_backward)

        graph_delta = self.projection(torch.cat(features, dim=-1))
        out = self.dropout(h + self.beta_graph * graph_delta)
        if not return_aux:
            return out

        hop1 = forward_hops[0] if forward_hops else h
        hop2 = forward_hops[min(1, len(forward_hops) - 1)] if forward_hops else h
        input_norm = h.norm(dim=-1).mean().clamp_min(self.eps)
        aux = {
            "graph_operator": "bidirectional_diffusion",
            "diffusion_order": torch.tensor(self.diffusion_order, device=h.device),
            "diffusion_use_bidirectional": torch.tensor(self.use_bidirectional, device=h.device),
            "diffusion_forward_row_sum_min": A_forward.sum(dim=-1).min().detach(),
            "diffusion_forward_row_sum_max": A_forward.sum(dim=-1).max().detach(),
            "diffusion_backward_row_sum_min": A_backward.sum(dim=-1).min().detach(),
            "diffusion_backward_row_sum_max": A_backward.sum(dim=-1).max().detach(),
            "input_node_cosine_similarity": self._node_cosine_similarity(h).detach(),
            "hop1_node_cosine_similarity": self._node_cosine_similarity(hop1).detach(),
            "hop2_node_cosine_similarity": self._node_cosine_similarity(hop2).detach(),
            "output_node_cosine_similarity": self._node_cosine_similarity(out).detach(),
            "graph_delta_norm": graph_delta.norm(dim=-1).mean().detach(),
            "graph_delta_ratio": (graph_delta.norm(dim=-1).mean() / input_norm).detach(),
            "beta_graph": self.beta_graph.detach(),
        }
        for idx in range(self.diffusion_order):
            aux[f"forward_hop{idx + 1}_norm"] = forward_hops[idx].norm(dim=-1).mean().detach()
            if self.use_bidirectional:
                aux[f"backward_hop{idx + 1}_norm"] = backward_hops[idx].norm(dim=-1).mean().detach()
        return out, aux
