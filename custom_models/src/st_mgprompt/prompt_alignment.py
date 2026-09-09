from __future__ import annotations

import torch
from torch import Tensor, nn


class MacroTrendPrompt(nn.Module):
    """Generate macro-trend prompts from graph-enhanced coarse history."""

    def __init__(
        self,
        hidden_dim: int,
        prompt_len: int = 4,
        dropout: float = 0.0,
        pooling: str = "attention",
        diagnostics_level: str = "standard",
    ) -> None:
        super().__init__()
        if prompt_len <= 0:
            raise ValueError("prompt_len must be positive.")
        if pooling not in {"attention", "mean"}:
            raise ValueError("pooling must be attention or mean.")
        self.hidden_dim = int(hidden_dim)
        self.prompt_len = int(prompt_len)
        self.pooling = pooling
        self.diagnostics_level = diagnostics_level
        self.score = nn.Linear(self.hidden_dim, 1)
        self.project = nn.Linear(self.hidden_dim, self.prompt_len * self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_coarse: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if h_coarse.ndim != 4:
            raise ValueError(f"h_coarse must be [B,L,N,D], got {tuple(h_coarse.shape)}.")
        B, L, N, D = h_coarse.shape
        if D != self.hidden_dim:
            raise ValueError(f"Expected D={self.hidden_dim}, got {D}.")

        if self.pooling == "attention":
            logits = self.score(h_coarse).squeeze(-1).permute(0, 2, 1).contiguous()
            weights = torch.softmax(logits, dim=-1)
            pooled = torch.einsum("bnl,blnd->bnd", weights, h_coarse)
        else:
            pooled = h_coarse.mean(dim=1)

        prompt = self.project(pooled).reshape(B, N, self.prompt_len, D)
        prompt = self.dropout(self.norm(prompt))
        entropy = None
        if self.pooling == "attention" and self.diagnostics_level in {"standard", "full"}:
            with torch.no_grad():
                probs = weights.detach().reshape(-1, weights.shape[-1])
                if probs.shape[0] > 4096:
                    step = max(1, probs.shape[0] // 4096)
                    probs = probs[::step][:4096]
                probs = probs.float().clamp_min(1e-8)
                entropy = float(torch.special.entr(probs).sum(dim=-1).mean().cpu().item())
        prompt_norm_mean = None
        if self.diagnostics_level != "none":
            with torch.no_grad():
                prompt_norm_mean = float(prompt.detach().float().norm(dim=-1).mean().cpu().item())
        aux = {
            "macro_prompt_attn_entropy": entropy,
            "macro_prompt_norm_mean": prompt_norm_mean,
            "macro_prompt_pooling": self.pooling,
            "macro_prompt_attention_enabled": self.pooling == "attention",
        }
        return prompt, aux


class STPromptEmbedding(nn.Module):
    """Future-step spatio-temporal prompt used by direct multi-output heads."""

    def __init__(
        self,
        num_nodes: int,
        max_pred_len: int,
        hidden_dim: int,
        num_granularities: int = 2,
        dropout: float = 0.0,
        mode: str = "full",
        use_node_identity: bool = True,
    ) -> None:
        super().__init__()
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")
        if max_pred_len <= 0:
            raise ValueError("max_pred_len must be positive.")
        if mode not in {"full", "horizon_only"}:
            raise ValueError("mode must be full or horizon_only.")
        self.num_nodes = int(num_nodes)
        self.max_pred_len = int(max_pred_len)
        self.hidden_dim = int(hidden_dim)
        self.mode = mode
        if not isinstance(use_node_identity, bool):
            raise ValueError("use_node_identity must be a boolean.")
        self.use_node_identity = use_node_identity
        self.node_embedding = (
            nn.Embedding(self.num_nodes, self.hidden_dim) if self.use_node_identity else None
        )
        self.future_step_embedding = nn.Embedding(self.max_pred_len, self.hidden_dim)
        self.granularity_embedding = nn.Embedding(int(num_granularities), self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        num_nodes: int | None = None,
        horizon: int | None = None,
        granularity_index: int = 0,
    ) -> Tensor:
        N = int(num_nodes or self.num_nodes)
        H = int(horizon or self.max_pred_len)
        if N > self.num_nodes:
            raise ValueError(f"Requested N={N} exceeds configured num_nodes={self.num_nodes}.")
        if H > self.max_pred_len:
            raise ValueError(f"Requested horizon={H} exceeds max_pred_len={self.max_pred_len}.")
        device = self.future_step_embedding.weight.device
        step_ids = torch.arange(H, device=device)
        step = self.future_step_embedding(step_ids).view(1, H, 1, self.hidden_dim)
        if self.mode == "horizon_only":
            # Legacy compatibility only. Formal node-identity ablations use
            # mode="full" with use_node_identity=False so granularity is retained.
            prompt = step.expand(1, H, N, self.hidden_dim)
        else:
            granularity_id = torch.tensor(int(granularity_index), device=device)
            granularity = self.granularity_embedding(granularity_id).view(1, 1, 1, self.hidden_dim)
            if self.use_node_identity:
                node_ids = torch.arange(N, device=device)
                node = self.node_embedding(node_ids).view(1, 1, N, self.hidden_dim)
                prompt = self.dropout(self.norm(node + step + granularity))
                return prompt
            # Expand after dropout so node-identity ablation prompts are exactly
            # shared across nodes even when the module is in training mode.
            prompt = self.dropout(self.norm(step + granularity))
            return prompt.expand(1, H, N, self.hidden_dim)
        prompt = self.norm(prompt)
        return self.dropout(prompt)
