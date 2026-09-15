from __future__ import annotations

import torch
from torch import Tensor, nn


def _valid_num_heads(hidden_dim: int, requested_heads: int) -> int:
    heads = max(1, int(requested_heads))
    while hidden_dim % heads != 0 and heads > 1:
        heads -= 1
    return heads


def _attention_entropy(weights: Tensor, max_rows: int = 4096) -> Tensor:
    """Compute a bounded, deterministic diagnostic without joining autograd."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")
    with torch.no_grad():
        flat = weights.detach().reshape(-1, weights.shape[-1])
        if flat.shape[0] > max_rows:
            step = max(1, flat.shape[0] // max_rows)
            flat = flat[::step][:max_rows]
        probs = flat.float().clamp_min(1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.special.entr(probs).sum(dim=-1).mean()


class SymmetricCrossFusion(nn.Module):
    """Node-wise cross-fusion for graph-enhanced fine/coarse histories."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        recent_len: int = 24,
        fusion_mode: str = "cross",
        disable_reverse_cross: bool = False,
        disable_macro_to_fine_cross: bool = False,
        diagnostics_level: str = "standard",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"cross", "add", "concat"}:
            raise ValueError("fusion_mode must be cross, add, or concat.")
        if recent_len <= 0:
            raise ValueError("recent_len must be positive.")
        heads = _valid_num_heads(int(hidden_dim), int(num_heads))
        self.hidden_dim = int(hidden_dim)
        self.recent_len = int(recent_len)
        self.fusion_mode = fusion_mode
        self.disable_reverse_cross = bool(disable_reverse_cross)
        self.disable_macro_to_fine_cross = bool(disable_macro_to_fine_cross)
        if diagnostics_level not in {"none", "minimal", "standard", "full"}:
            raise ValueError("diagnostics_level must be none, minimal, standard, or full.")
        self.diagnostics_level = diagnostics_level
        self.last_attention_diagnostics: dict[str, object] = {}
        self.macro_to_fine = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fine_to_coarse = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.fine_concat = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.coarse_concat = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fine_norm = nn.LayerNorm(self.hidden_dim)
        self.coarse_norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        h_fine: Tensor,
        h_coarse: Tensor,
        macro_prompt: Tensor | None,
    ) -> tuple[Tensor, Tensor, dict[str, object]]:
        if h_fine.shape != h_coarse.shape:
            raise ValueError(f"h_fine shape {tuple(h_fine.shape)} must match h_coarse {tuple(h_coarse.shape)}.")
        if h_fine.ndim != 4:
            raise ValueError(f"h_fine/h_coarse must be [B,L,N,D], got {tuple(h_fine.shape)}.")
        B, L, N, D = h_fine.shape
        if D != self.hidden_dim:
            raise ValueError(f"Expected D={self.hidden_dim}, got {D}.")
        if macro_prompt is not None:
            if macro_prompt.ndim != 4:
                raise ValueError(f"macro_prompt must be [B,N,P,D], got {tuple(macro_prompt.shape)}.")
            if macro_prompt.shape[0] != B or macro_prompt.shape[1] != N or macro_prompt.shape[3] != D:
                raise ValueError("macro_prompt shape must align with h_fine/h_coarse on B,N,D.")

        fine_seq = h_fine.permute(0, 2, 1, 3).reshape(B * N, L, D)
        coarse_seq = h_coarse.permute(0, 2, 1, 3).reshape(B * N, L, D)
        # Without Macro Prompt, preserve coarse->fine cross-attention by using
        # the graph-temporal coarse history directly as key/value memory.
        prompt_seq = (
            macro_prompt.reshape(B * N, macro_prompt.shape[2], D)
            if macro_prompt is not None
            else coarse_seq
        )
        need_attention_diagnostics = self.diagnostics_level in {"standard", "full"}
        if self.disable_macro_to_fine_cross:
            out_a = torch.zeros_like(h_fine)
            macro_attn = None
            out_a_seq = None
        else:
            out_a_seq, macro_attn = self.macro_to_fine(
                fine_seq,
                prompt_seq,
                prompt_seq,
                need_weights=need_attention_diagnostics,
                average_attn_weights=True,
            )
            out_a = out_a_seq.reshape(B, N, L, D).permute(0, 2, 1, 3).contiguous()

        if self.disable_reverse_cross:
            out_b = torch.zeros_like(h_coarse)
            fine_attn = None
        else:
            recent = min(self.recent_len, L)
            fine_recent = h_fine[:, -recent:, :, :].permute(0, 2, 1, 3).reshape(B * N, recent, D)
            out_b_seq, fine_attn = self.fine_to_coarse(
                coarse_seq,
                fine_recent,
                fine_recent,
                need_weights=need_attention_diagnostics,
                average_attn_weights=True,
            )
            out_b = out_b_seq.reshape(B, N, L, D).permute(0, 2, 1, 3).contiguous()

        out_a = self.dropout(out_a)
        out_b = self.dropout(out_b)
        if self.fusion_mode == "cross":
            gate = self.gate(torch.cat([out_a, out_b, h_fine], dim=-1))
            new_fine = self.fine_norm(h_fine if self.disable_macro_to_fine_cross else gate * out_a + h_fine)
            if self.disable_reverse_cross:
                new_coarse = self.coarse_norm(h_coarse)
            else:
                new_coarse = self.coarse_norm((1.0 - gate) * out_b + h_coarse)
        elif self.fusion_mode == "add":
            gate = torch.ones_like(h_fine)
            new_fine = self.fine_norm(h_fine if self.disable_macro_to_fine_cross else h_fine + out_a)
            new_coarse = self.coarse_norm(h_coarse if self.disable_reverse_cross else h_coarse + out_b)
        else:
            gate = torch.ones_like(h_fine)
            new_fine = self.fine_norm(
                h_fine
                if self.disable_macro_to_fine_cross
                else self.fine_concat(torch.cat([h_fine, out_a], dim=-1)) + h_fine
            )
            new_coarse = self.coarse_norm(
                h_coarse
                if self.disable_reverse_cross
                else self.coarse_concat(torch.cat([h_coarse, out_b], dim=-1)) + h_coarse
            )

        macro_attn_entropy = None
        fine_attn_entropy = None
        if need_attention_diagnostics:
            if macro_attn is not None:
                macro_attn_entropy = float(_attention_entropy(macro_attn).cpu().item())
            if fine_attn is not None:
                fine_attn_entropy = float(_attention_entropy(fine_attn).cpu().item())
        self.last_attention_diagnostics = {
            "attention_weights_requested": need_attention_diagnostics,
            "entropy_called": need_attention_diagnostics,
            "macro_attention_output_shape": list(out_a_seq.shape) if out_a_seq is not None else None,
            "macro_attention_output_dtype": str(out_a_seq.dtype) if out_a_seq is not None else None,
            "macro_attention_weights_shape": list(macro_attn.shape) if macro_attn is not None else None,
            "macro_attention_weights_dtype": str(macro_attn.dtype) if macro_attn is not None else None,
            "fine_attention_output_shape": list(out_b_seq.shape) if not self.disable_reverse_cross else None,
            "fine_attention_output_dtype": str(out_b_seq.dtype) if not self.disable_reverse_cross else None,
            "fine_attention_weights_shape": list(fine_attn.shape) if fine_attn is not None else None,
            "fine_attention_weights_dtype": str(fine_attn.dtype) if fine_attn is not None else None,
        }
        if self.diagnostics_level == "none":
            gate_mean = None
            gate_std = None
        else:
            gate_mean = float(gate.detach().float().mean().cpu().item())
            gate_std = float(gate.detach().float().std(unbiased=False).cpu().item())
        aux = {
            "macro_attn_entropy": macro_attn_entropy,
            "fine_attn_entropy": fine_attn_entropy,
            "fusion_gate_mean": gate_mean,
            "fusion_gate_std": gate_std,
            "attention_weights_requested": need_attention_diagnostics,
            "entropy_called": need_attention_diagnostics,
            "cross_fusion_uses_spatial_enhanced_features": True,
            "disable_reverse_cross": self.disable_reverse_cross,
            "disable_macro_to_fine_cross": self.disable_macro_to_fine_cross,
            "fusion_mode": self.fusion_mode,
            "coarse_to_fine_source": (
                "none"
                if self.disable_macro_to_fine_cross
                else "macro_prompt" if macro_prompt is not None else "coarse_history"
            ),
        }
        return new_fine, new_coarse, aux
