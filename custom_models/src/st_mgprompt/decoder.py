from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class STPromptDirectDecoder(nn.Module):
    """Direct multi-output decoder driven by future ST prompt queries."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0, diagnostics_level: str = "standard") -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.diagnostics_level = diagnostics_level
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_dim),
        )
        self.predict_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.metadata = {
            "decoder_type": "STPromptDirectDecoder",
            "decoder_input_strategy": "direct_multi_output_prompt_query",
            "teacher_forcing": False,
            "autoregressive": False,
            "uses_history_only": True,
            "future_observed_features_used": False,
        }

    def forward(
        self,
        z_fine: Tensor,
        z_coarse: Tensor,
        st_prompt: Tensor,
        macro_prompt: Tensor | None = None,
        return_aux: bool = False,
    ):
        if z_fine.shape != z_coarse.shape:
            raise ValueError(f"z_fine shape {tuple(z_fine.shape)} must match z_coarse {tuple(z_coarse.shape)}.")
        if z_fine.ndim != 4:
            raise ValueError(f"z_fine/z_coarse must be [B,L,N,D], got {tuple(z_fine.shape)}.")
        if st_prompt.ndim != 4:
            raise ValueError(f"st_prompt must be [1,H,N,D], got {tuple(st_prompt.shape)}.")
        B, _, N, D = z_fine.shape
        if D != self.hidden_dim:
            raise ValueError(f"Expected D={self.hidden_dim}, got {D}.")
        if st_prompt.shape[0] != 1 or st_prompt.shape[2] != N or st_prompt.shape[3] != D:
            raise ValueError("st_prompt must align with z_fine/z_coarse as [1,H,N,D].")

        fine_last = z_fine[:, -1, :, :]
        coarse_last = z_coarse[:, -1, :, :]
        if macro_prompt is None:
            macro_summary = torch.zeros_like(fine_last)
        else:
            if macro_prompt.ndim != 4:
                raise ValueError(f"macro_prompt must be [B,N,P,D], got {tuple(macro_prompt.shape)}.")
            if macro_prompt.shape[0] != B or macro_prompt.shape[1] != N or macro_prompt.shape[3] != D:
                raise ValueError("macro_prompt must align with z_fine/z_coarse on B,N,D.")
            macro_summary = macro_prompt.mean(dim=2)

        context = self.fusion_layer(torch.cat([fine_last, coarse_last, macro_summary], dim=-1))
        context_future = context[:, None, :, :] + st_prompt
        pred = self.predict_head(context_future).squeeze(-1).contiguous()
        if not return_aux:
            return pred
        diagnostic_scalars = {
            "decoder_context_norm_mean": None,
            "decoder_macro_summary_norm_mean": None,
            "decoder_st_prompt_norm_mean": None,
            "decoder_context_future_norm_mean": None,
        }
        if self.diagnostics_level != "none":
            with torch.no_grad():
                diagnostic_scalars = {
                "decoder_context_norm_mean": float(context.detach().float().norm(dim=-1).mean().cpu().item()),
                "decoder_macro_summary_norm_mean": float(macro_summary.detach().float().norm(dim=-1).mean().cpu().item()),
                "decoder_st_prompt_norm_mean": float(st_prompt.detach().float().norm(dim=-1).mean().cpu().item()),
                "decoder_context_future_norm_mean": float(context_future.detach().float().norm(dim=-1).mean().cpu().item()),
                }
        aux = {
            "decoder_type": self.metadata["decoder_type"],
            "decoder_input_strategy": self.metadata["decoder_input_strategy"],
            "teacher_forcing": False,
            "autoregressive": False,
            "decoder_uses_history_only": True,
            "future_observed_features_used": False,
            **diagnostic_scalars,
        }
        return pred, aux


class HorizonDirectDecoder(nn.Module):
    """Ordinary history-to-horizon head with no ST prompt/query embedding."""

    def __init__(
        self,
        hidden_dim: int,
        max_pred_len: int,
        dropout: float = 0.0,
        diagnostics_level: str = "standard",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_pred_len = int(max_pred_len)
        self.diagnostics_level = diagnostics_level
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_dim),
        )
        self.predict_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.max_pred_len),
        )
        self.metadata = {
            "decoder_type": "HorizonDirectDecoder",
            "decoder_input_strategy": "direct_multi_output_horizon_head",
            "decoder_context_mode": "last_state",
            "teacher_forcing": False,
            "autoregressive": False,
            "uses_history_only": True,
            "uses_st_prompt": False,
            "future_observed_features_used": False,
        }

    def forward(
        self,
        z_fine: Tensor,
        z_coarse: Tensor,
        st_prompt: Tensor | None = None,
        macro_prompt: Tensor | None = None,
        return_aux: bool = False,
    ):
        if z_fine.shape != z_coarse.shape or z_fine.ndim != 4:
            raise ValueError("z_fine/z_coarse must have matching [B,L,N,D] shapes.")
        B, _, N, D = z_fine.shape
        if D != self.hidden_dim:
            raise ValueError(f"Expected D={self.hidden_dim}, got {D}.")
        fine_last = z_fine[:, -1, :, :]
        coarse_last = z_coarse[:, -1, :, :]
        if macro_prompt is None:
            macro_summary = torch.zeros_like(fine_last)
        else:
            if macro_prompt.ndim != 4 or macro_prompt.shape[:2] != (B, N) or macro_prompt.shape[3] != D:
                raise ValueError("macro_prompt must align with z_fine/z_coarse on B,N,D.")
            macro_summary = macro_prompt.mean(dim=2)
        context = self.fusion_layer(torch.cat([fine_last, coarse_last, macro_summary], dim=-1))
        pred = self.predict_head(context).transpose(1, 2).contiguous()
        if not return_aux:
            return pred
        diagnostic_scalars = {
            "decoder_context_norm_mean": None,
            "decoder_macro_summary_norm_mean": None,
            "decoder_st_prompt_norm_mean": None,
            "decoder_context_future_norm_mean": None,
        }
        if self.diagnostics_level != "none":
            with torch.no_grad():
                diagnostic_scalars.update(
                    {
                        "decoder_context_norm_mean": float(context.detach().float().norm(dim=-1).mean().cpu().item()),
                        "decoder_macro_summary_norm_mean": float(
                            macro_summary.detach().float().norm(dim=-1).mean().cpu().item()
                        ),
                        "decoder_context_future_norm_mean": float(
                            context.detach().float().norm(dim=-1).mean().cpu().item()
                        ),
                    }
                )
        aux = {
            "decoder_type": self.metadata["decoder_type"],
            "decoder_input_strategy": self.metadata["decoder_input_strategy"],
            "decoder_context_mode": self.metadata["decoder_context_mode"],
            "teacher_forcing": False,
            "autoregressive": False,
            "decoder_uses_history_only": True,
            "future_observed_features_used": False,
            **diagnostic_scalars,
        }
        return pred, aux


class STPromptFullHistoryDecoder(nn.Module):
    """Full-history per-turbine cross-attention decoder driven by ST prompt queries."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        history_len: int | None = None,
        diagnostics_level: str = "standard",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.history_len = history_len
        self.diagnostics_level = diagnostics_level
        heads = max(1, int(num_heads))
        while self.hidden_dim % heads != 0 and heads > 1:
            heads -= 1
        self.fine_history_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.coarse_history_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.macro_prompt_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_gate = nn.Linear(self.hidden_dim * 4, 3)
        self.context_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.predict_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.metadata = {
            "decoder_type": "STPromptFullHistoryDecoder",
            "decoder_input_strategy": "direct_multi_output_prompt_query",
            "decoder_context_mode": "full_history_cross_attention",
            "history_len": self.history_len,
            "teacher_forcing": False,
            "autoregressive": False,
            "uses_history_only": True,
            "future_observed_features_used": False,
        }

    @staticmethod
    def _sample_attention_rows(weights: Tensor, max_rows: int = 4096) -> Tensor:
        rows = weights.detach()
        if rows.shape[0] > max_rows:
            step = max(1, rows.shape[0] // max_rows)
            rows = rows[::step][:max_rows]
        return rows.float()

    @classmethod
    def _attention_entropy(cls, weights: Tensor) -> Tensor:
        with torch.no_grad():
            probs = cls._sample_attention_rows(weights).clamp_min(1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return torch.special.entr(probs).sum(dim=-1).mean(dim=0).cpu()

    @classmethod
    def _expected_lag(cls, weights: Tensor, full_history_len: int) -> Tensor:
        weights = cls._sample_attention_rows(weights)
        L = weights.shape[-1]
        start_index = full_history_len - L
        history_index = torch.arange(start_index, full_history_len, device=weights.device, dtype=weights.dtype)
        lag = (full_history_len - 1) - history_index
        return (weights * lag.view(1, 1, L)).sum(dim=-1).mean(dim=0).cpu()

    def forward(
        self,
        z_fine: Tensor,
        z_coarse: Tensor,
        st_prompt: Tensor,
        macro_prompt: Tensor | None = None,
        return_aux: bool = False,
    ):
        if z_fine.shape != z_coarse.shape:
            raise ValueError(f"z_fine shape {tuple(z_fine.shape)} must match z_coarse {tuple(z_coarse.shape)}.")
        if z_fine.ndim != 4:
            raise ValueError(f"z_fine/z_coarse must be [B,L,N,D], got {tuple(z_fine.shape)}.")
        if st_prompt.ndim != 4:
            raise ValueError(f"st_prompt must be [1,H,N,D], got {tuple(st_prompt.shape)}.")
        B, L, N, D = z_fine.shape
        if D != self.hidden_dim:
            raise ValueError(f"Expected D={self.hidden_dim}, got {D}.")
        if st_prompt.shape[0] != 1 or st_prompt.shape[2] != N or st_prompt.shape[3] != D:
            raise ValueError("st_prompt must align with z_fine/z_coarse as [1,H,N,D].")
        if macro_prompt is None:
            macro_prompt = torch.zeros(B, N, 1, D, device=z_fine.device, dtype=z_fine.dtype)
        if macro_prompt.ndim != 4:
            raise ValueError(f"macro_prompt must be [B,N,P,D], got {tuple(macro_prompt.shape)}.")
        if macro_prompt.shape[0] != B or macro_prompt.shape[1] != N or macro_prompt.shape[3] != D:
            raise ValueError("macro_prompt must align with z_fine/z_coarse on B,N,D.")

        if self.history_len is None:
            z_fine_memory = z_fine
            z_coarse_memory = z_coarse
            actual_history_len = L
        else:
            actual_history_len = min(int(self.history_len), L)
            z_fine_memory = z_fine[:, -actual_history_len:, :, :]
            z_coarse_memory = z_coarse[:, -actual_history_len:, :, :]
        H = int(st_prompt.shape[1])
        query_bhnd = st_prompt.expand(B, -1, -1, -1)
        query = query_bhnd.permute(0, 2, 1, 3).reshape(B * N, H, D)
        fine_memory = z_fine_memory.permute(0, 2, 1, 3).reshape(B * N, actual_history_len, D)
        coarse_memory = z_coarse_memory.permute(0, 2, 1, 3).reshape(B * N, actual_history_len, D)
        macro_memory = macro_prompt.reshape(B * N, macro_prompt.shape[2], D)

        need_attention_diagnostics = self.diagnostics_level in {"standard", "full"}
        fine_context, fine_weights = self.fine_history_attention(
            query, fine_memory, fine_memory, need_weights=need_attention_diagnostics
        )
        coarse_context, coarse_weights = self.coarse_history_attention(
            query, coarse_memory, coarse_memory, need_weights=need_attention_diagnostics
        )
        macro_context, macro_weights = self.macro_prompt_attention(
            query, macro_memory, macro_memory, need_weights=need_attention_diagnostics
        )
        context_weights = torch.softmax(
            self.context_gate(torch.cat([query, fine_context, coarse_context, macro_context], dim=-1)),
            dim=-1,
        )
        fused_context = (
            context_weights[..., 0:1] * fine_context
            + context_weights[..., 1:2] * coarse_context
            + context_weights[..., 2:3] * macro_context
        )
        hidden = self.context_norm(query + fused_context)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        pred = self.predict_head(hidden).reshape(B, N, H).transpose(1, 2).contiguous()
        if not return_aux:
            return pred

        diagnostic_scalars = {
            "decoder_context_norm_mean": None,
            "decoder_macro_summary_norm_mean": None,
            "decoder_st_prompt_norm_mean": None,
            "decoder_context_future_norm_mean": None,
        }
        if self.diagnostics_level != "none":
            with torch.no_grad():
                diagnostic_scalars = {
                "decoder_context_norm_mean": float(fused_context.detach().float().norm(dim=-1).mean().cpu().item()),
                "decoder_macro_summary_norm_mean": float(macro_context.detach().float().norm(dim=-1).mean().cpu().item()),
                "decoder_st_prompt_norm_mean": float(st_prompt.detach().float().norm(dim=-1).mean().cpu().item()),
                "decoder_context_future_norm_mean": float(hidden.detach().float().norm(dim=-1).mean().cpu().item()),
                }
        aux = {
            "decoder_type": self.metadata["decoder_type"],
            "decoder_input_strategy": self.metadata["decoder_input_strategy"],
            "decoder_context_mode": self.metadata["decoder_context_mode"],
            "teacher_forcing": False,
            "autoregressive": False,
            "decoder_uses_history_only": True,
            "future_observed_features_used": False,
            "decoder_history_len_config": self.history_len,
            "decoder_actual_history_len": actual_history_len,
            "decoder_used_complete_history": self.history_len is None and actual_history_len == L,
            "decoder_query_shape": [B, H, N, D],
            "attention_weights_requested": need_attention_diagnostics,
            **diagnostic_scalars,
        }
        if need_attention_diagnostics:
            fine_entropy = self._attention_entropy(fine_weights)
            coarse_entropy = self._attention_entropy(coarse_weights)
            macro_entropy = self._attention_entropy(macro_weights)
            with torch.no_grad():
                weights_per_h = context_weights.detach().float().mean(dim=0).cpu()
                aux.update(
                    {
                        "fine_history_attention_entropy_per_horizon": fine_entropy,
                        "coarse_history_attention_entropy_per_horizon": coarse_entropy,
                        "macro_attention_entropy_per_horizon": macro_entropy,
                        "fine_effective_history_steps_per_horizon": torch.exp(fine_entropy),
                        "coarse_effective_history_steps_per_horizon": torch.exp(coarse_entropy),
                        "fine_expected_lag_per_horizon": self._expected_lag(fine_weights, L),
                        "coarse_expected_lag_per_horizon": self._expected_lag(coarse_weights, L),
                        "fine_context_norm_per_horizon": fine_context.detach().float().norm(dim=-1).mean(dim=0).cpu(),
                        "coarse_context_norm_per_horizon": coarse_context.detach().float().norm(dim=-1).mean(dim=0).cpu(),
                        "macro_context_norm_per_horizon": macro_context.detach().float().norm(dim=-1).mean(dim=0).cpu(),
                        "context_weight_fine_per_horizon": weights_per_h[:, 0],
                        "context_weight_coarse_per_horizon": weights_per_h[:, 1],
                        "context_weight_macro_per_horizon": weights_per_h[:, 2],
                        "fine_attention_weight_sum_mean": fine_weights.detach().float().sum(dim=-1).mean().cpu(),
                        "coarse_attention_weight_sum_mean": coarse_weights.detach().float().sum(dim=-1).mean().cpu(),
                        "macro_attention_weight_sum_mean": macro_weights.detach().float().sum(dim=-1).mean().cpu(),
                    }
                )
        return pred, aux
