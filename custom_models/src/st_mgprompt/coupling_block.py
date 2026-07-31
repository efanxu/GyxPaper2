from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import STMGPromptConfig
from .cross_fusion import SymmetricCrossFusion
from .prompt_alignment import MacroTrendPrompt
from .temporal_layers import CoarseMacroGraphTemporalEncoder, FineMicroGraphTemporalEncoder


class STMGPromptCouplingBlock(nn.Module):
    """Graph-temporal encoding, macro prompt generation, and symmetric cross-fusion."""

    def __init__(self, config: STMGPromptConfig, hidden_dim: int | None = None) -> None:
        super().__init__()
        D = int(hidden_dim or config.hidden_dim)
        self.config = config
        self.collect_detailed_diagnostics = config.diagnostics_level in {"standard", "full"}
        self.use_graph = bool(config.use_graph_in_temporal_encoder)
        self.use_macro_prompt = bool(config.use_macro_prompt)
        self.use_cross_fusion = bool(config.use_cross_fusion)
        self.fine_micro_graph_temporal_encoder = FineMicroGraphTemporalEncoder(
            hidden_dim=D,
            kernel_size=config.fine_tcn_kernel_size,
            dilations=config.fine_tcn_dilations,
            dropout=config.dropout,
            use_graph=self.use_graph,
            graph_operator=config.graph_operator,
            diffusion_order=config.diffusion_order_micro,
            diffusion_use_bidirectional=config.diffusion_use_bidirectional,
            diffusion_beta_init=config.diffusion_beta_init,
            use_temporal_attention=config.use_temporal_attention,
            temporal_attention_heads=config.temporal_attention_heads,
        )
        self.coarse_macro_graph_temporal_encoder = CoarseMacroGraphTemporalEncoder(
            hidden_dim=D,
            kernel_size=config.coarse_tcn_kernel_size,
            dilations=config.coarse_tcn_dilations,
            dropout=config.dropout,
            use_graph=self.use_graph,
            graph_operator=config.graph_operator,
            diffusion_order=config.diffusion_order_macro,
            diffusion_use_bidirectional=config.diffusion_use_bidirectional,
            diffusion_beta_init=config.diffusion_beta_init,
            use_temporal_attention=config.use_temporal_attention,
            temporal_attention_heads=config.temporal_attention_heads,
        )
        self.macro_prompt_encoder = (
            MacroTrendPrompt(
                hidden_dim=D,
                prompt_len=config.macro_prompt_len,
                dropout=config.dropout,
                pooling="attention",
                diagnostics_level=config.diagnostics_level,
            )
            if self.use_macro_prompt
            else None
        )
        self.symmetric_cross_fusion = (
            SymmetricCrossFusion(
                hidden_dim=D,
                num_heads=config.cross_attention_heads,
                dropout=config.dropout,
                recent_len=config.cross_fusion_recent_len,
                fusion_mode=config.fusion_mode,
                disable_reverse_cross=config.disable_reverse_cross,
                diagnostics_level=config.diagnostics_level,
            )
            if self.use_cross_fusion
            else None
        )

    def forward(
        self,
        x_fine: Tensor,
        x_coarse: Tensor,
        A_micro: Tensor | None,
        A_macro: Tensor | None,
        st_prompt_context=None,
    ) -> tuple[Tensor, Tensor, dict]:
        if self.collect_detailed_diagnostics:
            h_fine, micro_graph_aux = self.fine_micro_graph_temporal_encoder(
                x_fine, A_micro if self.use_graph else None, return_aux=True
            )
            h_coarse, macro_graph_aux = self.coarse_macro_graph_temporal_encoder(
                x_coarse, A_macro if self.use_graph else None, return_aux=True
            )
        else:
            h_fine = self.fine_micro_graph_temporal_encoder(x_fine, A_micro if self.use_graph else None)
            h_coarse = self.coarse_macro_graph_temporal_encoder(x_coarse, A_macro if self.use_graph else None)
            micro_graph_aux = {}
            macro_graph_aux = {}
        if self.macro_prompt_encoder is None:
            macro_prompt = None
            prompt_aux = {
                "macro_prompt_attn_entropy": None,
                "macro_prompt_norm_mean": None,
            }
        else:
            macro_prompt, prompt_aux = self.macro_prompt_encoder(h_coarse)
        if self.symmetric_cross_fusion is None:
            # Strict w/o Cross-Fusion: no add/concat substitute and no shared
            # cross-attention; the two graph-temporal histories remain separate.
            new_fine, new_coarse = h_fine, h_coarse
            fusion_aux = {
                "macro_attn_entropy": None,
                "fine_attn_entropy": None,
                "fusion_gate_mean": None,
                "fusion_gate_std": None,
                "attention_weights_requested": False,
                "entropy_called": False,
                "cross_fusion_uses_spatial_enhanced_features": False,
                "disable_reverse_cross": bool(self.config.disable_reverse_cross),
                "fusion_mode": "independent",
                "coarse_to_fine_source": "none",
            }
        else:
            new_fine, new_coarse, fusion_aux = self.symmetric_cross_fusion(
                h_fine=h_fine,
                h_coarse=h_coarse,
                macro_prompt=macro_prompt,
            )
        scalar_stats = {
            "h_fine_norm_mean": None,
            "h_fine_norm_std": None,
            "h_coarse_norm_mean": None,
            "h_coarse_norm_std": None,
            "h_fine_spatial_delta_norm": None,
            "h_coarse_spatial_delta_norm": None,
            "h_fine_spatial_delta_ratio": None,
            "h_coarse_spatial_delta_ratio": None,
        }
        if self.config.diagnostics_level != "none":
            with torch.no_grad():
                fine_delta = (h_fine.detach() - x_fine.detach()).float().norm(dim=-1).mean()
                coarse_delta = (h_coarse.detach() - x_coarse.detach()).float().norm(dim=-1).mean()
                fine_input_norm = x_fine.detach().float().norm(dim=-1).mean().clamp_min(1e-8)
                coarse_input_norm = x_coarse.detach().float().norm(dim=-1).mean().clamp_min(1e-8)
                scalar_stats = {
                "h_fine_norm_mean": float(h_fine.detach().float().norm(dim=-1).mean().cpu().item()),
                "h_fine_norm_std": float(h_fine.detach().float().norm(dim=-1).std(unbiased=False).cpu().item()),
                "h_coarse_norm_mean": float(h_coarse.detach().float().norm(dim=-1).mean().cpu().item()),
                "h_coarse_norm_std": float(h_coarse.detach().float().norm(dim=-1).std(unbiased=False).cpu().item()),
                "h_fine_spatial_delta_norm": float(fine_delta.cpu().item()),
                "h_coarse_spatial_delta_norm": float(coarse_delta.cpu().item()),
                "h_fine_spatial_delta_ratio": float((fine_delta / fine_input_norm).cpu().item()),
                "h_coarse_spatial_delta_ratio": float((coarse_delta / coarse_input_norm).cpu().item()),
                }
        aux = {
            "macro_prompt": macro_prompt,
            "h_fine": h_fine,
            "h_coarse": h_coarse,
            **scalar_stats,
            "macro_prompt_from_spatial_enhanced_coarse": self.use_macro_prompt,
            "cross_fusion_uses_spatial_enhanced_features": self.use_cross_fusion,
            "uses_macro_prompt": self.use_macro_prompt,
            "uses_cross_fusion": self.use_cross_fusion,
        }
        for key, value in micro_graph_aux.items():
            aux[f"micro_{key}"] = value
        for key, value in macro_graph_aux.items():
            aux[f"macro_{key}"] = value
        aux.update(prompt_aux)
        aux.update(fusion_aux)
        return new_fine, new_coarse, aux
