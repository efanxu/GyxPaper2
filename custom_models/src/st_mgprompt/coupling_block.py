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
        self.branch_mode = str(config.vadsp_gate_mode)
        self.temporal_branch_transform = str(config.temporal_branch_transform)
        self.cross_granularity_interaction = str(config.cross_granularity_interaction)
        self.use_macro_prompt = bool(
            config.use_macro_prompt
            and self.cross_granularity_interaction == "cross_fusion"
            and self.branch_mode != "fine_only"
        )
        self.use_cross_fusion = bool(
            config.use_cross_fusion
            and self.cross_granularity_interaction == "cross_fusion"
            and self.branch_mode not in {"fine_only", "coarse_only"}
        )
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
        self.shared_temporal_output_projections = None
        if self.temporal_branch_transform == "shared":
            shared_transform = self.fine_micro_graph_temporal_encoder.block.causal_tcn
            self.coarse_macro_graph_temporal_encoder.block.causal_tcn = shared_transform
            self.shared_temporal_output_projections = nn.ModuleDict(
                {
                    "fine": nn.Linear(D, D),
                    "coarse": nn.Linear(D, D),
                }
            )
            for projection in self.shared_temporal_output_projections.values():
                nn.init.eye_(projection.weight)
                nn.init.zeros_(projection.bias)
        self.macro_prompt_encoder = (
            MacroTrendPrompt(
                hidden_dim=D,
                prompt_len=config.macro_prompt_len,
                dropout=config.dropout,
                pooling=config.macro_prompt_pooling,
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
                macro_to_fine_exclude_recent_len=config.macro_to_fine_exclude_recent_len,
                fusion_mode=config.fusion_mode,
                disable_reverse_cross=config.disable_reverse_cross,
                disable_macro_to_fine_cross=config.disable_macro_to_fine_cross,
                macro_to_fine_mode=config.macro_to_fine_mode,
                share_cross_attention_projections=config.share_cross_attention_projections,
                diagnostics_level=config.diagnostics_level,
            )
            if self.use_cross_fusion
            else None
        )
        self.direct_concat_mlp = None
        if self.cross_granularity_interaction == "direct_concat_mlp":
            activation = nn.GELU if config.direct_concat_mlp_activation == "gelu" else nn.ReLU
            layers: list[nn.Module]
            if config.direct_concat_mlp_layers == 2:
                width = int(config.direct_concat_mlp_hidden_dim)
                layers = [
                    nn.Linear(D * 2, width),
                    activation(),
                    nn.Dropout(config.dropout),
                    nn.Linear(width, D),
                ]
            else:
                layers = [nn.Linear(D * 2, D)]
            layers.append(nn.LayerNorm(D))
            self.direct_concat_mlp = nn.Sequential(*layers)

    def forward(
        self,
        x_fine: Tensor,
        x_coarse: Tensor,
        A_micro: Tensor | None,
        A_macro: Tensor | None,
        st_prompt_context=None,
    ) -> tuple[Tensor, Tensor, dict]:
        micro_graph_aux = {}
        macro_graph_aux = {}
        if self.branch_mode == "coarse_only":
            h_fine = torch.zeros_like(x_fine)
        elif self.collect_detailed_diagnostics:
            h_fine, micro_graph_aux = self.fine_micro_graph_temporal_encoder(
                x_fine, A_micro if self.use_graph else None, return_aux=True
            )
        else:
            h_fine = self.fine_micro_graph_temporal_encoder(x_fine, A_micro if self.use_graph else None)
        if self.branch_mode == "fine_only":
            h_coarse = torch.zeros_like(x_coarse)
        elif self.collect_detailed_diagnostics:
            h_coarse, macro_graph_aux = self.coarse_macro_graph_temporal_encoder(
                x_coarse, A_macro if self.use_graph else None, return_aux=True
            )
        else:
            h_coarse = self.coarse_macro_graph_temporal_encoder(x_coarse, A_macro if self.use_graph else None)
        if self.shared_temporal_output_projections is not None:
            h_fine = self.shared_temporal_output_projections["fine"](h_fine)
            h_coarse = self.shared_temporal_output_projections["coarse"](h_coarse)
        if self.branch_mode == "coarse_only":
            h_fine = torch.zeros_like(h_fine)
        if self.branch_mode == "fine_only":
            h_coarse = torch.zeros_like(h_coarse)
        if self.macro_prompt_encoder is None:
            macro_prompt = None
            prompt_aux = {
                "macro_prompt_attn_entropy": None,
                "macro_prompt_norm_mean": None,
                "macro_prompt_pooling": None,
                "macro_prompt_attention_enabled": False,
            }
        else:
            macro_prompt, prompt_aux = self.macro_prompt_encoder(h_coarse)
        if self.direct_concat_mlp is not None:
            macro_prompt = None
            fused = self.direct_concat_mlp(torch.cat([h_fine, h_coarse], dim=-1))
            new_fine, new_coarse = fused, fused
            fusion_aux = {
                "macro_attn_entropy": None,
                "fine_attn_entropy": None,
                "fusion_gate_mean": None,
                "fusion_gate_std": None,
                "attention_weights_requested": False,
                "entropy_called": False,
                "cross_fusion_uses_spatial_enhanced_features": False,
                "disable_reverse_cross": True,
                "disable_macro_to_fine_cross": True,
                "macro_to_fine_mode": "disabled",
                "macro_to_fine_exclude_recent_len": 0,
                "share_cross_attention_projections": False,
                "fusion_mode": "direct_concat_mlp",
                "coarse_to_fine_source": "none",
                "direct_concat_mlp_called": True,
            }
        elif self.symmetric_cross_fusion is None:
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
                "disable_macro_to_fine_cross": bool(self.config.disable_macro_to_fine_cross),
                "macro_to_fine_mode": self.config.macro_to_fine_mode,
                "macro_to_fine_exclude_recent_len": self.config.macro_to_fine_exclude_recent_len,
                "share_cross_attention_projections": bool(
                    self.config.share_cross_attention_projections
                ),
                "fusion_mode": "independent",
                "coarse_to_fine_source": "none",
                "direct_concat_mlp_called": False,
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
            "temporal_branch_transform": self.temporal_branch_transform,
            "shared_temporal_parameters": self.temporal_branch_transform == "shared",
            "cross_granularity_interaction": self.cross_granularity_interaction,
            "direct_concat_mlp_called": bool(fusion_aux.get("direct_concat_mlp_called", False)),
            "fine_branch_active": self.branch_mode != "coarse_only",
            "coarse_branch_active": self.branch_mode != "fine_only",
        }
        for key, value in micro_graph_aux.items():
            aux[f"micro_{key}"] = value
        for key, value in macro_graph_aux.items():
            aux[f"macro_{key}"] = value
        aux.update(prompt_aux)
        aux.update(fusion_aux)
        return new_fine, new_coarse, aux
