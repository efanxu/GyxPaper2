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
        self.basic_fusion_mode = config.fusion_mode if config.fusion_mode in {"add", "concat", "unified_gated"} else None
        self.use_macro_prompt = bool(
            config.use_macro_prompt
            and self.cross_granularity_interaction == "cross_fusion"
            and self.basic_fusion_mode is None
            and self.branch_mode != "fine_only"
        )
        self.use_cross_fusion = bool(
            config.use_cross_fusion
            and self.cross_granularity_interaction == "cross_fusion"
            and self.basic_fusion_mode is None
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
            diffusion_direction=config.diffusion_direction,
            diffusion_projection_mode=config.diffusion_projection_mode,
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
            diffusion_direction=config.diffusion_direction,
            diffusion_projection_mode=config.diffusion_projection_mode,
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
        self.basic_add_norm = nn.LayerNorm(D) if self.basic_fusion_mode == "add" else None
        self.basic_concat_mlp = None
        if self.basic_fusion_mode == "concat":
            self.basic_concat_mlp = nn.Sequential(
                nn.Linear(D * 2, D),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(D, D),
                nn.LayerNorm(D),
            )
        self.unified_gate = None
        self.unified_gate_norm = None
        if self.basic_fusion_mode == "unified_gated":
            self.unified_gate = nn.Linear(D * 2, D)
            nn.init.xavier_uniform_(self.unified_gate.weight)
            nn.init.constant_(self.unified_gate.bias, float(config.unified_gate_init_bias))
            self.unified_gate_norm = nn.LayerNorm(D)

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
                "macro_prompt_pairwise_cosine_mean": None,
                "macro_prompt_pairwise_cosine_max": None,
                "macro_prompt_temporal_weight_mean": None,
                "macro_prompt_source_history_range": "NOT_APPLICABLE",
                "macro_prompt_token_count": 0,
            }
        else:
            macro_prompt, prompt_aux = self.macro_prompt_encoder(h_coarse)
        if self.basic_fusion_mode is not None:
            macro_prompt = None
            gate = None
            if self.basic_fusion_mode == "add":
                fused = self.basic_add_norm(h_fine + h_coarse)
            elif self.basic_fusion_mode == "concat":
                fused = self.basic_concat_mlp(torch.cat([h_fine, h_coarse], dim=-1))
            else:
                gate_input_fine = h_fine.detach() if self.config.unified_gate_stop_gradient else h_fine
                gate_input_coarse = h_coarse.detach() if self.config.unified_gate_stop_gradient else h_coarse
                gate = torch.sigmoid(
                    self.unified_gate(torch.cat([gate_input_fine, gate_input_coarse], dim=-1))
                )
                fused = self.unified_gate_norm(gate * h_fine + (1.0 - gate) * h_coarse)
            new_fine, new_coarse = fused, fused
            gate_stats = {
                "fusion_gate_mean": None,
                "fusion_gate_std": None,
                "fusion_gate_q05": None,
                "fusion_gate_q25": None,
                "fusion_gate_q50": None,
                "fusion_gate_q75": None,
                "fusion_gate_q95": None,
                "fusion_gate_saturation_ratio": None,
            }
            if gate is not None and self.config.diagnostics_level != "none":
                with torch.no_grad():
                    gate_flat = gate.detach().float().reshape(-1)
                    if gate_flat.numel() > 1_000_000:
                        sample_step = (gate_flat.numel() + 999_999) // 1_000_000
                        gate_for_quantiles = gate_flat[::sample_step]
                    else:
                        gate_for_quantiles = gate_flat
                    quantiles = torch.quantile(
                        gate_for_quantiles,
                        torch.tensor(
                            [0.05, 0.25, 0.5, 0.75, 0.95],
                            device=gate_for_quantiles.device,
                        ),
                    )
                    gate_stats = {
                        "fusion_gate_mean": float(gate_flat.mean().cpu().item()),
                        "fusion_gate_std": float(gate_flat.std(unbiased=False).cpu().item()),
                        "fusion_gate_q05": float(quantiles[0].cpu().item()),
                        "fusion_gate_q25": float(quantiles[1].cpu().item()),
                        "fusion_gate_q50": float(quantiles[2].cpu().item()),
                        "fusion_gate_q75": float(quantiles[3].cpu().item()),
                        "fusion_gate_q95": float(quantiles[4].cpu().item()),
                        "fusion_gate_saturation_ratio": float(
                            ((gate_flat <= 0.05) | (gate_flat >= 0.95)).float().mean().cpu().item()
                        ),
                    }
            fusion_aux = {
                "macro_attn_entropy": None,
                "fine_attn_entropy": None,
                **gate_stats,
                "attention_weights_requested": False,
                "entropy_called": False,
                "cross_fusion_uses_spatial_enhanced_features": False,
                "disable_reverse_cross": True,
                "disable_macro_to_fine_cross": True,
                "macro_to_fine_mode": "not_applicable",
                "macro_to_fine_exclude_recent_len": 0,
                "share_cross_attention_projections": False,
                "fusion_mode": self.basic_fusion_mode,
                "coarse_to_fine_source": "not_applicable",
                "fine_to_coarse_source": "not_applicable",
                "direct_concat_mlp_called": False,
                "basic_fusion_called": True,
                "gate_input": self.config.unified_gate_input if gate is not None else "NOT_APPLICABLE",
                "gate_level": self.config.unified_gate_level if gate is not None else "NOT_APPLICABLE",
                "gate_range": "[0,1]" if gate is not None else "NOT_APPLICABLE",
                "gate_initial_bias": float(self.config.unified_gate_init_bias) if gate is not None else None,
                "gate_stop_gradient": bool(self.config.unified_gate_stop_gradient) if gate is not None else None,
                "attention_not_applicable_reason": "basic_fusion_has_no_cross_attention",
            }
        elif self.direct_concat_mlp is not None:
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
                "basic_fusion_called": False,
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
                "basic_fusion_called": False,
            }
        else:
            new_fine, new_coarse, fusion_aux = self.symmetric_cross_fusion(
                h_fine=h_fine,
                h_coarse=h_coarse,
                macro_prompt=macro_prompt,
            )
            fusion_aux["basic_fusion_called"] = False
        if self.config.diagnostics_level != "none":
            with torch.no_grad():
                fine_flat = h_fine.detach().float().reshape(-1, h_fine.shape[-1])
                coarse_flat = h_coarse.detach().float().reshape(-1, h_coarse.shape[-1])
                new_fine_flat = new_fine.detach().float().reshape(-1, new_fine.shape[-1])
                new_coarse_flat = new_coarse.detach().float().reshape(-1, new_coarse.shape[-1])
                fusion_aux.update(
                    {
                        "pre_fusion_cosine_similarity": float(
                            torch.nn.functional.cosine_similarity(fine_flat, coarse_flat, dim=-1).mean().cpu().item()
                        ),
                        "post_fusion_cosine_similarity": float(
                            torch.nn.functional.cosine_similarity(new_fine_flat, new_coarse_flat, dim=-1).mean().cpu().item()
                        ),
                        "fine_representation_shift": float(
                            (new_fine.detach().float() - h_fine.detach().float()).norm(dim=-1).mean().cpu().item()
                        ),
                        "coarse_representation_shift": float(
                            (new_coarse.detach().float() - h_coarse.detach().float()).norm(dim=-1).mean().cpu().item()
                        ),
                        "fine_output_shape": list(new_fine.shape),
                        "coarse_output_shape": list(new_coarse.shape),
                    }
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
            "basic_fusion_called": bool(fusion_aux.get("basic_fusion_called", False)),
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
