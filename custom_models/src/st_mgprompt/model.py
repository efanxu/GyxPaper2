from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import STMGPromptConfig
from .coupling_block import STMGPromptCouplingBlock
from .decoder import (
    HorizonDirectDecoder,
    STPromptDirectDecoder,
    STPromptFullHistoryDecoder,
    STPromptHistoryPoolingDecoder,
)
from .graph_layers import AdaptiveGraphBuilder, FixedPriorGraphBuilder, SimpleGraphConv
from .prompt_alignment import STPromptEmbedding
from .temporal_layers import DualGraphTemporalEncoder
from .volatility_patching import VolatilityAwareDynamicSemanticPatching


class STMGPromptTemporalOnly(nn.Module):
    """Step-0 temporal-only baseline with parameters shared across turbines."""

    def __init__(self, config: STMGPromptConfig, input_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.input_dim = int(input_dim or len(config.feature_cols))
        self.hidden_dim = int(config.hidden_dim)
        self.max_pred_len = int(config.max_pred_len)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.vadsp = (
            VolatilityAwareDynamicSemanticPatching(
                hidden_dim=self.hidden_dim,
                feature_names=config.feature_cols,
                volatility_source_cols=config.volatility_source_cols,
                volatility_mode=config.volatility_mode,
                vol_window=config.vol_window,
                fine_kernel_size=config.fine_kernel_size,
                coarse_windows=config.coarse_windows,
                coarse_alignment_mode=config.coarse_alignment_mode,
                coarse_upsample_rule=config.coarse_upsample_rule,
                dropout=config.dropout,
                use_train_robust_volatility=config.use_train_robust_volatility,
                volatility_scale_eps=config.volatility_scale_eps,
                tau_min=config.vadsp_tau_min,
                tau_max=config.vadsp_tau_max,
                residual_scale=config.vadsp_residual_scale,
                branch_beta_init=config.vadsp_branch_beta_init,
                diagnostics_level=config.diagnostics_level,
                gate_mode=config.vadsp_gate_mode,
                random_gate_seed=config.vadsp_random_seed,
            )
            if config.use_vadsp
            else None
        )
        self.encoder = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(self.hidden_dim, self.max_pred_len)

    def forward(self, x: Tensor | dict[str, Tensor], graph_data=None, aux_data=None) -> dict[str, Tensor | dict]:
        x_tensor = x["x"] if isinstance(x, dict) else x
        if x_tensor.ndim != 4:
            raise ValueError(f"x must be [B,L,N,C], got {tuple(x_tensor.shape)}.")
        B, L, N, C = x_tensor.shape
        if C != self.input_dim:
            raise ValueError(f"Expected C={self.input_dim}, got {C}.")
        shared = x_tensor.permute(0, 2, 1, 3).reshape(B * N, L, C)
        projected = self.input_proj(shared).reshape(B, N, L, self.hidden_dim).permute(0, 2, 1, 3)
        vadsp_out = None
        if self.vadsp is not None:
            vadsp_out = self.vadsp(projected, raw_x=x_tensor)
            projected = vadsp_out["x_fine"] + vadsp_out["x_coarse"]
        encoded = projected.permute(0, 2, 1, 3).reshape(B * N, L, self.hidden_dim)
        _, h_n = self.encoder(encoded)
        node_context = self.dropout(h_n[-1])
        pred_bnh = self.head(node_context).reshape(B, N, self.max_pred_len)
        pred_bhn = pred_bnh.transpose(1, 2).contiguous()
        result = {
            "pred": pred_bhn,
            "aux": {
                "pred_bnh": pred_bnh,
                "model_family": "STMGPrompt",
                "model_name": self.config.model_name,
                "uses_target_mask_as_input": False,
                "uses_vadsp": self.vadsp is not None,
            },
        }
        if vadsp_out is not None:
            result["aux"].update(vadsp_out)
        return result


class STMGPrompt_TrendPriorGraph(nn.Module):
    """Step-2 graph smoke ablation; not the final ST-MG Coupling Block model."""

    def __init__(
        self,
        config: STMGPromptConfig,
        input_dim: int | None = None,
        graph_data: dict | None = None,
    ) -> None:
        super().__init__()
        if graph_data is None:
            raise ValueError("STMGPrompt_TrendPriorGraph requires train-only graph_data.")
        self.config = config
        self.input_dim = int(input_dim or len(config.feature_cols))
        self.hidden_dim = int(config.hidden_dim)
        self.max_pred_len = int(config.max_pred_len)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.vadsp = VolatilityAwareDynamicSemanticPatching(
            hidden_dim=self.hidden_dim,
            feature_names=config.feature_cols,
            volatility_source_cols=config.volatility_source_cols,
            volatility_mode=config.volatility_mode,
            vol_window=config.vol_window,
            fine_kernel_size=config.fine_kernel_size,
            coarse_windows=config.coarse_windows,
            coarse_alignment_mode=config.coarse_alignment_mode,
            coarse_upsample_rule=config.coarse_upsample_rule,
            dropout=config.dropout,
            use_train_robust_volatility=config.use_train_robust_volatility,
            volatility_scale_eps=config.volatility_scale_eps,
            tau_min=config.vadsp_tau_min,
            tau_max=config.vadsp_tau_max,
            residual_scale=config.vadsp_residual_scale,
            branch_beta_init=config.vadsp_branch_beta_init,
            diagnostics_level=config.diagnostics_level,
            dynamic_gating=config.use_vadsp,
            gate_mode=config.vadsp_gate_mode if config.use_vadsp else "fixed_dual",
            random_gate_seed=config.vadsp_random_seed,
        )
        A_macro = torch.as_tensor(graph_data["A_macro_trend"], dtype=torch.float32)
        A_micro = torch.as_tensor(graph_data["A_micro_local"], dtype=torch.float32)
        if A_macro.shape != A_micro.shape or A_macro.ndim != 2 or A_macro.shape[0] != A_macro.shape[1]:
            raise ValueError("A_macro_trend and A_micro_local must both be square [N,N] graphs.")
        self.num_nodes = int(A_macro.shape[0])
        self.register_buffer("A_macro_prior", A_macro)
        self.register_buffer("A_micro_prior", A_micro)
        adaptive_enabled = bool(config.use_adaptive_graph and config.adaptive_support_mode != "fixed")
        if adaptive_enabled:
            self.macro_graph_builder = AdaptiveGraphBuilder(
                num_nodes=self.num_nodes,
                node_embed_dim=config.node_embed_dim,
                temperature=config.adaptive_graph_temperature,
                fusion_mode=config.graph_fusion_mode,
                use_prior=config.adaptive_support_mode == "prior_constrained",
            )
            self.micro_graph_builder = AdaptiveGraphBuilder(
                num_nodes=self.num_nodes,
                node_embed_dim=config.node_embed_dim,
                temperature=config.adaptive_graph_temperature,
                fusion_mode=config.graph_fusion_mode,
                use_prior=config.adaptive_support_mode == "prior_constrained",
            )
        else:
            self.macro_graph_builder = FixedPriorGraphBuilder()
            self.micro_graph_builder = FixedPriorGraphBuilder()
        self.micro_graph_conv = SimpleGraphConv(self.hidden_dim, dropout=config.dropout)
        self.macro_graph_conv = SimpleGraphConv(self.hidden_dim, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(self.hidden_dim, self.max_pred_len)
        self.graph_metadata = graph_data.get("metadata", {})

    def forward(self, x: Tensor | dict[str, Tensor], graph_data=None, aux_data=None) -> dict[str, Tensor | dict]:
        x_tensor = x["x"] if isinstance(x, dict) else x
        if x_tensor.ndim != 4:
            raise ValueError(f"x must be [B,L,N,C], got {tuple(x_tensor.shape)}.")
        B, L, N, C = x_tensor.shape
        if C != self.input_dim:
            raise ValueError(f"Expected C={self.input_dim}, got {C}.")
        if N != self.num_nodes:
            raise ValueError(f"Graph node count {self.num_nodes} does not match x N={N}.")
        shared = x_tensor.permute(0, 2, 1, 3).reshape(B * N, L, C)
        projected = self.input_proj(shared).reshape(B, N, L, self.hidden_dim).permute(0, 2, 1, 3)
        vadsp_out = self.vadsp(projected, raw_x=x_tensor)
        A_micro = self.micro_graph_builder(self.A_micro_prior)
        A_macro = self.macro_graph_builder(self.A_macro_prior)
        h_fine = self.micro_graph_conv(vadsp_out["x_fine"], A_micro)
        h_coarse = self.macro_graph_conv(vadsp_out["x_coarse"], A_macro)
        h = self.dropout(h_fine + h_coarse)
        pred_bnh = self.head(h[:, -1, :, :])
        pred_bhn = pred_bnh.transpose(1, 2).contiguous()
        aux = {
            "pred_bnh": pred_bnh,
            "model_family": "STMGPrompt",
            "model_name": self.config.model_name,
            "uses_target_mask_as_input": False,
            "uses_vadsp": True,
            "uses_trend_prior_graph": True,
            "uses_adaptive_graph": True,
            "graph_metadata": self.graph_metadata,
            "macro_row_sum": A_macro.sum(dim=-1).detach(),
            "micro_row_sum": A_micro.sum(dim=-1).detach(),
        }
        aux.update(vadsp_out)
        return {"pred": pred_bhn, "aux": aux}


class STMGPrompt_GraphTemporalSmoke(nn.Module):
    """Step-3 graph-temporal encoder smoke ablation; not the FairFull coupling model."""

    def __init__(
        self,
        config: STMGPromptConfig,
        input_dim: int | None = None,
        graph_data: dict | None = None,
    ) -> None:
        super().__init__()
        if graph_data is None:
            raise ValueError("STMGPrompt_GraphTemporalSmoke requires train-only graph_data.")
        self.config = config
        self.input_dim = int(input_dim or len(config.feature_cols))
        self.hidden_dim = int(config.hidden_dim)
        self.max_pred_len = int(config.max_pred_len)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.vadsp = VolatilityAwareDynamicSemanticPatching(
            hidden_dim=self.hidden_dim,
            feature_names=config.feature_cols,
            volatility_source_cols=config.volatility_source_cols,
            volatility_mode=config.volatility_mode,
            vol_window=config.vol_window,
            fine_kernel_size=config.fine_kernel_size,
            coarse_windows=config.coarse_windows,
            coarse_alignment_mode=config.coarse_alignment_mode,
            coarse_upsample_rule=config.coarse_upsample_rule,
            dropout=config.dropout,
            use_train_robust_volatility=config.use_train_robust_volatility,
            volatility_scale_eps=config.volatility_scale_eps,
            tau_min=config.vadsp_tau_min,
            tau_max=config.vadsp_tau_max,
            residual_scale=config.vadsp_residual_scale,
            branch_beta_init=config.vadsp_branch_beta_init,
            diagnostics_level=config.diagnostics_level,
            dynamic_gating=config.use_vadsp,
            gate_mode=config.vadsp_gate_mode if config.use_vadsp else "fixed_dual",
            random_gate_seed=config.vadsp_random_seed,
        )
        A_macro = torch.as_tensor(graph_data["A_macro_trend"], dtype=torch.float32)
        A_micro = torch.as_tensor(graph_data["A_micro_local"], dtype=torch.float32)
        if A_macro.shape != A_micro.shape or A_macro.ndim != 2 or A_macro.shape[0] != A_macro.shape[1]:
            raise ValueError("A_macro_trend and A_micro_local must both be square [N,N] graphs.")
        self.num_nodes = int(A_macro.shape[0])
        self.register_buffer("A_macro_prior", A_macro)
        self.register_buffer("A_micro_prior", A_micro)
        adaptive_enabled = bool(config.use_adaptive_graph and config.adaptive_support_mode != "fixed")
        if adaptive_enabled:
            self.macro_graph_builder = AdaptiveGraphBuilder(
                num_nodes=self.num_nodes,
                node_embed_dim=config.node_embed_dim,
                temperature=config.adaptive_graph_temperature,
                fusion_mode=config.graph_fusion_mode,
                use_prior=config.adaptive_support_mode == "prior_constrained",
            )
            self.micro_graph_builder = AdaptiveGraphBuilder(
                num_nodes=self.num_nodes,
                node_embed_dim=config.node_embed_dim,
                temperature=config.adaptive_graph_temperature,
                fusion_mode=config.graph_fusion_mode,
                use_prior=config.adaptive_support_mode == "prior_constrained",
            )
        else:
            self.macro_graph_builder = FixedPriorGraphBuilder()
            self.micro_graph_builder = FixedPriorGraphBuilder()
        self.graph_temporal_encoder = DualGraphTemporalEncoder(
            hidden_dim=self.hidden_dim,
            fine_kernel_size=config.fine_tcn_kernel_size,
            coarse_kernel_size=config.coarse_tcn_kernel_size,
            fine_dilations=config.fine_tcn_dilations,
            coarse_dilations=config.coarse_tcn_dilations,
            dropout=config.dropout,
            use_graph=config.use_graph_in_temporal_encoder,
            graph_operator=config.graph_operator,
            diffusion_order_micro=config.diffusion_order_micro,
            diffusion_order_macro=config.diffusion_order_macro,
            diffusion_use_bidirectional=config.diffusion_use_bidirectional,
            diffusion_direction=config.diffusion_direction,
            diffusion_projection_mode=config.diffusion_projection_mode,
            diffusion_beta_init=config.diffusion_beta_init,
            use_temporal_attention=config.use_temporal_attention,
            temporal_attention_heads=config.temporal_attention_heads,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(self.hidden_dim, self.max_pred_len)
        self.graph_metadata = graph_data.get("metadata", {})

    @staticmethod
    def _encoder_stats(h_fine: Tensor, h_coarse: Tensor, x_fine: Tensor, x_coarse: Tensor) -> dict[str, Tensor]:
        return {
            "h_fine_norm_mean": h_fine.norm(dim=-1).mean().detach(),
            "h_fine_norm_std": h_fine.norm(dim=-1).std(unbiased=False).detach(),
            "h_coarse_norm_mean": h_coarse.norm(dim=-1).mean().detach(),
            "h_coarse_norm_std": h_coarse.norm(dim=-1).std(unbiased=False).detach(),
            "h_fine_spatial_delta_norm": (h_fine - x_fine).norm(dim=-1).mean().detach(),
            "h_coarse_spatial_delta_norm": (h_coarse - x_coarse).norm(dim=-1).mean().detach(),
        }

    def forward(self, x: Tensor | dict[str, Tensor], graph_data=None, aux_data=None) -> dict[str, Tensor | dict]:
        x_tensor = x["x"] if isinstance(x, dict) else x
        if x_tensor.ndim != 4:
            raise ValueError(f"x must be [B,L,N,C], got {tuple(x_tensor.shape)}.")
        B, L, N, C = x_tensor.shape
        if C != self.input_dim:
            raise ValueError(f"Expected C={self.input_dim}, got {C}.")
        if N != self.num_nodes:
            raise ValueError(f"Graph node count {self.num_nodes} does not match x N={N}.")
        shared = x_tensor.permute(0, 2, 1, 3).reshape(B * N, L, C)
        projected = self.input_proj(shared).reshape(B, N, L, self.hidden_dim).permute(0, 2, 1, 3)
        vadsp_out = self.vadsp(projected, raw_x=x_tensor)
        A_micro = self.micro_graph_builder(self.A_micro_prior)
        A_macro = self.macro_graph_builder(self.A_macro_prior)
        h_fine, h_coarse = self.graph_temporal_encoder(
            vadsp_out["x_fine"],
            vadsp_out["x_coarse"],
            A_micro if self.config.use_graph_in_temporal_encoder else None,
            A_macro if self.config.use_graph_in_temporal_encoder else None,
        )
        h = self.dropout(h_fine + h_coarse)
        pred_bnh = self.head(h[:, -1, :, :])
        pred_bhn = pred_bnh.transpose(1, 2).contiguous()
        aux = {
            "pred_bnh": pred_bnh,
            "model_family": "STMGPrompt",
            "model_name": self.config.model_name,
            "uses_target_mask_as_input": False,
            "uses_vadsp": True,
            "uses_trend_prior_graph": True,
            "uses_adaptive_graph": True,
            "uses_graph_temporal_encoder": True,
            "uses_graph_in_temporal_encoder": bool(self.config.use_graph_in_temporal_encoder),
            "uses_temporal_attention": bool(self.config.use_temporal_attention),
            "h_fine": h_fine,
            "h_coarse": h_coarse,
            "graph_metadata": self.graph_metadata,
            "macro_row_sum": A_macro.sum(dim=-1).detach(),
            "micro_row_sum": A_micro.sum(dim=-1).detach(),
        }
        aux.update(self._encoder_stats(h_fine, h_coarse, vadsp_out["x_fine"], vadsp_out["x_coarse"]))
        aux.update(vadsp_out)
        return {"pred": pred_bhn, "aux": aux}


class STMGPrompt_FairFull(nn.Module):
    """FairFull path: VADSP -> ST-MG Coupling Block(s) -> ST-Prompt Direct Decoder."""

    def __init__(
        self,
        config: STMGPromptConfig,
        input_dim: int | None = None,
        graph_data: dict | None = None,
    ) -> None:
        super().__init__()
        if config.use_graph_in_temporal_encoder and graph_data is None:
            raise ValueError("STMGPrompt_FairFull requires train-only graph_data when graph encoding is enabled.")
        self.config = config
        self.input_dim = int(input_dim or len(config.feature_cols))
        self.hidden_dim = int(config.hidden_dim)
        self.max_pred_len = int(config.max_pred_len)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.vadsp = VolatilityAwareDynamicSemanticPatching(
            hidden_dim=self.hidden_dim,
            feature_names=config.feature_cols,
            volatility_source_cols=config.volatility_source_cols,
            volatility_mode=config.volatility_mode,
            vol_window=config.vol_window,
            fine_kernel_size=config.fine_kernel_size,
            coarse_windows=config.coarse_windows,
            coarse_alignment_mode=config.coarse_alignment_mode,
            coarse_upsample_rule=config.coarse_upsample_rule,
            dropout=config.dropout,
            use_train_robust_volatility=config.use_train_robust_volatility,
            volatility_scale_eps=config.volatility_scale_eps,
            tau_min=config.vadsp_tau_min,
            tau_max=config.vadsp_tau_max,
            residual_scale=config.vadsp_residual_scale,
            branch_beta_init=config.vadsp_branch_beta_init,
            diagnostics_level=config.diagnostics_level,
            dynamic_gating=config.use_vadsp,
            gate_mode=config.vadsp_gate_mode if config.use_vadsp else "fixed_dual",
            random_gate_seed=config.vadsp_random_seed,
        )
        if graph_data is None:
            self.num_nodes = int(config.num_nodes)
            self.register_buffer("A_macro_prior", torch.eye(self.num_nodes))
            self.register_buffer("A_micro_prior", torch.eye(self.num_nodes))
            self.graph_metadata = {}
        else:
            A_macro = torch.as_tensor(graph_data["A_macro_trend"], dtype=torch.float32)
            A_micro = torch.as_tensor(graph_data["A_micro_local"], dtype=torch.float32)
            if A_macro.shape != A_micro.shape or A_macro.ndim != 2 or A_macro.shape[0] != A_macro.shape[1]:
                raise ValueError("A_macro_trend and A_micro_local must both be square [N,N] graphs.")
            self.num_nodes = int(A_macro.shape[0])
            self.register_buffer("A_macro_prior", A_macro)
            self.register_buffer("A_micro_prior", A_micro)
            self.graph_metadata = graph_data.get("metadata", {})
        adaptive_enabled = bool(config.use_adaptive_graph and config.adaptive_support_mode != "fixed")
        if adaptive_enabled:
            self.macro_graph_builder = AdaptiveGraphBuilder(
                num_nodes=self.num_nodes,
                node_embed_dim=config.node_embed_dim,
                temperature=config.adaptive_graph_temperature,
                fusion_mode=config.graph_fusion_mode,
                use_prior=config.adaptive_support_mode == "prior_constrained",
            )
            self.micro_graph_builder = AdaptiveGraphBuilder(
                num_nodes=self.num_nodes,
                node_embed_dim=config.node_embed_dim,
                temperature=config.adaptive_graph_temperature,
                fusion_mode=config.graph_fusion_mode,
                use_prior=config.adaptive_support_mode == "prior_constrained",
            )
        else:
            self.macro_graph_builder = FixedPriorGraphBuilder()
            self.micro_graph_builder = FixedPriorGraphBuilder()
        self.coupling_blocks = nn.ModuleList(
            [STMGPromptCouplingBlock(config, hidden_dim=self.hidden_dim) for _ in range(config.num_coupling_layers)]
        )
        self.st_prompt = (
            STPromptEmbedding(
                num_nodes=self.num_nodes,
                max_pred_len=self.max_pred_len,
                hidden_dim=self.hidden_dim,
                dropout=config.dropout,
                mode=config.st_prompt_mode,
                use_node_identity=config.st_prompt_use_node_identity,
                use_horizon_identity=config.st_prompt_use_horizon_identity,
                use_shared_horizon_embedding=config.st_prompt_use_shared_horizon_embedding,
                use_type_embedding=config.st_prompt_use_type_embedding,
                type_semantics=config.st_prompt_type_semantics,
            )
            if config.use_st_prompt
            else None
        )
        if not config.use_st_prompt:
            self.direct_decoder = HorizonDirectDecoder(
                hidden_dim=self.hidden_dim,
                max_pred_len=self.max_pred_len,
                dropout=config.dropout,
                diagnostics_level=config.diagnostics_level,
            )
        elif config.decoder_context_mode == "full_history_cross_attention":
            self.direct_decoder = STPromptFullHistoryDecoder(
                hidden_dim=self.hidden_dim,
                num_heads=config.cross_attention_heads,
                dropout=config.dropout,
                history_len=config.decoder_history_len,
                diagnostics_level=config.diagnostics_level,
            )
        elif config.decoder_context_mode in {"mean_pooling_history", "attention_pooling_history"}:
            self.direct_decoder = STPromptHistoryPoolingDecoder(
                hidden_dim=self.hidden_dim,
                pooling="mean" if config.decoder_context_mode == "mean_pooling_history" else "attention",
                dropout=config.dropout,
                diagnostics_level=config.diagnostics_level,
            )
        else:
            self.direct_decoder = STPromptDirectDecoder(
                hidden_dim=self.hidden_dim,
                dropout=config.dropout,
                diagnostics_level=config.diagnostics_level,
            )
        self.coupling_metadata = {
            "stmg_coupling_block_enabled": True,
            "model": config.model_name,
            "num_coupling_layers": int(config.num_coupling_layers),
            "hidden_dim": int(config.hidden_dim),
            "coupling_mode": config.coupling_mode,
            "graph_operator": config.graph_operator,
            "diffusion_order_micro": int(config.diffusion_order_micro),
            "diffusion_order_macro": int(config.diffusion_order_macro),
            "diffusion_use_bidirectional": bool(config.diffusion_use_bidirectional),
            "diffusion_direction": config.diffusion_direction,
            "diffusion_projection_mode": config.diffusion_projection_mode,
            "diffusion_beta_init": float(config.diffusion_beta_init),
            "macro_prompt_from_spatial_enhanced_coarse": bool(self.coupling_blocks[0].use_macro_prompt),
            "macro_prompt_pooling": config.macro_prompt_pooling,
            "macro_prompt_attention_enabled": config.macro_prompt_pooling == "attention",
            "cross_fusion_uses_spatial_enhanced_features": bool(self.coupling_blocks[0].use_cross_fusion),
            "disable_reverse_cross": bool(config.disable_reverse_cross),
            "disable_macro_to_fine_cross": bool(config.disable_macro_to_fine_cross),
            "macro_to_fine_exclude_recent_len": config.macro_to_fine_exclude_recent_len,
            "macro_to_fine_mode": config.macro_to_fine_mode,
            "share_cross_attention_projections": bool(config.share_cross_attention_projections),
            "fusion_mode": config.fusion_mode,
            "unified_gate_input": config.unified_gate_input,
            "unified_gate_level": config.unified_gate_level,
            "unified_gate_init_bias": float(config.unified_gate_init_bias),
            "unified_gate_stop_gradient": bool(config.unified_gate_stop_gradient),
            "temporal_branch_transform": config.temporal_branch_transform,
            "coarse_alignment_mode": config.coarse_alignment_mode,
            "coarse_upsample_rule": config.coarse_upsample_rule,
            "cross_granularity_interaction": config.cross_granularity_interaction,
            "direct_concat_mlp_layers": int(config.direct_concat_mlp_layers),
            "direct_concat_mlp_hidden_dim": int(config.direct_concat_mlp_hidden_dim),
            "direct_concat_mlp_activation": config.direct_concat_mlp_activation,
            "serial_graph_then_fusion": False,
            "st_prompt_direct_decoder_enabled": bool(config.use_st_prompt),
            "st_prompt_use_node_identity": bool(config.st_prompt_use_node_identity),
            "st_prompt_use_horizon_identity": bool(config.st_prompt_use_horizon_identity),
            "st_prompt_use_shared_horizon_embedding": bool(config.st_prompt_use_shared_horizon_embedding),
            "st_prompt_use_type_embedding": bool(config.st_prompt_use_type_embedding),
            "st_prompt_type_semantics": config.st_prompt_type_semantics,
            "st_prompt_information": "+".join(
                name
                for name, enabled in (
                    ("node_identity", config.st_prompt_use_node_identity),
                    ("horizon_identity", config.st_prompt_use_horizon_identity),
                    ("shared_horizon", config.st_prompt_use_shared_horizon_embedding),
                    ("fixed_input_type", config.st_prompt_use_type_embedding),
                )
                if enabled
            ) or "zero_prompt",
            "decoder_context_mode": config.decoder_context_mode,
            "decoder_history_len": config.decoder_history_len,
            "decoder_input_strategy": config.decoder_input_strategy,
            "teacher_forcing": False,
            "future_observed_features_used": False,
            "loss_function": config.loss_function,
            "loss_changed_from_fair_protocol": bool(config.use_msmg_dwu),
            "eligible_for_fair_main_table": bool(config.eligible_for_fair_main_table),
            "eligible_for_method_full_table": bool(
                config.model_name
                in {
                    "STMGPrompt_Full_MSMGDWU",
                    "STMGPrompt_Full_MSMGDWU_DiffusionHistory",
                    "STMGPrompt_ComponentAblation",
                }
            ),
            "fair_loss_comparison": not bool(config.use_msmg_dwu),
            "component_ablation": config.component_ablation,
            "use_vadsp": bool(config.use_vadsp),
            "use_trend_prior_graph": bool(config.use_trend_prior_graph),
            "use_adaptive_graph": adaptive_enabled,
            "branch_graph_assignment": config.branch_graph_assignment,
            "graph_prior_component": config.graph_prior_component,
            "adaptive_support_mode": config.adaptive_support_mode,
            "graph_rewire_mode": config.graph_rewire_mode,
            "use_macro_prompt": bool(self.coupling_blocks[0].use_macro_prompt),
            "use_cross_fusion": bool(self.coupling_blocks[0].use_cross_fusion),
            "use_st_prompt": bool(config.use_st_prompt),
        }

    def assigned_graph_priors(self) -> tuple[Tensor, Tensor, str, str]:
        assignment = self.config.branch_graph_assignment
        if assignment == "matched":
            return self.A_micro_prior, self.A_macro_prior, "micro", "macro"
        if assignment == "shared_micro":
            return self.A_micro_prior, self.A_micro_prior, "micro", "micro"
        if assignment == "shared_macro":
            return self.A_macro_prior, self.A_macro_prior, "macro", "macro"
        if assignment == "swapped":
            return self.A_macro_prior, self.A_micro_prior, "macro", "micro"
        raise ValueError(f"Unsupported branch_graph_assignment: {assignment}")

    def effective_graphs(self) -> tuple[Tensor, Tensor, dict[str, object]]:
        fine_prior, coarse_prior, fine_id, coarse_id = self.assigned_graph_priors()
        fine_graph = self.micro_graph_builder(fine_prior)
        coarse_graph = self.macro_graph_builder(coarse_prior)
        eps = 1e-8
        trace = {
            "assignment": self.config.branch_graph_assignment,
            "fine_graph_id": fine_id,
            "coarse_graph_id": coarse_id,
            "fine_builder": type(self.micro_graph_builder).__name__,
            "coarse_builder": type(self.macro_graph_builder).__name__,
            "adaptive_support_mode": self.config.adaptive_support_mode,
            "fine_edges_outside_assigned_prior": int(
                torch.logical_and(fine_graph > eps, fine_prior <= eps).sum().detach().cpu().item()
            ),
            "coarse_edges_outside_assigned_prior": int(
                torch.logical_and(coarse_graph > eps, coarse_prior <= eps).sum().detach().cpu().item()
            ),
        }
        return fine_graph, coarse_graph, trace

    @staticmethod
    def _scalar_aux(aux: dict, prefix: str, layer_aux: dict) -> None:
        for key, value in layer_aux.items():
            if torch.is_tensor(value) and value.ndim == 0:
                aux[f"{prefix}_{key}"] = value.detach()
            elif isinstance(value, (bool, float, int)):
                aux[f"{prefix}_{key}"] = value

    def forward(self, x: Tensor | dict[str, Tensor], graph_data=None, aux_data=None) -> dict[str, Tensor | dict]:
        x_tensor = x["x"] if isinstance(x, dict) else x
        if x_tensor.ndim != 4:
            raise ValueError(f"x must be [B,L,N,C], got {tuple(x_tensor.shape)}.")
        B, L, N, C = x_tensor.shape
        if C != self.input_dim:
            raise ValueError(f"Expected C={self.input_dim}, got {C}.")
        if N != self.num_nodes:
            raise ValueError(f"Graph/prompt node count {self.num_nodes} does not match x N={N}.")

        shared = x_tensor.permute(0, 2, 1, 3).reshape(B * N, L, C)
        projected = self.input_proj(shared).reshape(B, N, L, self.hidden_dim).permute(0, 2, 1, 3)
        vadsp_out = self.vadsp(projected, raw_x=x_tensor)
        x_fine = vadsp_out["x_fine"]
        x_coarse = vadsp_out["x_coarse"]
        A_micro, A_macro, graph_assignment_trace = self.effective_graphs()

        layer_auxes = []
        aux: dict = {}
        for layer_idx, block in enumerate(self.coupling_blocks):
            prev_fine, prev_coarse = x_fine, x_coarse
            x_fine, x_coarse, layer_aux = block(
                x_fine,
                x_coarse,
                A_micro if self.config.use_graph_in_temporal_encoder else None,
                A_macro if self.config.use_graph_in_temporal_encoder else None,
            )
            if self.config.diagnostics_level != "none":
                with torch.no_grad():
                    layer_aux["updated_fine_delta_norm"] = float(
                        (x_fine.detach() - prev_fine.detach()).float().norm(dim=-1).mean().cpu().item()
                    )
                    layer_aux["updated_coarse_delta_norm"] = float(
                        (x_coarse.detach() - prev_coarse.detach()).float().norm(dim=-1).mean().cpu().item()
                    )
            layer_auxes.append(layer_aux)
            self._scalar_aux(aux, f"coupling_layer_{layer_idx}", layer_aux)

        st_prompt = (
            self.st_prompt(num_nodes=N, horizon=self.max_pred_len, granularity_index=0)
            if self.st_prompt is not None
            else None
        )
        last_aux = layer_auxes[-1]
        pred_bhn, decoder_aux = self.direct_decoder(
            z_fine=x_fine,
            z_coarse=x_coarse,
            st_prompt=st_prompt,
            macro_prompt=last_aux["macro_prompt"],
            return_aux=True,
        )
        pred_bnh = pred_bhn.transpose(1, 2).contiguous()

        st_prompt_norm_mean = None
        macro_row_sum = None
        micro_row_sum = None
        if self.config.diagnostics_level != "none":
            with torch.no_grad():
                if st_prompt is not None:
                    st_prompt_norm_mean = float(st_prompt.detach().float().norm(dim=-1).mean().cpu().item())
                macro_row_sum = A_macro.detach().float().sum(dim=-1).cpu()
                micro_row_sum = A_micro.detach().float().sum(dim=-1).cpu()

        aux.update(
            {
                "pred_bnh": pred_bnh,
                "model_family": "STMGPrompt",
                "model_name": self.config.model_name,
                "uses_target_mask_as_input": False,
                "uses_vadsp": bool(self.config.use_vadsp),
                "uses_fixed_dual_granularity": self.vadsp.gate_mode == "fixed_dual",
                "vadsp_mode": self.vadsp.gate_mode,
                "uses_trend_prior_graph": bool(self.config.use_trend_prior_graph),
                "uses_adaptive_graph": bool(
                    self.config.use_adaptive_graph and self.config.adaptive_support_mode != "fixed"
                ),
                "uses_graph_temporal_encoder": True,
                "uses_graph_in_temporal_encoder": bool(self.config.use_graph_in_temporal_encoder),
                "uses_temporal_attention": bool(self.config.use_temporal_attention),
                "uses_stmg_coupling_block": True,
                "uses_macro_prompt": bool(last_aux.get("uses_macro_prompt", False)),
                "uses_cross_fusion": bool(last_aux.get("uses_cross_fusion", False)),
                "uses_st_prompt": bool(self.config.use_st_prompt),
                "st_prompt_mode": self.config.st_prompt_mode,
                "st_prompt_use_node_identity": bool(self.config.st_prompt_use_node_identity),
                "st_prompt_use_horizon_identity": bool(self.config.st_prompt_use_horizon_identity),
                "st_prompt_use_shared_horizon_embedding": bool(
                    self.config.st_prompt_use_shared_horizon_embedding
                ),
                "st_prompt_use_type_embedding": bool(self.config.st_prompt_use_type_embedding),
                "st_prompt_type_semantics": self.config.st_prompt_type_semantics,
                "st_prompt_horizon_representation": (
                    self.st_prompt.horizon_representation if self.st_prompt is not None else "not_applicable"
                ),
                "st_prompt_information": self.coupling_metadata["st_prompt_information"],
                "macro_prompt_pooling": last_aux.get("macro_prompt_pooling"),
                "macro_prompt_attention_enabled": last_aux.get("macro_prompt_attention_enabled"),
                "uses_direct_decoder": True,
                "x_fine_coupled": x_fine,
                "x_coarse_coupled": x_coarse,
                "h_fine": last_aux["h_fine"],
                "h_coarse": last_aux["h_coarse"],
                "h_fine_shape": list(last_aux["h_fine"].shape),
                "h_coarse_shape": list(last_aux["h_coarse"].shape),
                "macro_prompt": last_aux["macro_prompt"],
                "macro_prompt_shape": (
                    list(last_aux["macro_prompt"].shape) if last_aux["macro_prompt"] is not None else None
                ),
                "st_prompt": st_prompt.detach() if st_prompt is not None else None,
                "st_prompt_shape": list(st_prompt.shape) if st_prompt is not None else None,
                "st_prompt_norm_mean": st_prompt_norm_mean,
                "h_fine_norm_mean": last_aux["h_fine_norm_mean"],
                "h_fine_norm_std": last_aux["h_fine_norm_std"],
                "h_coarse_norm_mean": last_aux["h_coarse_norm_mean"],
                "h_coarse_norm_std": last_aux["h_coarse_norm_std"],
                "h_fine_spatial_delta_norm": last_aux["h_fine_spatial_delta_norm"],
                "h_coarse_spatial_delta_norm": last_aux["h_coarse_spatial_delta_norm"],
                "h_fine_spatial_delta_ratio": last_aux["h_fine_spatial_delta_ratio"],
                "h_coarse_spatial_delta_ratio": last_aux["h_coarse_spatial_delta_ratio"],
                "macro_prompt_norm_mean": last_aux["macro_prompt_norm_mean"],
                "macro_prompt_pairwise_cosine_mean": last_aux.get("macro_prompt_pairwise_cosine_mean"),
                "macro_prompt_pairwise_cosine_max": last_aux.get("macro_prompt_pairwise_cosine_max"),
                "macro_prompt_temporal_weight_mean": last_aux.get("macro_prompt_temporal_weight_mean"),
                "macro_prompt_source_history_range": last_aux.get("macro_prompt_source_history_range"),
                "macro_prompt_token_count": last_aux.get("macro_prompt_token_count"),
                "macro_attn_entropy": last_aux["macro_attn_entropy"],
                "fine_attn_entropy": last_aux["fine_attn_entropy"],
                "fusion_gate_mean": last_aux["fusion_gate_mean"],
                "fusion_gate_std": last_aux["fusion_gate_std"],
                "fusion_gate_q05": last_aux.get("fusion_gate_q05"),
                "fusion_gate_q25": last_aux.get("fusion_gate_q25"),
                "fusion_gate_q50": last_aux.get("fusion_gate_q50"),
                "fusion_gate_q75": last_aux.get("fusion_gate_q75"),
                "fusion_gate_q95": last_aux.get("fusion_gate_q95"),
                "fusion_gate_saturation_ratio": last_aux.get("fusion_gate_saturation_ratio"),
                "pre_fusion_cosine_similarity": last_aux.get("pre_fusion_cosine_similarity"),
                "post_fusion_cosine_similarity": last_aux.get("post_fusion_cosine_similarity"),
                "fine_representation_shift": last_aux.get("fine_representation_shift"),
                "coarse_representation_shift": last_aux.get("coarse_representation_shift"),
                "macro_to_fine_interaction_increment_norm": last_aux.get(
                    "macro_to_fine_interaction_increment_norm"
                ),
                "fine_to_coarse_interaction_increment_norm": last_aux.get(
                    "fine_to_coarse_interaction_increment_norm"
                ),
                "macro_to_fine_query_source": last_aux.get("macro_to_fine_query_source"),
                "macro_to_fine_key_source": last_aux.get("macro_to_fine_key_source"),
                "macro_to_fine_value_source": last_aux.get("macro_to_fine_value_source"),
                "macro_to_fine_query_history_range": last_aux.get(
                    "macro_to_fine_query_history_range"
                ),
                "fine_to_coarse_query_source": last_aux.get("fine_to_coarse_query_source"),
                "fine_to_coarse_key_source": last_aux.get("fine_to_coarse_key_source"),
                "fine_to_coarse_value_source": last_aux.get("fine_to_coarse_value_source"),
                "fine_to_coarse_history_range": last_aux.get("fine_to_coarse_history_range"),
                "macro_to_fine_attention_shape": last_aux.get("macro_to_fine_attention_shape"),
                "fine_to_coarse_attention_shape": last_aux.get("fine_to_coarse_attention_shape"),
                "macro_to_fine_attention_row_sum_max_error": last_aux.get(
                    "macro_to_fine_attention_row_sum_max_error"
                ),
                "fine_to_coarse_attention_row_sum_max_error": last_aux.get(
                    "fine_to_coarse_attention_row_sum_max_error"
                ),
                "attention_not_applicable_reason": last_aux.get("attention_not_applicable_reason"),
                "gate_input": last_aux.get("gate_input"),
                "gate_level": last_aux.get("gate_level"),
                "gate_range": last_aux.get("gate_range"),
                "gate_initial_bias": last_aux.get("gate_initial_bias"),
                "gate_stop_gradient": last_aux.get("gate_stop_gradient"),
                "cross_fusion_uses_spatial_enhanced_features": bool(
                    last_aux.get("cross_fusion_uses_spatial_enhanced_features", False)
                ),
                "disable_reverse_cross": bool(self.config.disable_reverse_cross),
                "disable_macro_to_fine_cross": bool(self.config.disable_macro_to_fine_cross),
                "macro_to_fine_exclude_recent_len": self.config.macro_to_fine_exclude_recent_len,
                "macro_to_fine_mode": self.config.macro_to_fine_mode,
                "share_cross_attention_projections": bool(
                    self.config.share_cross_attention_projections
                ),
                "macro_prompt_from_spatial_enhanced_coarse": bool(
                    last_aux.get("macro_prompt_from_spatial_enhanced_coarse", False)
                ),
                "temporal_branch_transform": self.config.temporal_branch_transform,
                "coarse_alignment_mode": self.config.coarse_alignment_mode,
                "coarse_upsample_rule": self.config.coarse_upsample_rule,
                "cross_granularity_interaction": self.config.cross_granularity_interaction,
                "direct_concat_mlp_called": bool(last_aux.get("direct_concat_mlp_called", False)),
                "basic_fusion_called": bool(last_aux.get("basic_fusion_called", False)),
                "fine_branch_active": bool(last_aux.get("fine_branch_active", True)),
                "coarse_branch_active": bool(last_aux.get("coarse_branch_active", True)),
                "coupling_metadata": self.coupling_metadata,
                "decoder_metadata": self.direct_decoder.metadata,
                "graph_metadata": self.graph_metadata,
                "branch_graph_assignment": self.config.branch_graph_assignment,
                "fine_graph_id": graph_assignment_trace["fine_graph_id"],
                "coarse_graph_id": graph_assignment_trace["coarse_graph_id"],
                "branch_graph_assignment_trace": graph_assignment_trace,
                "graph_prior_component": self.config.graph_prior_component,
                "adaptive_support_mode": self.config.adaptive_support_mode,
                "graph_rewire_mode": self.config.graph_rewire_mode,
                "graph_operator": self.config.graph_operator,
                "diffusion_direction": self.config.diffusion_direction,
                "diffusion_order_micro": self.config.diffusion_order_micro,
                "diffusion_order_macro": self.config.diffusion_order_macro,
                "diffusion_use_bidirectional": self.config.diffusion_use_bidirectional,
                "diffusion_projection_mode": self.config.diffusion_projection_mode,
                "decoder_context_mode": self.config.decoder_context_mode,
                "diagnostics_level": self.config.diagnostics_level,
                "attention_weights_requested": bool(last_aux.get("attention_weights_requested", False)),
                "entropy_called": bool(last_aux.get("entropy_called", False)),
                "macro_row_sum": macro_row_sum,
                "micro_row_sum": micro_row_sum,
            }
        )
        for key, value in last_aux.items():
            if key.startswith("micro_") or key.startswith("macro_"):
                aux[key] = value
        aux.update(decoder_aux)
        aux.update(vadsp_out)
        if self.config.diagnostics_level in {"none", "minimal"}:
            lean_aux: dict = {}
            for key, value in aux.items():
                if key == "pred_bnh":
                    lean_aux[key] = value
                elif torch.is_tensor(value) and value.ndim == 0:
                    lean_aux[key] = value.detach().cpu().item()
                elif not torch.is_tensor(value):
                    lean_aux[key] = value
            aux = lean_aux
        return {"pred": pred_bhn, "aux": aux}


STMGPromptBase = STMGPromptTemporalOnly
