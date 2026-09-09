from __future__ import annotations

from dataclasses import dataclass

from .config import STMGPromptConfig
from .model import (
    STMGPromptTemporalOnly,
    STMGPrompt_FairFull,
    STMGPrompt_GraphTemporalSmoke,
    STMGPrompt_TrendPriorGraph,
)


@dataclass(frozen=True)
class ModelRegistryEntry:
    name: str
    implemented: bool
    eligible_for_fair_main_table: bool
    uses_msmg_dwu: bool
    requires_coupling_block: bool
    description: str
    graph_operator: str = "simple"
    diffusion_order_micro: int | None = None
    diffusion_order_macro: int | None = None
    decoder_context_mode: str = "last_state"
    decoder_history_len: int | None = None
    loss_protocol: str = "fair_main"


MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    "STMGPrompt_TemporalOnly": ModelRegistryEntry(
        name="STMGPrompt_TemporalOnly",
        implemented=True,
        eligible_for_fair_main_table=True,
        uses_msmg_dwu=False,
        requires_coupling_block=False,
        description="Step-0 shared-parameter GRU temporal-only baseline.",
    ),
    "STMGPrompt_TemporalOnly_VADSP": ModelRegistryEntry(
        name="STMGPrompt_TemporalOnly_VADSP",
        implemented=True,
        eligible_for_fair_main_table=True,
        uses_msmg_dwu=False,
        requires_coupling_block=False,
        description="Step-1 VADSP + TemporalOnly smoke path; not the FairFull coupled architecture.",
    ),
    "STMGPrompt_DynamicPatching": ModelRegistryEntry(
        name="STMGPrompt_DynamicPatching",
        implemented=True,
        eligible_for_fair_main_table=False,
        uses_msmg_dwu=False,
        requires_coupling_block=False,
        description="Step-1 dynamic patching ablation alias for the VADSP smoke path.",
    ),
    "STMGPrompt_FairFull": ModelRegistryEntry(
        name="STMGPrompt_FairFull",
        implemented=True,
        eligible_for_fair_main_table=True,
        uses_msmg_dwu=False,
        requires_coupling_block=True,
        description="Step-4 VADSP + TrendPriorDualGraph + ST-MG Coupling Block + direct ST-prompt head.",
    ),
    "STMGPrompt_Full_MSMGDWU": ModelRegistryEntry(
        name="STMGPrompt_Full_MSMGDWU",
        implemented=True,
        eligible_for_fair_main_table=False,
        uses_msmg_dwu=True,
        requires_coupling_block=True,
        description="Method-full model with MS-MG-DWU, not eligible for fair main table.",
        loss_protocol="method_full",
    ),
    "STMGPrompt_ComponentAblation": ModelRegistryEntry(
        name="STMGPrompt_ComponentAblation",
        implemented=True,
        eligible_for_fair_main_table=False,
        uses_msmg_dwu=False,
        requires_coupling_block=True,
        description="Formal Fixed-Dual A0-A9 component-ablation architecture; loss is variant-controlled.",
        graph_operator="bidirectional_diffusion",
        diffusion_order_micro=2,
        diffusion_order_macro=2,
        loss_protocol="method_full",
    ),
    "STMGPrompt_FairFull_Diffusion": ModelRegistryEntry(
        name="STMGPrompt_FairFull_Diffusion",
        implemented=True,
        eligible_for_fair_main_table=True,
        uses_msmg_dwu=False,
        requires_coupling_block=True,
        description="FairFull with only the bidirectional diffusion graph operator enabled.",
        graph_operator="bidirectional_diffusion",
        diffusion_order_micro=2,
        diffusion_order_macro=2,
    ),
    "STMGPrompt_FairFull_HistoryDecoder": ModelRegistryEntry(
        name="STMGPrompt_FairFull_HistoryDecoder",
        implemented=True,
        eligible_for_fair_main_table=True,
        uses_msmg_dwu=False,
        requires_coupling_block=True,
        description="FairFull with only the full-history ST-prompt cross-attention decoder enabled.",
        decoder_context_mode="full_history_cross_attention",
        decoder_history_len=None,
    ),
    "STMGPrompt_FairFull_DiffusionHistory": ModelRegistryEntry(
        name="STMGPrompt_FairFull_DiffusionHistory",
        implemented=True,
        eligible_for_fair_main_table=True,
        uses_msmg_dwu=False,
        requires_coupling_block=True,
        description="FairFull with bidirectional diffusion graph and full-history decoder enabled.",
        graph_operator="bidirectional_diffusion",
        diffusion_order_micro=2,
        diffusion_order_macro=2,
        decoder_context_mode="full_history_cross_attention",
        decoder_history_len=None,
    ),
    "STMGPrompt_Full_MSMGDWU_DiffusionHistory": ModelRegistryEntry(
        name="STMGPrompt_Full_MSMGDWU_DiffusionHistory",
        implemented=True,
        eligible_for_fair_main_table=False,
        uses_msmg_dwu=True,
        requires_coupling_block=True,
        description="Method-full MS-MG-DWU model with diffusion graph and full-history decoder.",
        graph_operator="bidirectional_diffusion",
        diffusion_order_micro=2,
        diffusion_order_macro=2,
        decoder_context_mode="full_history_cross_attention",
        decoder_history_len=None,
        loss_protocol="method_full",
    ),
    "STMGPrompt_SerialGraphThenFusion": ModelRegistryEntry(
        name="STMGPrompt_SerialGraphThenFusion",
        implemented=False,
        eligible_for_fair_main_table=False,
        uses_msmg_dwu=False,
        requires_coupling_block=False,
        description="Serial ablation only; never the FairFull default.",
    ),
    "STMGPrompt_CouplingCrossFusion": ModelRegistryEntry(
        name="STMGPrompt_CouplingCrossFusion",
        implemented=True,
        eligible_for_fair_main_table=False,
        uses_msmg_dwu=False,
        requires_coupling_block=True,
        description="Step-4/5 coupling and cross-fusion ablation using the ST-Prompt direct decoder.",
    ),
    "STMGPrompt_TrendPriorGraph": ModelRegistryEntry(
        name="STMGPrompt_TrendPriorGraph",
        implemented=True,
        eligible_for_fair_main_table=False,
        uses_msmg_dwu=False,
        requires_coupling_block=False,
        description="Step-2 VADSP + Trend-Prior Adaptive Dual Graph smoke ablation; not the FairFull model.",
    ),
    "STMGPrompt_GraphTemporalSmoke": ModelRegistryEntry(
        name="STMGPrompt_GraphTemporalSmoke",
        implemented=True,
        eligible_for_fair_main_table=False,
        uses_msmg_dwu=False,
        requires_coupling_block=False,
        description="Step-3 VADSP + adaptive dual graph + Graph-Temporal Encoder smoke ablation.",
    ),
}


def list_registered_models() -> list[dict]:
    return [entry.__dict__ for entry in MODEL_REGISTRY.values()]


def build_model(config: STMGPromptConfig, input_dim: int, graph_data: dict | None = None):
    entry = MODEL_REGISTRY.get(config.model_name)
    if entry is None:
        raise KeyError(f"Unknown ST-MGPrompt model: {config.model_name}")
    if not entry.implemented:
        raise NotImplementedError(
            f"{config.model_name} is registered for later steps but not implemented in Step 0."
        )
    if config.model_name in {"STMGPrompt_TemporalOnly", "STMGPrompt_TemporalOnly_VADSP", "STMGPrompt_DynamicPatching"}:
        return STMGPromptTemporalOnly(config, input_dim=input_dim)
    if config.model_name == "STMGPrompt_TrendPriorGraph":
        return STMGPrompt_TrendPriorGraph(config, input_dim=input_dim, graph_data=graph_data)
    if config.model_name == "STMGPrompt_GraphTemporalSmoke":
        return STMGPrompt_GraphTemporalSmoke(config, input_dim=input_dim, graph_data=graph_data)
    if config.model_name in {
        "STMGPrompt_FairFull",
        "STMGPrompt_FairFull_Diffusion",
        "STMGPrompt_FairFull_HistoryDecoder",
        "STMGPrompt_FairFull_DiffusionHistory",
        "STMGPrompt_CouplingCrossFusion",
        "STMGPrompt_Full_MSMGDWU",
        "STMGPrompt_Full_MSMGDWU_DiffusionHistory",
        "STMGPrompt_ComponentAblation",
    }:
        return STMGPrompt_FairFull(config, input_dim=input_dim, graph_data=graph_data)
    raise NotImplementedError(f"No builder for {config.model_name}.")
