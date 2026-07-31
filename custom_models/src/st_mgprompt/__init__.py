"""ST-MGPrompt v4.2 coupled skeleton package."""

from .config import COMPONENT_ABLATION_IDS, DEFAULT_16_FEATURES, STMGPromptConfig, apply_component_ablation
from .graph_prior import build_graph_artifacts, prepare_graph_artifacts
from .coupling_block import STMGPromptCouplingBlock
from .decoder import HorizonDirectDecoder, STPromptDirectDecoder, STPromptFullHistoryDecoder
from .graph_layers import AdaptiveGraphBuilder, FixedPriorGraphBuilder, PriorConstrainedDiffusionGraphConv
from .losses import MSMGDWULoss, get_loss_fn, masked_score_aligned_hybrid_loss
from .model import (
    STMGPromptBase,
    STMGPromptTemporalOnly,
    STMGPrompt_FairFull,
    STMGPrompt_GraphTemporalSmoke,
    STMGPrompt_TrendPriorGraph,
)
from .prompt_alignment import MacroTrendPrompt, STPromptEmbedding
from .registry import MODEL_REGISTRY, build_model, list_registered_models
from .temporal_layers import DualGraphTemporalEncoder
from .volatility_patching import VolatilityAwareDynamicSemanticPatching

__all__ = [
    "DEFAULT_16_FEATURES",
    "COMPONENT_ABLATION_IDS",
    "STMGPromptConfig",
    "apply_component_ablation",
    "STMGPromptBase",
    "STMGPromptTemporalOnly",
    "STMGPrompt_FairFull",
    "STMGPrompt_TrendPriorGraph",
    "STMGPrompt_GraphTemporalSmoke",
    "STMGPromptCouplingBlock",
    "STPromptDirectDecoder",
    "STPromptFullHistoryDecoder",
    "HorizonDirectDecoder",
    "AdaptiveGraphBuilder",
    "FixedPriorGraphBuilder",
    "PriorConstrainedDiffusionGraphConv",
    "MSMGDWULoss",
    "get_loss_fn",
    "masked_score_aligned_hybrid_loss",
    "MacroTrendPrompt",
    "STPromptEmbedding",
    "build_graph_artifacts",
    "prepare_graph_artifacts",
    "DualGraphTemporalEncoder",
    "MODEL_REGISTRY",
    "build_model",
    "list_registered_models",
    "VolatilityAwareDynamicSemanticPatching",
]
