from __future__ import annotations

from typing import Any

from .constants import CANONICAL_ROOT, PROJECT_ROOT
from .io_utils import read_json

YES, NO, NA, NX, NV = "YES", "NO", "NOT_APPLICABLE", "NOT_EXTRACTABLE", "NOT_VERIFIED"


def _row(model_id: str, display_name: str, **values: Any) -> dict[str, Any]:
    base = {
        "model_id": model_id,
        "display_name": display_name,
        "model_family": NV,
        "temporal_backbone": NV,
        "uses_physical_prior_graph": NV,
        "uses_fixed_graph": NV,
        "learns_adaptive_graph": NV,
        "uses_directed_supports": NV,
        "uses_bidirectional_propagation": NV,
        "uses_diffusion_propagation": NV,
        "uses_recurrent_graph_propagation": NV,
        "uses_node_identity_embedding": NV,
        "uses_Graph_Protocol": NV,
        "graph_provider_ID": NV,
        "self_loop_policy": NV,
        "normalization_policy": NV,
        "selected_k": NV,
        "effective_graph_extraction_supported": NV,
        "source_evidence": [],
        "notes": "",
    }
    base.update(values)
    return base


def build_capability_matrix(external_evidence: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    a0_config = read_json(
        PROJECT_ROOT
        / "custom_models"
        / "results"
        / "st_mgprompt_component_ablation"
        / "component_ablation_fixed_dual_seed2026"
        / "A0"
        / "effective_config.json"
    )
    rows = [
        _row(
            "st_mgprompt",
            "ST-MGPrompt A0",
            model_family="joint prior-adaptive graph temporal prompt model",
            temporal_backbone="fine/coarse dilated TCN + cross fusion",
            uses_physical_prior_graph=NO,
            uses_fixed_graph=YES,
            learns_adaptive_graph=YES if a0_config.get("use_adaptive_graph") else NO,
            uses_directed_supports=YES,
            uses_bidirectional_propagation=YES if a0_config.get("diffusion_use_bidirectional") else NO,
            uses_diffusion_propagation=YES if a0_config.get("graph_operator") == "bidirectional_diffusion" else NO,
            uses_recurrent_graph_propagation=NO,
            uses_node_identity_embedding=YES,
            uses_Graph_Protocol=NO,
            graph_provider_ID=a0_config.get("graph_tag", NV),
            self_loop_policy="disabled in prior; adaptive softmax may retain diagonal",
            normalization_policy="prior-constrained adaptive row normalization",
            selected_k=f"macro={a0_config.get('macro_top_k')};micro={a0_config.get('micro_top_k')}",
            effective_graph_extraction_supported=YES,
            source_evidence=[
                "custom_models/src/st_mgprompt/model.py",
                "custom_models/src/st_mgprompt/graph_layers.py",
                "custom_models/results/st_mgprompt_component_ablation/component_ablation_fixed_dual_seed2026/A0/effective_config.json",
            ],
            notes="Uses ST-MGPrompt train-only macro/micro priors, not the benchmark physical-kNN matrix as its forward prior.",
        ),
        _row(
            "gcn", "GCN", model_family="fixed graph convolution", temporal_backbone="shared direct temporal projection",
            uses_physical_prior_graph=YES, uses_fixed_graph=YES, learns_adaptive_graph=NO,
            uses_directed_supports=NO, uses_bidirectional_propagation=NO, uses_diffusion_propagation=NO,
            uses_recurrent_graph_propagation=NO, uses_node_identity_embedding=NO, uses_Graph_Protocol=YES,
            graph_provider_ID="sdwpf_physical_knn_v1:A_gcn", self_loop_policy="provider A_gcn",
            normalization_policy="symmetric normalized adjacency", selected_k=4,
            effective_graph_extraction_supported=YES,
            source_evidence=["custom_models/src/benchmark_v2/models/graph_models/gcn.py"],
        ),
        _row(
            "stgcn", "STGCN", model_family="fixed spatiotemporal graph convolution", temporal_backbone="two T-G-T ST-Conv blocks",
            uses_physical_prior_graph=YES, uses_fixed_graph=YES, learns_adaptive_graph=NO,
            uses_directed_supports=NO, uses_bidirectional_propagation=NO, uses_diffusion_propagation=NO,
            uses_recurrent_graph_propagation=NO, uses_node_identity_embedding=NO, uses_Graph_Protocol=YES,
            graph_provider_ID="sdwpf_physical_knn_v1:L_tilde", self_loop_policy="Chebyshev T0 identity term",
            normalization_policy="scaled symmetric normalized Laplacian", selected_k=4,
            effective_graph_extraction_supported=YES,
            source_evidence=["custom_models/src/benchmark_v2/models/graph_models/stgcn.py"],
        ),
        _row(
            "dcrnn", "DCRNN", model_family="recurrent diffusion graph network", temporal_backbone="two-layer DCGRU encoder/decoder",
            uses_physical_prior_graph=YES, uses_fixed_graph=YES, learns_adaptive_graph=NO,
            uses_directed_supports=YES, uses_bidirectional_propagation=YES, uses_diffusion_propagation=YES,
            uses_recurrent_graph_propagation=YES, uses_node_identity_embedding=NO, uses_Graph_Protocol=YES,
            graph_provider_ID="sdwpf_physical_knn_v1:P_forward+P_reverse", self_loop_policy="T0 diffusion basis",
            normalization_policy="dual random-walk", selected_k=4, effective_graph_extraction_supported=YES,
            source_evidence=["custom_models/src/benchmark_v2/models/graph_models/dcrnn.py"],
        ),
        _row(
            "mtgnn", "MTGNN", model_family="adaptive graph structure learning", temporal_backbone="dilated inception temporal stack",
            uses_physical_prior_graph=NO, uses_fixed_graph=NO, learns_adaptive_graph=YES,
            uses_directed_supports=YES, uses_bidirectional_propagation=YES, uses_diffusion_propagation=YES,
            uses_recurrent_graph_propagation=NO, uses_node_identity_embedding=YES, uses_Graph_Protocol=NO,
            graph_provider_ID="native MTGNN graph constructor", self_loop_policy="native top-k; diagonal is not explicitly masked",
            normalization_policy="directed row top-k MixProp", selected_k=20, effective_graph_extraction_supported=YES,
            source_evidence=["custom_models/src/benchmark_v2/models/graph_models/mtgnn.py"],
            notes="Physical GraphBundle is validated as context but is not used in forward.",
        ),
        _row(
            "stid", "STID", model_family="spatial-temporal identity MLP", temporal_backbone="history projection + residual MLP",
            uses_physical_prior_graph=NA, uses_fixed_graph=NA, learns_adaptive_graph=NA,
            uses_directed_supports=NA, uses_bidirectional_propagation=NA, uses_diffusion_propagation=NA,
            uses_recurrent_graph_propagation=NA, uses_node_identity_embedding=YES, uses_Graph_Protocol=NA,
            graph_provider_ID=NA, self_loop_policy=NA, normalization_policy=NA, selected_k=NA,
            effective_graph_extraction_supported=NA,
            source_evidence=["custom_models/src/benchmark_v2/models/graph_models/stid.py"],
            notes="Graph-free spatial identity control; embedding cosine similarity is not adjacency.",
        ),
        _row(
            "tsmixer", "TSMixer", model_family="graph-free temporal Mixer", temporal_backbone="node-shared TSMixer",
            uses_physical_prior_graph=NA, uses_fixed_graph=NA, learns_adaptive_graph=NA,
            uses_directed_supports=NA, uses_bidirectional_propagation=NA, uses_diffusion_propagation=NA,
            uses_recurrent_graph_propagation=NA, uses_node_identity_embedding=NO, uses_Graph_Protocol=NA,
            graph_provider_ID=NA, self_loop_policy=NA, normalization_policy=NA, selected_k=NA,
            effective_graph_extraction_supported=NA,
            source_evidence=["custom_models/src/benchmark_v2/models/tsmixer.py"],
            notes="Graph-free node-shared temporal/Mixer control.",
        ),
    ]
    return rows
