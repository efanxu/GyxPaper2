"""Lazy graph-model exports.

Graph implementations are independent source-identity units.  Importing the
package for one native graph model must not import every other graph model.
The compatibility exports below are a fixed, auditable map.
"""

from __future__ import annotations

import importlib


_EXPORTS: dict[str, tuple[str, str]] = {
    "GCNLayer": ("benchmark_v2.models.graph_models.gcn", "GCNLayer"),
    "GCNForecast": ("benchmark_v2.models.graph_models.gcn", "GCNForecast"),
    "TemporalGLUConv": ("benchmark_v2.models.graph_models.stgcn", "TemporalGLUConv"),
    "ChebyshevGraphConv": ("benchmark_v2.models.graph_models.stgcn", "ChebyshevGraphConv"),
    "STConvBlock": ("benchmark_v2.models.graph_models.stgcn", "STConvBlock"),
    "STGCNForecast": ("benchmark_v2.models.graph_models.stgcn", "STGCNForecast"),
    "chebyshev_basis": ("benchmark_v2.models.graph_models.stgcn", "chebyshev_basis"),
    "DiffusionLinear": ("benchmark_v2.models.graph_models.dcrnn", "DiffusionLinear"),
    "DCGRUCell": ("benchmark_v2.models.graph_models.dcrnn", "DCGRUCell"),
    "DCRNNForecast": ("benchmark_v2.models.graph_models.dcrnn", "DCRNNForecast"),
    "E3_C_POLICIES": ("benchmark_v2.models.graph_models.adaptive_common", "E3_C_POLICIES"),
    "canonical_tensor_hash": ("benchmark_v2.models.graph_models.adaptive_common", "canonical_tensor_hash"),
    "e3_c_graph_identity": ("benchmark_v2.models.graph_models.adaptive_common", "e3_c_graph_identity"),
    "learned_graph_summary": ("benchmark_v2.models.graph_models.adaptive_common", "learned_graph_summary"),
    "DiffusionGraphConv": ("benchmark_v2.models.graph_models.graph_wavenet", "DiffusionGraphConv"),
    "GraphWaveNetForecast": ("benchmark_v2.models.graph_models.graph_wavenet", "GraphWaveNetForecast"),
    "graph_propagate": ("benchmark_v2.models.graph_models.graph_wavenet", "graph_propagate"),
    "DirectedGraphConstructor": ("benchmark_v2.models.graph_models.mtgnn", "DirectedGraphConstructor"),
    "DilatedInception": ("benchmark_v2.models.graph_models.mtgnn", "DilatedInception"),
    "MixProp": ("benchmark_v2.models.graph_models.mtgnn", "MixProp"),
    "MTGNNForecast": ("benchmark_v2.models.graph_models.mtgnn", "MTGNNForecast"),
    "AdaptiveGraphConv": ("benchmark_v2.models.graph_models.agcrn", "AdaptiveGraphConv"),
    "AGCRNCell": ("benchmark_v2.models.graph_models.agcrn", "AGCRNCell"),
    "AGCRNForecast": ("benchmark_v2.models.graph_models.agcrn", "AGCRNForecast"),
    "ResidualMLP": ("benchmark_v2.models.graph_models.stid", "ResidualMLP"),
    "STIDForecast": ("benchmark_v2.models.graph_models.stid", "STIDForecast"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(_EXPORTS)
