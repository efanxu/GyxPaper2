"""Lazy adapter compatibility exports.

Importing ``benchmark_v2.adapters`` must not import every adapter.  The
allowlist below keeps legacy imports working while resolving only the named
symbol on demand.
"""

from __future__ import annotations

import importlib


_EXPORTS: dict[str, tuple[str, str]] = {
    "adapter_forward": ("benchmark_v2.adapters.base", "adapter_forward"),
    "NodeSharedAdapter": ("benchmark_v2.adapters.node_shared", "NodeSharedAdapter"),
    "NativeSpatiotemporalAdapter": ("benchmark_v2.adapters.native_spatiotemporal", "NativeSpatiotemporalAdapter"),
    "NodeSharedGRUAdapter": ("benchmark_v2.adapters.gru", "NodeSharedGRUAdapter"),
    "DLinearAdapter": ("benchmark_v2.adapters.dlinear", "DLinearAdapter"),
    "TransformerAdapter": ("benchmark_v2.adapters.e2_a", "TransformerAdapter"),
    "PatchTSTAdapter": ("benchmark_v2.adapters.e2_a", "PatchTSTAdapter"),
    "ITransformerAdapter": ("benchmark_v2.adapters.e2_a", "ITransformerAdapter"),
    "TimeXerAdapter": ("benchmark_v2.adapters.e2_a", "TimeXerAdapter"),
    "LightTSAdapter": ("benchmark_v2.adapters.lightts", "LightTSAdapter"),
    "TiDEAdapter": ("benchmark_v2.adapters.tide", "TiDEAdapter"),
    "SegRNNAdapter": ("benchmark_v2.adapters.segrnn", "SegRNNAdapter"),
    "TimesNetAdapter": ("benchmark_v2.adapters.e2_b", "TimesNetAdapter"),
    "MICNAdapter": ("benchmark_v2.adapters.e2_b", "MICNAdapter"),
    "WPMixerAdapter": ("benchmark_v2.adapters.e2_b", "WPMixerAdapter"),
    "MultiPatchFormerAdapter": ("benchmark_v2.adapters.e2_b", "MultiPatchFormerAdapter"),
    "TimeMixerAdapter": ("benchmark_v2.adapters.e2_c", "TimeMixerAdapter"),
    "TSMixerAdapter": ("benchmark_v2.adapters.e2_c", "TSMixerAdapter"),
    "FreTSAdapter": ("benchmark_v2.adapters.e2_c", "FreTSAdapter"),
    "CrossformerAdapter": ("benchmark_v2.adapters.e2_d", "CrossformerAdapter"),
    "MSGNetAdapter": ("benchmark_v2.adapters.e2_d", "MSGNetAdapter"),
    "TimeFilterAdapter": ("benchmark_v2.adapters.e2_d", "TimeFilterAdapter"),
    "NativeGraphAdapter": ("benchmark_v2.adapters.e3_b", "NativeGraphAdapter"),
    "NativeAdaptiveNodeAdapter": ("benchmark_v2.adapters.e3_c", "NativeAdaptiveNodeAdapter"),
    "BaselineScalerContext": ("benchmark_v2.adapters.statistical_baselines", "BaselineScalerContext"),
    "StatisticalBaselineAdapter": ("benchmark_v2.adapters.statistical_baselines", "StatisticalBaselineAdapter"),
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
