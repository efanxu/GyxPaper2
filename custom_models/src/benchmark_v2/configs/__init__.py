"""Lazy configuration compatibility exports.

Configuration modules are model identity inputs and must not all be imported
when one model is selected.  This module exposes the historical symbols via a
fixed allowlist and resolves only the requested module.
"""

from __future__ import annotations

import importlib


_EXPORTS: dict[str, tuple[str, str]] = {}
for _module, _names in {
    "dlinear": ("DLINEAR_CONFIG", "resolve_dlinear_config"),
    "gru": ("GRU_CONFIG", "resolve_gru_config"),
    "lightts": ("LIGHTTS_CONFIG", "resolve_lightts_config"),
    "itransformer": ("ITRANSFORMER_CONFIG", "resolve_itransformer_config"),
    "moving_average": ("MOVING_AVERAGE_CONFIG", "resolve_moving_average_config"),
    "patchtst": ("PATCHTST_CONFIG", "resolve_patchtst_config"),
    "persistence": ("PERSISTENCE_CONFIG", "resolve_persistence_config"),
    "segrnn": ("SEGRNN_CONFIG", "resolve_segrnn_config"),
    "tide": ("TIDE_CONFIG", "resolve_tide_config"),
    "timexer": ("TIMEXER_CONFIG", "resolve_timexer_config"),
    "transformer": ("TRANSFORMER_CONFIG", "resolve_transformer_config"),
    "timesnet": ("TIMESNET_CONFIG", "resolve_timesnet_config"),
    "micn": ("MICN_CONFIG", "resolve_micn_config"),
    "wpmixer": ("WPMIXER_CONFIG", "resolve_wpmixer_config"),
    "multipatchformer": ("MULTIPATCHFORMER_CONFIG", "resolve_multipatchformer_config"),
    "timemixer": ("TIMEMIXER_CONFIG", "resolve_timemixer_config"),
    "tsmixer": ("TSMIXER_CONFIG", "resolve_tsmixer_config"),
    "frets": ("FRETS_CONFIG", "resolve_frets_config"),
    "crossformer": ("CROSSFORMER_CONFIG", "resolve_crossformer_config"),
    "msgnet": ("MSGNET_CONFIG", "resolve_msgnet_config"),
    "timefilter": ("TIMEFILTER_CONFIG", "resolve_timefilter_config"),
    "gcn": ("GCN_CONFIG", "resolve_gcn_config"),
    "stgcn": ("STGCN_CONFIG", "resolve_stgcn_config"),
    "dcrnn": ("DCRNN_CONFIG", "resolve_dcrnn_config"),
    "graph_wavenet": ("GRAPH_WAVENET_CONFIG", "resolve_graph_wavenet_config"),
    "mtgnn": ("MTGNN_CONFIG", "resolve_mtgnn_config"),
    "agcrn": ("AGCRN_CONFIG", "resolve_agcrn_config"),
    "stid": ("STID_CONFIG", "resolve_stid_config"),
}.items():
    for _name in _names:
        _EXPORTS[_name] = (f"benchmark_v2.configs.{_module}", _name)
del _module, _names, _name


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
