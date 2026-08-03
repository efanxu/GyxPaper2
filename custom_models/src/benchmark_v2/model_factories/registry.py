from __future__ import annotations

import importlib
from types import ModuleType

from ..errors import ModelUnavailableError


# This is intentionally a literal, reviewed allowlist.  The value is never
# constructed from user input; callers can select only a canonical id that is
# already present in the benchmark registry.
_FACTORY_MODULES: dict[str, str] = {
    "persistence": "benchmark_v2.model_factories.persistence",
    "moving_average": "benchmark_v2.model_factories.moving_average",
    "gru": "benchmark_v2.model_factories.gru",
    "dlinear": "benchmark_v2.model_factories.dlinear",
    "lightts": "benchmark_v2.model_factories.lightts",
    "tide": "benchmark_v2.model_factories.tide",
    "segrnn": "benchmark_v2.model_factories.segrnn",
    "transformer": "benchmark_v2.model_factories.transformer",
    "patchtst": "benchmark_v2.model_factories.patchtst",
    "itransformer": "benchmark_v2.model_factories.itransformer",
    "timexer": "benchmark_v2.model_factories.timexer",
    "timesnet": "benchmark_v2.model_factories.timesnet",
    "micn": "benchmark_v2.model_factories.micn",
    "wpmixer": "benchmark_v2.model_factories.wpmixer",
    "multipatchformer": "benchmark_v2.model_factories.multipatchformer",
    "timemixer": "benchmark_v2.model_factories.timemixer",
    "tsmixer": "benchmark_v2.model_factories.tsmixer",
    "frets": "benchmark_v2.model_factories.frets",
    "crossformer": "benchmark_v2.model_factories.crossformer",
    "msgnet": "benchmark_v2.model_factories.msgnet",
    "timefilter": "benchmark_v2.model_factories.timefilter",
    "gcn": "benchmark_v2.model_factories.gcn",
    "stgcn": "benchmark_v2.model_factories.stgcn",
    "dcrnn": "benchmark_v2.model_factories.dcrnn",
    "graph_wavenet": "benchmark_v2.model_factories.graph_wavenet",
    "mtgnn": "benchmark_v2.model_factories.mtgnn",
    "agcrn": "benchmark_v2.model_factories.agcrn",
    "stid": "benchmark_v2.model_factories.stid",
}


def factory_module_path(model_id: str) -> str:
    """Return the fixed factory module path or fail closed."""

    try:
        return _FACTORY_MODULES[str(model_id)]
    except KeyError as exc:
        raise ModelUnavailableError(
            f"No audited model factory exists for {model_id!r}."
        ) from exc


def get_model_factory(model_id: str) -> ModuleType:
    """Load exactly one allowlisted factory module for ``model_id``."""

    module_name = factory_module_path(model_id)
    module = importlib.import_module(module_name)
    if not callable(getattr(module, "resolve_config", None)):
        raise ModelUnavailableError(
            f"Audited factory {module_name} has no resolve_config function."
        )
    if not callable(getattr(module, "create_adapter", None)):
        raise ModelUnavailableError(
            f"Audited factory {module_name} has no create_adapter function."
        )
    if not callable(getattr(module, "create_model", None)):
        raise ModelUnavailableError(
            f"Audited factory {module_name} has no create_model function."
        )
    return module


__all__ = ["factory_module_path", "get_model_factory"]
