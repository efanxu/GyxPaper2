from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

from ..upstream import load_tslib_model_class


def create_model(config: Mapping[str, Any] | None = None, protocol=None):
    model_class, source = load_tslib_model_class("crossformer")
    model = model_class(SimpleNamespace(**dict(config or {})))
    model._benchmark_v2_upstream_provenance = source.to_dict()
    return model
