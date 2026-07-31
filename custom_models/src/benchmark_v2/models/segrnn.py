from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

from ..upstream import load_tslib_model_class


def create_model(
    config: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
):
    values = dict(config or {})
    model_class, source = load_tslib_model_class("segrnn")
    model = model_class(SimpleNamespace(**values))
    model._benchmark_v2_upstream_provenance = source.to_dict()
    return model
