from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

from ..upstream import load_tslib_model_class


def create_model(config: Mapping[str, Any] | None = None, protocol=None):
    values = dict(config or {})
    model_class, source = load_tslib_model_class("micn")
    model = model_class(
        SimpleNamespace(**values),
        conv_kernel=list(values["conv_kernel"]),
    )
    model._benchmark_v2_upstream_provenance = source.to_dict()
    return model
