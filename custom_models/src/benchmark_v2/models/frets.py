from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

from ..upstream import load_tslib_model_class


def create_model(config: Mapping[str, Any] | None = None, protocol=None):
    values = dict(config or {})
    if values.get("channel_independence") != "0":
        raise ValueError(
            "benchmark_v2 FreTS requires channel_independence as string '0' "
            "to activate the audited channel-frequency branch."
        )
    model_class, source = load_tslib_model_class("frets")
    model = model_class(SimpleNamespace(**values))
    model._benchmark_v2_amp_forward_policy = (
        "Native Trainer autocast; isolated raw-source CUDA AMP passed"
    )
    model._benchmark_v2_upstream_provenance = source.to_dict()
    return model
