from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import torch.nn as nn

from ..upstream import load_tslib_model_class


def create_model(config: Mapping[str, Any] | None = None, protocol=None):
    values = dict(config or {})
    model_class, source = load_tslib_model_class("multipatchformer")
    model = model_class(SimpleNamespace(**values))
    model.padding_patch_layer3 = nn.Identity()
    model.padding_patch_layer4 = nn.ReplicationPad1d(
        (0, int(values["patch_paddings"][3]))
    )
    model.stride3 = int(values["patch_strides"][2])
    model.stride4 = int(values["patch_strides"][3])
    model.embedding_patch_4.stride = (model.stride4,)
    chunk = int(values["pred_len"]) // 8
    final_chunk = int(values["pred_len"]) - 7 * chunk
    width = int(values["d_model"])
    model.out_linear_5 = nn.Linear(width + 4 * chunk, chunk)
    model.out_linear_6 = nn.Linear(width + 5 * chunk, chunk)
    model.out_linear_7 = nn.Linear(width + 6 * chunk, chunk)
    model.out_linear_8 = nn.Linear(width + 7 * chunk, final_chunk)
    model._benchmark_v2_patch_compatibility = dict(
        values["compatibility_adjustments"]
    )
    model._benchmark_v2_upstream_provenance = source.to_dict()
    return model
