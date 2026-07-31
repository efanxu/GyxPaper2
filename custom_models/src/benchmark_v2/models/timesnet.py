from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import torch

from ..upstream import load_tslib_model_class


def _amp_safe_fft_for_period(x, k=2):
    with torch.autocast(device_type=x.device.type, enabled=False):
        xf = torch.fft.rfft(x.float(), dim=1)
        frequency_list = xf.abs().mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        weights = xf.abs().mean(-1)[:, top_list]
    return period, weights


def create_model(config: Mapping[str, Any] | None = None, protocol=None):
    model_class, source = load_tslib_model_class("timesnet")
    model_class.__init__.__globals__["FFT_for_Period"] = _amp_safe_fft_for_period
    model = model_class(SimpleNamespace(**dict(config or {})))
    model._benchmark_v2_amp_fft_policy = "FFT FP32; convolution/backward retain AMP"
    model._benchmark_v2_upstream_provenance = source.to_dict()
    return model
