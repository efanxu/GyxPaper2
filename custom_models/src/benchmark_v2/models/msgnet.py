from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import torch

from ..upstream import load_tslib_model_class


def _amp_safe_traced_fft_for_period(x, k=2):
    with torch.autocast(device_type=x.device.type, enabled=False):
        xf = torch.fft.rfft(x.float(), dim=1)
        frequency_list = xf.abs().mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_numpy = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_numpy
        weights = xf.abs().mean(-1)[:, top_list]
    _amp_safe_traced_fft_for_period.trace.append(
        {
            "input_shape": [int(v) for v in x.shape],
            "frequency_indices": [int(v) for v in top_list.detach().cpu()],
            "periods": [int(v) for v in period],
            "scale_weight_shape": [int(v) for v in weights.shape],
            "scale_weights_raw": [
                [float(value) for value in row]
                for row in weights.detach().cpu()
            ],
            "scale_weights_softmax": [
                [float(value) for value in row]
                for row in torch.softmax(weights.detach().float(), dim=1).cpu()
            ],
            "fft_dtype": str(xf.dtype),
        }
    )
    return period, weights


def create_model(config: Mapping[str, Any] | None = None, protocol=None):
    model_class, source = load_tslib_model_class("msgnet")
    module_globals = model_class.__init__.__globals__
    module_globals["FFT_for_Period"] = _amp_safe_traced_fft_for_period
    _amp_safe_traced_fft_for_period.trace = []
    model = model_class(SimpleNamespace(**dict(config or {})))
    model._benchmark_v2_fft_trace_function = _amp_safe_traced_fft_for_period
    model._benchmark_v2_amp_fft_policy = (
        "FFT and frequency statistics FP32; graph/attention/backward retain AMP"
    )
    model._benchmark_v2_upstream_provenance = source.to_dict()
    return model


_amp_safe_traced_fft_for_period.trace = []
