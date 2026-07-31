from __future__ import annotations

from types import SimpleNamespace
from types import MethodType
from typing import Any, Mapping

import torch

from ..upstream import load_tslib_model_class


def _amp_safe_core_forward(self, xL):
    with torch.autocast(device_type=xL.device.type, enabled=False):
        x = xL.float().transpose(1, 2)
        xA, xD = self.Decomposition_model.transform(x)
    yA = self.resolutionBranch[0](xA)
    yD = [
        self.resolutionBranch[index + 1](value)
        for index, value in enumerate(xD)
    ]
    with torch.autocast(device_type=xL.device.type, enabled=False):
        y = self.Decomposition_model.inv_transform(
            yA.float(), [value.float() for value in yD]
        )
    return y.transpose(1, 2)[:, -self.pred_length :, :]


def create_model(config: Mapping[str, Any] | None = None, protocol=None):
    values = dict(config or {})
    values["device"] = torch.device("cpu")
    model_class, source = load_tslib_model_class("wpmixer")
    model = model_class(
        SimpleNamespace(**values),
        tfactor=int(values["tfactor"]),
        dfactor=int(values["dfactor"]),
        wavelet=str(values["wavelet"]),
        level=int(values["level"]),
        stride=int(values["stride"]),
        no_decomposition=bool(values["no_decomposition"]),
    )
    model.wpmixerCore.forward = MethodType(
        _amp_safe_core_forward, model.wpmixerCore
    )
    model._benchmark_v2_amp_wavelet_policy = (
        "DWT/IDWT FP32; resolution mixers/backward retain Trainer AMP"
    )
    model._benchmark_v2_upstream_provenance = source.to_dict()
    return model
