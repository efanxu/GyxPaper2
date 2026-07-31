from __future__ import annotations

from .tslib_channel import TSLibPowerChannelAdapter


class DLinearAdapter(TSLibPowerChannelAdapter):
    model_id = "dlinear"
