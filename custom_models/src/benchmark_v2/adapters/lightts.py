from __future__ import annotations

from .tslib_channel import TSLibPowerChannelAdapter


class LightTSAdapter(TSLibPowerChannelAdapter):
    model_id = "lightts"
