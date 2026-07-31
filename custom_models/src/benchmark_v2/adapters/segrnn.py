from __future__ import annotations

from .tslib_channel import TSLibPowerChannelAdapter


class SegRNNAdapter(TSLibPowerChannelAdapter):
    model_id = "segrnn"

    def normalize_output(self, raw_output, batch, **kwargs):
        output = super().normalize_output(raw_output, batch, **kwargs)
        output.semantic_trace.update(
            {
                "effective_seg_len": 2,
                "seg_num_x": 72,
                "seg_num_y": 5,
                "shape_fix": "protocol_divisibility_constraint",
            }
        )
        return output
