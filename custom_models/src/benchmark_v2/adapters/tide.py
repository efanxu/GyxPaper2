from __future__ import annotations

from typing import Any

from ..errors import ContractError
from .tslib_channel import TSLibPowerChannelAdapter


class TiDEAdapter(TSLibPowerChannelAdapter):
    model_id = "tide"

    def __init__(self, protocol):
        super().__init__(protocol)
        self.last_placeholder_trace: dict[str, Any] | None = None

    def prepare_model_inputs(self, batch, **kwargs: Any):
        import torch

        if batch.x_mark is not None or batch.y_mark is not None:
            raise ContractError(
                "TiDE formal protocol disables supplied calendar/time marks."
            )
        future_observed = kwargs.get("future_observed_covariates")
        if future_observed is not None:
            values = (
                future_observed.detach()
                if hasattr(future_observed, "detach")
                else torch.as_tensor(future_observed)
            )
            if bool(torch.any(values != 0)):
                raise ContractError(
                    "TiDE rejects non-zero future observed covariates."
                )
        flat = super().prepare_model_inputs(batch, **kwargs)
        feature_dim = 4
        x_mark_enc = torch.zeros(
            flat.shape[0],
            int(self.protocol["lookback"]),
            feature_dim,
            dtype=flat.dtype,
            device=flat.device,
        )
        y_mark = torch.zeros(
            flat.shape[0],
            int(self.protocol["max_pred_len"]),
            feature_dim,
            dtype=flat.dtype,
            device=flat.device,
        )
        x_dec = torch.zeros(
            flat.shape[0],
            0,
            int(self.protocol["feature_count"]),
            dtype=flat.dtype,
            device=flat.device,
        )
        self.last_placeholder_trace = {
            "x_mark_enc_shape": tuple(int(v) for v in x_mark_enc.shape),
            "y_mark_shape": tuple(int(v) for v in y_mark.shape),
            "x_dec_shape": tuple(int(v) for v in x_dec.shape),
            "x_mark_enc_all_zero": bool(torch.count_nonzero(x_mark_enc) == 0),
            "y_mark_all_zero": bool(torch.count_nonzero(y_mark) == 0),
            "x_dec_all_zero": bool(torch.count_nonzero(x_dec) == 0),
        }
        return flat, x_mark_enc, x_dec, y_mark

    def forward_model(self, model, model_inputs: Any, **kwargs: Any):
        x_enc, x_mark_enc, x_dec, y_mark = model_inputs
        return model(x_enc, x_mark_enc, x_dec, y_mark)

    def normalize_output(self, raw_output, batch, **kwargs):
        output = super().normalize_output(raw_output, batch, **kwargs)
        output.aux["zero_placeholder_trace"] = dict(
            self.last_placeholder_trace or {}
        )
        output.semantic_trace.update(
            {
                "uses_zero_placeholders": True,
                "zero_placeholder_covariates": True,
                "effective_future_covariates": "none",
                "uses_future_target": False,
                "uses_future_observed_covariates": False,
                "uses_future_calendar_covariates": False,
            }
        )
        return output
