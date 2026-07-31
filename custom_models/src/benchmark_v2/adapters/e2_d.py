from __future__ import annotations

from typing import Any, Mapping

from ..contracts import BenchmarkOutput
from ..errors import ContractError
from .tslib_channel import TSLibPowerChannelAdapter


class _E2DHistoryOnlyAdapter(TSLibPowerChannelAdapter):
    model_id = "e2_d"

    def _reject_future_inputs(self, batch, kwargs: Mapping[str, Any]) -> None:
        import torch

        if batch.x_mark is not None or batch.y_mark is not None:
            raise ContractError(
                f"{self.model_id} disables supplied calendar/time marks."
            )
        for key in (
            "future_observed_covariates",
            "future_exogenous",
            "future_weather",
            "future_calendar_marks",
        ):
            value = kwargs.get(key)
            if value is not None and bool(torch.any(torch.as_tensor(value) != 0)):
                raise ContractError(f"{self.model_id} rejects non-zero {key}.")

    def prepare_model_inputs(self, batch, **kwargs: Any):
        self._reject_future_inputs(batch, kwargs)
        return super().prepare_model_inputs(batch, **kwargs)


class CrossformerAdapter(_E2DHistoryOnlyAdapter):
    model_id = "crossformer"

    def normalize_output(self, raw_output, batch, **kwargs: Any) -> BenchmarkOutput:
        output = super().normalize_output(raw_output, batch, **kwargs)
        b, _, n, _ = (int(v) for v in batch.x.shape)
        output.aux["shape_trace"] = {
            "embedded": [b * n, 16, 12, 32],
            "encoder_outputs": [
                [b * n, 16, 12, 32],
                [b * n, 16, 12, 32],
                [b * n, 16, 6, 32],
            ],
            "decoder_position_tokens": [b * n, 16, 1, 32],
            "decoder_layer_outputs": [[b * n, 16, 1, 32]] * 3,
            "decoder_layer_predictions": [[b * n, 16, 1, 12]] * 3,
            "prediction_sum": [b * n, 12, 16],
            "upstream_slice": "dec_out[:, -pred_len:, :]",
            "native_return": [b * n, 10, 16],
            "adapter_horizon_crop": False,
        }
        return output


class MSGNetAdapter(_E2DHistoryOnlyAdapter):
    model_id = "msgnet"

    def __init__(self, protocol: Mapping[str, Any]):
        super().__init__(protocol)
        self.last_period_trace: list[dict[str, Any]] = []

    def forward_model(self, model, model_inputs: Any, **kwargs: Any):
        import torch

        trace_function = model._benchmark_v2_fft_trace_function
        trace_function.trace = []
        outputs = []
        for index in range(int(model_inputs.shape[0])):
            outputs.append(model(model_inputs[index:index + 1], None, None, None))
        self.last_period_trace = list(
            trace_function.trace
        )
        return torch.cat(outputs, dim=0)

    def normalize_output(self, raw_output, batch, **kwargs: Any) -> BenchmarkOutput:
        output = super().normalize_output(raw_output, batch, **kwargs)
        periods = [
            value
            for trace in self.last_period_trace
            for value in trace["periods"]
        ]
        output.aux["period_graph_trace"] = {
            "calls_per_flattened_item": True,
            "shared_model_instances": 1,
            "fft_batch_scope": 1,
            "fft_calls": len(self.last_period_trace),
            "traces": self.last_period_trace,
            "periods": periods,
            "branch_padding": [
                (144 if 144 % period == 0 else ((144 // period) + 1) * period)
                - 144
                for period in periods
            ],
            "adaptive_adjacency_shape": [16, 16],
            "graph_nodes": "16 within-turbine historical features",
            "crop": "each branch out[:, :seq_len, :]",
            "scale_aggregation": "softmax FFT weights then weighted sum",
        }
        return output


class TimeFilterAdapter(_E2DHistoryOnlyAdapter):
    model_id = "timefilter"

    def normalize_output(self, raw_output, batch, **kwargs: Any) -> BenchmarkOutput:
        output = super().normalize_output(raw_output, batch, **kwargs)
        b, _, n, _ = (int(v) for v in batch.x.shape)
        output.aux["token_mask_trace"] = {
            "flattened_items": b * n,
            "patch_len": 16,
            "patches_per_variable": 9,
            "token_count": 144,
            "token_mapping": (
                "feature-major flatten (16 variables x 144 steps), then "
                "non-overlapping 16-step patches"
            ),
            "embedded_shape": [b * n, 144, 512],
            "mask_shape": [144, 3, 144],
            "mask_regions": ["S:same patch offset across variables",
                             "T:same variable across time patches",
                             "ST:remaining cross-variable/time"],
            "graph_adjacency_shape": [b * n, 8, 144, 144],
            "alpha": 0.1,
            "top_p": 0.5,
            "noisy_gating_train_only": True,
            "eval_deterministic": True,
            "moe_auxiliary_loss_returned_by_backbone": True,
            "moe_auxiliary_loss_used_for_masked_mse": False,
        }
        return output
