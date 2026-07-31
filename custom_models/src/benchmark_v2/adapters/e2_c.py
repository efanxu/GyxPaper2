from __future__ import annotations

from typing import Any, Mapping

from ..contracts import BenchmarkOutput
from ..errors import ContractError
from .tslib_channel import TSLibPowerChannelAdapter


class _E2CHistoryOnlyAdapter(TSLibPowerChannelAdapter):
    model_id = "e2_c"

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


class TimeMixerAdapter(_E2CHistoryOnlyAdapter):
    model_id = "timemixer"

    def normalize_output(self, raw_output, batch, **kwargs: Any) -> BenchmarkOutput:
        output = super().normalize_output(raw_output, batch, **kwargs)
        b, _, n, _ = (int(value) for value in batch.x.shape)
        output.aux["scale_trace"] = {
            "outer_node_shared_input": [b * n, 144, 16],
            "history_lengths": [144, 72, 36, 18],
            "normalized_shapes": [
                [b * n, 144, 16],
                [b * n, 72, 16],
                [b * n, 36, 16],
                [b * n, 18, 16],
            ],
            "channel_independent_embedding_shapes": [
                [b * n * 16, 144, 16],
                [b * n * 16, 72, 16],
                [b * n * 16, 36, 16],
                [b * n * 16, 18, 16],
            ],
            "decomposition": "moving_avg_25_at_all_four_scales",
            "season_mixing": "bottom_up_144_to_72_to_36_to_18",
            "trend_mixing": "top_down_18_to_36_to_72_to_144",
            "predictor_outputs": [[b * n, 10, 16]] * 4,
            "fusion": "stack_four_predictor_outputs_then_sum",
            "denormalized_output": [b * n, 10, 16],
            "silent_crop": False,
            "disabled_scales": [],
        }
        return output


class TSMixerAdapter(_E2CHistoryOnlyAdapter):
    model_id = "tsmixer"

    def normalize_output(self, raw_output, batch, **kwargs: Any) -> BenchmarkOutput:
        output = super().normalize_output(raw_output, batch, **kwargs)
        output.aux["mixer_trace"] = {
            "layers": 2,
            "temporal_mlp": "Linear(144,32)->ReLU->Linear(32,144)->Dropout",
            "channel_mlp": "Linear(16,32)->ReLU->Linear(32,16)->Dropout",
            "residual_connections_per_layer": 2,
            "projection": "Linear(144,10)",
            "upstream_normalization": "none",
        }
        return output


class FreTSAdapter(_E2CHistoryOnlyAdapter):
    model_id = "frets"

    def normalize_output(self, raw_output, batch, **kwargs: Any) -> BenchmarkOutput:
        output = super().normalize_output(raw_output, batch, **kwargs)
        output.aux["frequency_trace"] = {
            "token_embedding": [1, 128],
            "channel_fft_dimension": "16_historical_features",
            "temporal_fft_dimension": "144_history_steps",
            "channel_frequency_branch": True,
            "temporal_frequency_branch": True,
            "complex_dtype": "autocast_native",
            "upstream_forward_dtype": "trainer_autocast_native",
            "trainer_amp_enabled": True,
            "sparsity_threshold": 0.01,
            "fc_input_width": 144 * 128,
        }
        return output
