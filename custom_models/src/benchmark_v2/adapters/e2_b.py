from __future__ import annotations

from typing import Any, Mapping

from ..contracts import BenchmarkOutput
from ..errors import ContractError
from .node_shared import NodeSharedAdapter
from .tslib_channel import TSLibPowerChannelAdapter


class _HistoryOnlyMixin:
    model_id = "e2_b"

    def _reject_future_inputs(self, batch, kwargs: Mapping[str, Any]) -> None:
        import torch

        if batch.x_mark is not None or batch.y_mark is not None:
            raise ContractError(
                f"{self.model_id} disables supplied calendar/time marks."
            )
        for key in ("future_observed_covariates", "future_exogenous"):
            value = kwargs.get(key)
            if value is not None and bool(torch.any(torch.as_tensor(value) != 0)):
                raise ContractError(f"{self.model_id} rejects non-zero {key}.")


class _HistoryOnlyPowerChannelAdapter(
    _HistoryOnlyMixin, TSLibPowerChannelAdapter
):
    def prepare_model_inputs(self, batch, **kwargs: Any):
        self._reject_future_inputs(batch, kwargs)
        return super().prepare_model_inputs(batch, **kwargs)


class TimesNetAdapter(_HistoryOnlyPowerChannelAdapter):
    model_id = "timesnet"

    def __init__(self, protocol: Mapping[str, Any], *, top_k: int):
        super().__init__(protocol)
        self.top_k = int(top_k)
        self.last_period_trace: dict[str, Any] = {}

    def prepare_model_inputs(self, batch, **kwargs: Any):
        import torch

        flat = super().prepare_model_inputs(batch, **kwargs)
        frequency_bins = int(flat.shape[1]) // 2 + 1
        if not 0 < self.top_k <= frequency_bins - 1:
            raise ContractError(
                f"TimesNet top_k={self.top_k} invalid for {frequency_bins} bins."
            )
        spectrum = torch.fft.rfft(flat.detach().float(), dim=1).abs()
        amplitude = spectrum.mean(dim=(0, 2))
        amplitude[0] = float("-inf")
        indices = torch.topk(amplitude, self.top_k).indices
        periods = [int(flat.shape[1]) // int(index) for index in indices]
        if any(period <= 0 for period in periods):
            raise ContractError("TimesNet detected a non-positive history period.")
        self.last_period_trace = {
            "history_length": int(flat.shape[1]),
            "configured_top_k": self.top_k,
            "frequency_bins": frequency_bins,
            "example_history_frequency_indices": [
                int(value) for value in indices.cpu()
            ],
            "example_history_periods": periods,
            "period_source": "BenchmarkBatch.x history only",
        }
        b, _, n, _ = (int(value) for value in batch.x.shape)
        grouped = flat.reshape(b, n, flat.shape[1], flat.shape[2]).permute(
            1, 0, 2, 3
        ).contiguous()
        return grouped, b, n

    def forward_model(self, model, model_inputs: Any, **kwargs: Any):
        import torch

        grouped, b, n = model_inputs
        per_node = [
            model(grouped[index], None, None, None)
            for index in range(n)
        ]
        stacked = torch.stack(per_node, dim=0)
        return stacked.permute(1, 0, 2, 3).contiguous().reshape(
            b * n, stacked.shape[2], stacked.shape[3]
        )

    def normalize_output(self, raw_output, batch, **kwargs: Any) -> BenchmarkOutput:
        output = super().normalize_output(raw_output, batch, **kwargs)
        output.aux["period_trace"] = dict(self.last_period_trace)
        output.aux["multiscale_geometry"] = {
            "mechanism": "dynamic_fft_periods_plus_2d_inception",
            "top_k": self.top_k,
            "padding": "per_period_to_multiple_then_exact_crop_to_154",
            "fft_batch_scope": (
                "same_turbine_across_B_samples; nodes are separate calls "
                "through one shared parameter set"
            ),
        }
        return output


class MICNAdapter(_HistoryOnlyPowerChannelAdapter):
    model_id = "micn"

    def __init__(self, protocol: Mapping[str, Any]):
        super().__init__(protocol)
        self.last_decoder_trace: dict[str, Any] = {}

    def prepare_model_inputs(self, batch, **kwargs: Any):
        import torch

        self._reject_future_inputs(batch, kwargs)
        flat = NodeSharedAdapter.prepare_model_inputs(self, batch, **kwargs)
        b_flat, history_length, features = flat.shape
        horizon = int(self.protocol["max_pred_len"])
        future = torch.zeros(
            b_flat, horizon, features, dtype=flat.dtype, device=flat.device
        )
        decoder = torch.cat([flat, future], dim=1)
        decoder_mark = torch.zeros(
            b_flat,
            history_length + horizon,
            4,
            dtype=flat.dtype,
            device=flat.device,
        )
        self.last_decoder_trace = {
            "x_enc_shape": tuple(int(v) for v in flat.shape),
            "x_dec_shape": tuple(int(v) for v in decoder.shape),
            "x_mark_dec_shape": tuple(int(v) for v in decoder_mark.shape),
            "seasonal_init_dec_length": history_length + horizon,
            "history_matches_observed_x": bool(
                torch.equal(decoder[:, :history_length], flat)
            ),
            "future_decoder_all_zero": bool(
                torch.count_nonzero(decoder[:, history_length:]) == 0
            ),
            "decoder_marks_all_zero": bool(
                torch.count_nonzero(decoder_mark) == 0
            ),
        }
        return flat, None, decoder, decoder_mark

    def forward_model(self, model, model_inputs: Any, **kwargs: Any):
        return model(*model_inputs)

    def normalize_output(self, raw_output, batch, **kwargs: Any) -> BenchmarkOutput:
        output = TSLibPowerChannelAdapter.normalize_output(
            self, raw_output, batch, **kwargs
        )
        output.aux["decoder_trace"] = dict(self.last_decoder_trace)
        output.aux["multiscale_geometry"] = {
            "conv_kernel": [12, 16],
            "decomp_kernel": [13, 17],
            "isometric_kernel": [13, 10],
            "branch_count": 2,
            "fusion": "Conv2d across both scale branches",
        }
        return output


class WPMixerAdapter(_HistoryOnlyPowerChannelAdapter):
    model_id = "wpmixer"

    def normalize_output(self, raw_output, batch, **kwargs: Any) -> BenchmarkOutput:
        output = super().normalize_output(raw_output, batch, **kwargs)
        core = kwargs.get("_unused_core")
        del core
        output.aux["multiscale_geometry"] = {
            "mechanism": "db2_level1_wavelet_approximation_and_detail",
            "branch_count": 2,
            "patch_len": 16,
            "stride": 8,
            "fusion": "inverse_discrete_wavelet_transform",
        }
        return output


class MultiPatchFormerAdapter(_HistoryOnlyPowerChannelAdapter):
    model_id = "multipatchformer"

    def normalize_output(self, raw_output, batch, **kwargs: Any) -> BenchmarkOutput:
        output = super().normalize_output(raw_output, batch, **kwargs)
        output.aux["multiscale_geometry"] = {
            "patch_lengths": [8, 16, 24, 32],
            "strides": [8, 8, 7, 7],
            "paddings": [0, 8, 0, 8],
            "patch_counts": [18, 18, 18, 18],
            "branch_count": 4,
            "silent_crop": False,
            "disabled_branches": [],
            "semi_autoregressive_head": (
                "exact cumulative chunk widths for pred_len=10"
            ),
            "fusion": "feature-axis concatenation before shared attention",
        }
        return output
