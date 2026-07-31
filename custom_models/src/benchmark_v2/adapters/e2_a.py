from __future__ import annotations

from typing import Any, Mapping

from ..contracts import BenchmarkOutput
from ..data.signatures import feature_order_hash
from ..errors import ContractError
from .node_shared import NodeSharedAdapter
from .tslib_channel import TSLibPowerChannelAdapter


class _HistoryOnlyMixin:
    def _reject_future_inputs(self, batch, kwargs: Mapping[str, Any]) -> None:
        import torch

        if batch.x_mark is not None or batch.y_mark is not None:
            raise ContractError(
                f"{self.model_id} formal protocol disables supplied calendar/time marks."
            )
        for key in ("future_observed_covariates", "future_exogenous"):
            value = kwargs.get(key)
            if value is not None and bool(torch.any(torch.as_tensor(value) != 0)):
                raise ContractError(f"{self.model_id} rejects non-zero {key}.")


def _single_target_trace(model_id: str) -> dict[str, Any]:
    return {
        "adapter_kind": "node_shared",
        "model_kind": model_id,
        "input_mode": "multivariate",
        "output_mode": "single_target",
        "output_policy": "native_c_out_1",
        "selected_feature": "Patv_clean_for_input",
        "target_column": "Patv_raw",
        "supervised_target": "Patv_raw",
        "output_semantics": "future Patv_raw power only",
        "output_space": "normalized_Patv_raw_target_space",
        "node_semantics": "node_shared",
        "parameter_sharing": "one upstream model shared by all nodes",
        "cross_node_interaction": False,
        "uses_node_embedding": False,
        "uses_node_specific_head": False,
        "uses_future_target": False,
        "uses_future_observed_covariates": False,
        "uses_future_calendar_covariates": False,
    }


class TransformerAdapter(_HistoryOnlyMixin, NodeSharedAdapter):
    model_id = "transformer"

    def __init__(self, protocol: Mapping[str, Any], *, label_len: int = 48):
        self.protocol = protocol
        self.label_len = int(label_len)
        self.last_decoder_trace: dict[str, Any] | None = None

    def prepare_model_inputs(self, batch, **kwargs: Any):
        import torch

        self._reject_future_inputs(batch, kwargs)
        flat = super().prepare_model_inputs(batch, **kwargs)
        b_flat, _, features = flat.shape
        horizon = int(self.protocol["max_pred_len"])
        history = flat[:, -self.label_len :, :]
        future = torch.zeros(
            b_flat, horizon, features, dtype=flat.dtype, device=flat.device
        )
        decoder = torch.cat([history, future], dim=1)
        enc_mark = torch.zeros(
            b_flat, int(self.protocol["lookback"]), 4,
            dtype=flat.dtype, device=flat.device
        )
        dec_mark = torch.zeros(
            b_flat, self.label_len + horizon, 4,
            dtype=flat.dtype, device=flat.device
        )
        self.last_decoder_trace = {
            "label_len": self.label_len,
            "decoder_shape": tuple(int(v) for v in decoder.shape),
            "history_matches_observed_x": bool(
                torch.equal(decoder[:, : self.label_len], history)
            ),
            "future_decoder_all_zero": bool(
                torch.count_nonzero(decoder[:, self.label_len :]) == 0
            ),
            "encoder_marks_all_zero": bool(torch.count_nonzero(enc_mark) == 0),
            "decoder_marks_all_zero": bool(torch.count_nonzero(dec_mark) == 0),
        }
        return flat, enc_mark, decoder, dec_mark

    def forward_model(self, model, model_inputs: Any, **kwargs: Any):
        return model(*model_inputs)

    def normalize_output(self, raw_output, batch, **kwargs: Any) -> BenchmarkOutput:
        b, _, n, _ = (int(value) for value in batch.x.shape)
        horizon = int(self.protocol["max_pred_len"])
        shape = tuple(int(value) for value in raw_output.shape)
        expected = (b * n, horizon, 1)
        if shape != expected:
            raise ContractError(f"transformer raw output must be {expected}; got {shape}.")
        return BenchmarkOutput(
            prediction=raw_output[..., 0].reshape(b, n, horizon),
            raw_output_shape=shape,
            aux={"decoder_trace": dict(self.last_decoder_trace or {})},
            semantic_trace=_single_target_trace(self.model_id),
        )


class PatchTSTAdapter(_HistoryOnlyMixin, TSLibPowerChannelAdapter):
    model_id = "patchtst"

    def __init__(self, protocol: Mapping[str, Any]):
        super().__init__(protocol)
        self.patch_trace = {
            "original_length": 144,
            "patch_len": 16,
            "stride": 8,
            "padding_patch": "end",
            "padding_length": 8,
            "padded_length": 152,
            "patch_num": 18,
            "final_patch_shape": "(B*N*16,18,512)",
        }

    def prepare_model_inputs(self, batch, **kwargs: Any):
        self._reject_future_inputs(batch, kwargs)
        return super().prepare_model_inputs(batch, **kwargs)

    def normalize_output(self, raw_output, batch, **kwargs: Any) -> BenchmarkOutput:
        output = super().normalize_output(raw_output, batch, **kwargs)
        output.aux["patch_trace"] = dict(self.patch_trace)
        return output


class ITransformerAdapter(_HistoryOnlyMixin, TSLibPowerChannelAdapter):
    model_id = "itransformer"

    def prepare_model_inputs(self, batch, **kwargs: Any):
        self._reject_future_inputs(batch, kwargs)
        flat = super().prepare_model_inputs(batch, **kwargs)
        self.variable_token_count = int(flat.shape[-1])
        if self.variable_token_count != 16:
            raise ContractError(
                f"iTransformer variable token count must be 16; got {self.variable_token_count}."
            )
        return flat

    def normalize_output(self, raw_output, batch, **kwargs: Any) -> BenchmarkOutput:
        output = super().normalize_output(raw_output, batch, **kwargs)
        output.aux["variable_token_count"] = self.variable_token_count
        output.aux["variable_token_semantics"] = "features_within_one_turbine"
        return output


class TimeXerAdapter(_HistoryOnlyMixin, NodeSharedAdapter):
    model_id = "timexer"

    def __init__(self, protocol: Mapping[str, Any]):
        self.protocol = protocol
        features = tuple(protocol["ordered_input_features"])
        if feature_order_hash(features) != protocol["feature_order_hash"]:
            raise ContractError("TimeXer feature order hash mismatch.")
        power = str(protocol["input_power_column"])
        if not features or features[-1] != power:
            raise ContractError(
                "TimeXer requires Patv_clean_for_input as the final endogenous channel."
            )
        self.features = features
        self.power_feature = power

    def prepare_model_inputs(self, batch, **kwargs: Any):
        self._reject_future_inputs(batch, kwargs)
        if kwargs.get("future_exogenous") is not None:
            raise ContractError("TimeXer has no future exogenous tensor.")
        return super().prepare_model_inputs(batch, **kwargs)

    def forward_model(self, model, model_inputs: Any, **kwargs: Any):
        return model(model_inputs, None, None, None)

    def normalize_output(self, raw_output, batch, **kwargs: Any) -> BenchmarkOutput:
        b, _, n, _ = (int(value) for value in batch.x.shape)
        horizon = int(self.protocol["max_pred_len"])
        shape = tuple(int(value) for value in raw_output.shape)
        expected = (b * n, horizon, 1)
        if shape != expected:
            raise ContractError(f"timexer raw output must be {expected}; got {shape}.")
        trace = _single_target_trace(self.model_id)
        trace.update(
            {
                "endogenous_feature": self.power_feature,
                "endogenous_feature_index": 15,
                "exogenous_features": list(self.features[:-1]),
                "future_exogenous_tensor": None,
            }
        )
        return BenchmarkOutput(
            prediction=raw_output[..., 0].reshape(b, n, horizon),
            raw_output_shape=shape,
            semantic_trace=trace,
        )
