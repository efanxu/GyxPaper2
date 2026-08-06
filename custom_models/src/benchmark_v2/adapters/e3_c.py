from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from ..contracts import BenchmarkBatch
from ..errors import ContractError
from ..graph import GraphBundle, GraphProtocolError, validate_native_graph_input
from ..models.graph_models.adaptive_common import (
    E3_C_POLICIES,
    e3_c_graph_identity,
    validate_e3_c_model_identity,
)
from .native_spatiotemporal import NativeSpatiotemporalAdapter


class NativeAdaptiveNodeAdapter(NativeSpatiotemporalAdapter):
    """Fail-closed E3-C native node-axis and graph-context boundary."""

    def __init__(self, model_id: str, bundle: GraphBundle):
        super().__init__(requires_graph=True)
        if model_id not in E3_C_POLICIES:
            raise ValueError(f"Unsupported E3-C model: {model_id}")
        self.model_id = model_id
        self.bundle = bundle
        self.graph_identity = e3_c_graph_identity(bundle, model_id)

    @staticmethod
    def _provided(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (list, tuple, dict, str, bytes)):
            return bool(value)
        try:
            import torch

            return bool(torch.any(torch.as_tensor(value) != 0))
        except (TypeError, ValueError):
            return True

    def _reject_future_inputs(
        self, batch: BenchmarkBatch, kwargs: Mapping[str, Any]
    ) -> None:
        if batch.x_mark is not None or batch.y_mark is not None:
            raise ContractError(
                f"{self.model_id} rejects supplied calendar/time sequences."
            )
        for key in (
            "future_observed_covariates",
            "future_exogenous",
            "future_weather",
            "future_calendar_marks",
            "future_timestamps",
            "decoder_target",
            "teacher_forcing_target",
        ):
            if self._provided(kwargs.get(key)):
                raise ContractError(f"{self.model_id} rejects supplied {key}.")
            if self._provided(batch.metadata.get(key)):
                raise ContractError(
                    f"{self.model_id} batch rejects supplied {key}."
                )

    def _validate_batch_context(self, batch: BenchmarkBatch) -> None:
        keys = (
            "graph_id",
            "graph_context_id",
            "node_count",
            "ordered_node_ids",
            "selected_k",
            "graph_support_names",
            "graph_support_shapes",
            "uses_physical_support",
            "physical_support_names",
            "physical_support_shapes",
            "adaptive_graph_policy",
            "node_identity_policy",
            "temporal_identity_policy",
            "adaptive_initialization_policy",
        )
        for key in keys:
            if key in batch.metadata:
                actual = batch.metadata[key]
                expected = self.graph_identity[key]
                if actual != expected:
                    raise GraphProtocolError(
                        f"Batch {key} mismatch: {actual!r} != {expected!r}"
                    )

    @staticmethod
    def _timestamp_ids(value: Any) -> tuple[int, int]:
        import pandas as pd

        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            raise ContractError(
                "STID expects source-naive timestamps; timezone conversion is forbidden."
            )
        if (
            timestamp.second != 0
            or timestamp.microsecond != 0
            or timestamp.minute % 10 != 0
        ):
            raise ContractError(
                f"Historical anchor is not aligned to 10 minutes: {timestamp!r}"
            )
        return int(timestamp.hour * 6 + timestamp.minute // 10), int(
            timestamp.dayofweek
        )

    def _stid_ids(self, batch: BenchmarkBatch) -> tuple[Any, Any]:
        import torch

        timestamps = batch.metadata.get("history_end_timestamp")
        provided_tod = batch.metadata.get("history_end_time_of_day_id")
        provided_dow = batch.metadata.get("history_end_day_of_week_id")
        if timestamps is None:
            raise ContractError(
                "STID requires history_end_timestamp from the last observed row."
            )
        if not isinstance(timestamps, (list, tuple)):
            timestamps = [timestamps]
        if len(timestamps) != batch.batch_size:
            raise ContractError(
                "STID history_end_timestamp must contain one value per sample."
            )
        derived = [self._timestamp_ids(value) for value in timestamps]
        tod = [value[0] for value in derived]
        dow = [value[1] for value in derived]
        if provided_tod is not None and list(provided_tod) != tod:
            raise ContractError(
                "STID historical time-of-day ID disagrees with anchor timestamp."
            )
        if provided_dow is not None and list(provided_dow) != dow:
            raise ContractError(
                "STID historical day-of-week ID disagrees with anchor timestamp."
            )
        return (
            torch.as_tensor(tod, dtype=torch.long, device=batch.x.device),
            torch.as_tensor(dow, dtype=torch.long, device=batch.x.device),
        )

    def prepare_model_inputs(
        self, batch: BenchmarkBatch, **kwargs: Any
    ) -> Any:
        self._reject_future_inputs(batch, kwargs)
        validate_native_graph_input(
            batch.x,
            runtime_node_ids=batch.node_ids,
            bundle=self.bundle,
            expected_time=144,
            expected_features=16,
        )
        self._validate_batch_context(batch)
        if self.model_id == "stid":
            time_of_day_id, day_of_week_id = self._stid_ids(batch)
            return batch.x, time_of_day_id, day_of_week_id
        return batch.x

    def forward_model(self, model, model_inputs: Any, **kwargs: Any) -> Any:
        validate_e3_c_model_identity(
            model,
            self.bundle,
            self.model_id,
            context=f"{self.model_id} forward",
        )
        if self.model_id == "stid":
            x, time_of_day_id, day_of_week_id = model_inputs
            return model(x, time_of_day_id, day_of_week_id)
        return model(model_inputs)
