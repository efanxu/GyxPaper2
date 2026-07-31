from __future__ import annotations

from typing import Any, Mapping

from ..contracts import BenchmarkBatch
from ..errors import ContractError
from ..graph import GraphBundle, GraphProtocolError, validate_native_graph_input
from ..models.graph_models.common import (
    MODEL_SUPPORTS,
    graph_identity_dict,
    validate_model_graph_identity,
)
from .native_spatiotemporal import NativeSpatiotemporalAdapter


class NativeGraphAdapter(NativeSpatiotemporalAdapter):
    """Fail-closed native `(B,T,N,C)` graph-model boundary."""

    def __init__(self, model_id: str, bundle: GraphBundle):
        super().__init__(requires_graph=True)
        if model_id not in MODEL_SUPPORTS:
            raise ValueError(f"Unsupported E3-B graph model: {model_id}")
        self.model_id = model_id
        self.bundle = bundle
        self.support_names = MODEL_SUPPORTS[model_id]
        self.graph_identity = graph_identity_dict(bundle, self.support_names)

    @staticmethod
    def _nonzero(value: Any) -> bool:
        import torch

        return value is not None and bool(torch.any(torch.as_tensor(value) != 0))

    def _reject_future_inputs(
        self, batch: BenchmarkBatch, kwargs: Mapping[str, Any]
    ) -> None:
        if batch.x_mark is not None or batch.y_mark is not None:
            raise ContractError(
                f"{self.model_id} disables supplied calendar/time marks."
            )
        for key in (
            "future_observed_covariates",
            "future_exogenous",
            "future_weather",
            "future_calendar_marks",
            "decoder_target",
            "teacher_forcing_target",
        ):
            if self._nonzero(kwargs.get(key)):
                raise ContractError(f"{self.model_id} rejects non-zero {key}.")
            if self._nonzero(batch.metadata.get(key)):
                raise ContractError(
                    f"{self.model_id} batch rejects non-zero {key}."
                )

    def _validate_batch_graph_metadata(self, batch: BenchmarkBatch) -> None:
        for key in (
            "graph_id",
            "graph_protocol_hash",
            "node_order_hash",
            "graph_bundle_hash",
            "location_source_hash",
            "selected_k",
            "graph_support_names",
            "graph_support_hashes",
        ):
            if key in batch.metadata:
                expected = self.graph_identity[key]
                actual = batch.metadata[key]
                if actual != expected:
                    raise GraphProtocolError(
                        f"Batch {key} mismatch: {actual!r} != {expected!r}"
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
        self.bundle.validate_runtime_identity(
            graph_protocol_hash=self.graph_identity["graph_protocol_hash"],
            node_order_hash=self.graph_identity["node_order_hash"],
            graph_bundle_hash=self.graph_identity["graph_bundle_hash"],
        )
        self._validate_batch_graph_metadata(batch)
        return batch.x

    def forward_model(self, model, model_inputs: Any, **kwargs: Any) -> Any:
        validate_model_graph_identity(
            model,
            self.bundle,
            self.support_names,
            context=f"{self.model_id} forward",
        )
        return model(model_inputs)
