from __future__ import annotations

from typing import Any

from ..contracts import BenchmarkAdapter, BenchmarkBatch, BenchmarkOutput, GraphContext
from ..errors import ContractError


class NativeSpatiotemporalAdapter(BenchmarkAdapter):
    def __init__(self, *, requires_graph: bool = False):
        self.requires_graph = bool(requires_graph)

    def build_model(self, config: dict[str, Any], protocol: Any):
        factory = config.get("factory")
        if factory is None:
            raise ContractError("NativeSpatiotemporalAdapter has no model factory in E0-B.")
        return factory(config)

    def prepare_model_inputs(self, batch: BenchmarkBatch, **kwargs: Any):
        if self.requires_graph:
            context = kwargs.get("graph_context") or batch.metadata.get("graph_context")
            if not isinstance(context, GraphContext):
                raise ContractError("GraphContext is required by native graph adapter.")
            context.validate(batch.node_count)
        return batch.x

    def forward_model(self, model, model_inputs: Any, **kwargs: Any):
        if not self.requires_graph:
            return model(model_inputs)
        try:
            return model(model_inputs, graph_context=kwargs.get("graph_context"))
        except TypeError as exc:
            if "graph_context" not in str(exc):
                raise
            return model(model_inputs)

    def normalize_output(self, raw_output: Any, batch: BenchmarkBatch, **kwargs: Any) -> BenchmarkOutput:
        if isinstance(raw_output, dict):
            raw_output = raw_output.get("prediction", raw_output.get("pred"))
        if raw_output is None:
            raise ContractError("Native model output must explicitly provide prediction.")
        raw_shape = tuple(int(v) for v in raw_output.shape)
        expected = (batch.batch_size, batch.node_count, int(batch.target.shape[-1]))
        if raw_shape != expected:
            raise ContractError(f"Native spatiotemporal output must be {expected}, got {raw_shape}")
        return BenchmarkOutput(
            prediction=raw_output,
            raw_output_shape=raw_shape,
            semantic_trace={
                "adapter_kind": "native_spatiotemporal",
                "reshape": "none; native (B,T,N,C)->(B,N,H)",
                "target_column": "Patv_raw",
                "output_semantics": "future Patv_raw power only",
            },
        )
