from __future__ import annotations

from typing import Any

from ..contracts import BenchmarkAdapter, BenchmarkBatch, BenchmarkOutput
from ..errors import ContractError


class NodeSharedAdapter(BenchmarkAdapter):
    """Shared parameters per node: (B,T,N,C) -> (B*N,T,C) -> (B,N,H)."""

    def build_model(self, config: dict[str, Any], protocol: Any):
        factory = config.get("factory")
        if factory is None:
            raise ContractError("NodeSharedAdapter has no model factory in E0-B.")
        return factory(config)

    def prepare_model_inputs(self, batch: BenchmarkBatch, **kwargs: Any):
        x = batch.x
        b, t, n, c = (int(v) for v in x.shape)
        return x.permute(0, 2, 1, 3).contiguous().reshape(b * n, t, c)

    def forward_model(self, model, model_inputs: Any, **kwargs: Any):
        return model(model_inputs)

    def normalize_output(self, raw_output: Any, batch: BenchmarkBatch, **kwargs: Any) -> BenchmarkOutput:
        if isinstance(raw_output, dict):
            if "prediction" in raw_output:
                raw_output = raw_output["prediction"]
            elif "pred" in raw_output:
                raw_output = raw_output["pred"]
            else:
                raise ContractError("Model output dict must explicitly provide prediction/pred.")
        raw_shape = tuple(int(v) for v in raw_output.shape)
        b, _, n, _ = (int(v) for v in batch.x.shape)
        if raw_output.ndim == 3:
            if raw_shape[-1] != 1:
                raise ContractError("(B*N,H,C) output is not accepted without explicit target_output_policy; no silent variable selection.")
            raw_output = raw_output[..., 0]
        if raw_output.ndim != 2 or raw_output.shape[0] != b * n:
            raise ContractError(f"Node-shared model must return (B*N,H) or (B*N,H,1), got {raw_shape}; B={b}, N={n}")
        h = int(raw_output.shape[1])
        prediction = raw_output.reshape(b, n, h)
        return BenchmarkOutput(
            prediction=prediction,
            raw_output_shape=raw_shape,
            semantic_trace={
                "adapter_kind": "node_shared",
                "reshape": "(B,T,N,C)->(B,N,T,C)->(B*N,T,C)->(B,N,H)",
                "target_column": "Patv_raw",
                "output_semantics": "future Patv_raw power only",
                "node_count_source": "original_batch_shape",
            },
        )

