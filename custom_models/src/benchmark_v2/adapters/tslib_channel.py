from __future__ import annotations

from typing import Any, Mapping

from ..contracts import BenchmarkOutput
from ..errors import ContractError
from .node_shared import NodeSharedAdapter


class TSLibPowerChannelAdapter(NodeSharedAdapter):
    """Shared M-to-S normalization for audited TSLib models that emit enc_in."""

    model_id = "tslib"

    def __init__(self, protocol: Mapping[str, Any]):
        self.protocol = protocol
        features = tuple(protocol["ordered_input_features"])
        if len(features) != int(protocol["feature_count"]):
            raise ContractError("Protocol feature count/order mismatch.")
        power_feature = str(protocol["input_power_column"])
        if features.count(power_feature) != 1:
            raise ContractError(
                "Patv_clean_for_input must occur exactly once in feature order."
            )
        self.ordered_features = features
        self.power_feature = power_feature
        self.power_feature_index = features.index(power_feature)

    def forward_model(self, model, model_inputs: Any, **kwargs: Any):
        return model(model_inputs, None, None, None)

    def normalize_output(
        self, raw_output: Any, batch, **kwargs: Any
    ) -> BenchmarkOutput:
        if not hasattr(raw_output, "shape"):
            raise ContractError(
                f"{self.model_id} must return a tensor, got {type(raw_output)}"
            )
        raw_shape = tuple(int(value) for value in raw_output.shape)
        b, _, n, c = (int(value) for value in batch.x.shape)
        expected_horizon = int(self.protocol["max_pred_len"])
        expected = (b * n, expected_horizon, c)
        if raw_shape != expected:
            raise ContractError(
                f"{self.model_id} raw output must be {expected}; got {raw_shape}. "
                "No truncation, padding, broadcast, or channel averaging is allowed."
            )
        selected = raw_output[..., self.power_feature_index]
        prediction = selected.reshape(b, n, expected_horizon)
        return BenchmarkOutput(
            prediction=prediction,
            raw_output_shape=raw_shape,
            semantic_trace={
                "adapter_kind": "node_shared",
                "model_kind": self.model_id,
                "reshape": (
                    "(B,T,N,C)->(B,N,T,C)->(B*N,T,C)"
                    "->(B*N,H,C)->select_power_channel->(B,N,H)"
                ),
                "input_mode": "multivariate",
                "output_mode": "single_target",
                "output_policy": "select_power_associated_channel",
                "selected_feature": self.power_feature,
                "selected_feature_index": self.power_feature_index,
                "target_column": "Patv_raw",
                "supervised_target": "Patv_raw",
                "output_semantics": "future Patv_raw power only",
                "output_space": "normalized_Patv_raw_target_space",
                "node_semantics": "node_shared",
                "parameter_sharing": "one upstream model shared by all nodes",
                "cross_node_interaction": False,
                "uses_future_target": False,
                "uses_future_observed_covariates": False,
                "uses_future_calendar_covariates": False,
            },
        )
