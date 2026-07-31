from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..data.signatures import feature_order_hash
from ..errors import ContractError
from ..protocol.hashing import stable_hash
from .node_shared import NodeSharedAdapter


@dataclass(frozen=True)
class BaselineScalerContext:
    power_feature_index: int
    input_power_mean: float
    input_power_std: float
    target_mean: float
    target_std: float
    input_scaler_hash: str
    target_scaler_hash: str
    feature_order_hash: str

    @classmethod
    def from_scalers(
        cls, protocol: Mapping[str, Any], input_scaler: Any, target_scaler: Any
    ) -> "BaselineScalerContext":
        features = list(protocol["ordered_input_features"])
        power_name = str(protocol["input_power_column"])
        if len(features) != 16 or int(protocol["feature_count"]) != 16:
            raise ContractError("Baseline scaler context requires exactly 16 features.")
        if features.count(power_name) != 1:
            raise ContractError(
                f"{power_name} must occur exactly once in ordered_input_features."
            )
        calculated_hash = feature_order_hash(features)
        if calculated_hash != protocol["feature_order_hash"]:
            raise ContractError(
                "ordered_input_features does not match the frozen feature_order_hash."
            )
        input_mean = np.asarray(input_scaler.mean, dtype=np.float64).reshape(-1)
        input_std = np.asarray(input_scaler.std, dtype=np.float64).reshape(-1)
        if input_mean.size != 16 or input_std.size != 16:
            raise ContractError("Input scaler must expose 16 feature means/stds.")
        target_mean = float(np.asarray(target_scaler.mean, dtype=np.float64).reshape(-1)[0])
        target_std = float(np.asarray(target_scaler.std, dtype=np.float64).reshape(-1)[0])
        index = features.index(power_name)
        values = [input_mean[index], input_std[index], target_mean, target_std]
        if not np.isfinite(values).all() or input_std[index] <= 0 or target_std <= 0:
            raise ContractError("Baseline scaler means/stds must be finite with positive std.")
        input_meta = input_scaler.to_dict()
        target_meta = target_scaler.to_dict()
        return cls(
            power_feature_index=index,
            input_power_mean=float(input_mean[index]),
            input_power_std=float(input_std[index]),
            target_mean=target_mean,
            target_std=target_std,
            input_scaler_hash=stable_hash(input_meta),
            target_scaler_hash=stable_hash(target_meta),
            feature_order_hash=calculated_hash,
        )

    def input_to_physical(self, values: Any) -> Any:
        return values * self.input_power_std + self.input_power_mean

    def physical_to_target(self, values: Any) -> Any:
        return (values - self.target_mean) / self.target_std

    def inverse_target(self, values: Any) -> Any:
        return values * self.target_std + self.target_mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "power_feature_index": self.power_feature_index,
            "input_power_mean": self.input_power_mean,
            "input_power_std": self.input_power_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "input_scaler_hash": self.input_scaler_hash,
            "target_scaler_hash": self.target_scaler_hash,
            "feature_order_hash": self.feature_order_hash,
            "conversion": "input normalized -> physical kW -> target normalized",
        }


class StatisticalBaselineAdapter(NodeSharedAdapter):
    """Node-shared statistical adapter with explicit input/target scaler conversion."""

    def __init__(self, scaler_context: BaselineScalerContext):
        self.scaler_context = scaler_context

    def prepare_model_inputs(self, batch, **kwargs: Any):
        node_history = super().prepare_model_inputs(batch, **kwargs)
        normalized_power = node_history[:, :, self.scaler_context.power_feature_index]
        return self.scaler_context.input_to_physical(normalized_power)

    def normalize_output(self, raw_output: Any, batch, **kwargs: Any):
        target_space = self.scaler_context.physical_to_target(raw_output)
        output = super().normalize_output(target_space, batch, **kwargs)
        output.semantic_trace.update(
            {
                "input_power_column": "Patv_clean_for_input",
                "input_power_space": "feature_scaler_normalized",
                "baseline_compute_space": "physical_kW",
                "output_space": "normalized_Patv_raw_target_space",
                "cross_node_interaction": False,
                "scaler_conversion": "input scaler inverse -> statistic -> target scaler transform",
            }
        )
        return output
