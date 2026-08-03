from __future__ import annotations

from typing import Any, Mapping


def create_model(*, config: Mapping[str, Any], protocol: Mapping[str, Any], **_: Any) -> Any:
    from ..models.persistence import create_model as _create_model

    return _create_model(config=config, protocol=protocol)


def resolve_config(protocol: Mapping[str, Any], *, run_mode: str = "formal", ma_window: int | None = None) -> dict[str, Any]:
    from ..configs.persistence import resolve_persistence_config

    return resolve_persistence_config(protocol, run_mode=run_mode)


def create_adapter(*, input_scaler: Any, target_scaler: Any, protocol: Mapping[str, Any], **_: Any) -> Any:
    from ..adapters.statistical_baselines import BaselineScalerContext, StatisticalBaselineAdapter

    context = BaselineScalerContext.from_scalers(protocol, input_scaler, target_scaler)
    return StatisticalBaselineAdapter(context)
