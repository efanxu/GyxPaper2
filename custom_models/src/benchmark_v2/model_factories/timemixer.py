from __future__ import annotations

from typing import Any, Mapping


def create_model(*, config: Mapping[str, Any], protocol: Mapping[str, Any], **_: Any) -> Any:
    from ..models.timemixer import create_model as _create_model

    return _create_model(config=config, protocol=protocol)


def resolve_config(protocol: Mapping[str, Any], *, run_mode: str = "formal", ma_window: int | None = None) -> dict[str, Any]:
    from ..configs.timemixer import resolve_timemixer_config

    return resolve_timemixer_config(protocol, run_mode=run_mode)


def create_adapter(*, protocol: Mapping[str, Any], **_: Any) -> Any:
    from ..adapters.e2_c import TimeMixerAdapter

    return TimeMixerAdapter(protocol)
