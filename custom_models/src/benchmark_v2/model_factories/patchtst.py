from __future__ import annotations

from typing import Any, Mapping


def create_model(*, config: Mapping[str, Any], protocol: Mapping[str, Any], **_: Any) -> Any:
    from ..models.patchtst import create_model as _create_model

    return _create_model(config=config, protocol=protocol)


def resolve_config(protocol: Mapping[str, Any], *, run_mode: str = "formal", ma_window: int | None = None) -> dict[str, Any]:
    from ..configs.patchtst import resolve_patchtst_config

    return resolve_patchtst_config(protocol, run_mode=run_mode)


def create_adapter(*, protocol: Mapping[str, Any], **_: Any) -> Any:
    from ..adapters.e2_a import PatchTSTAdapter

    return PatchTSTAdapter(protocol)
