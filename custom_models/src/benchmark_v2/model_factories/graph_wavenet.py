from __future__ import annotations

from typing import Any, Mapping


def create_model(*, config: Mapping[str, Any], protocol: Mapping[str, Any], graph_bundle: Any, **_: Any) -> Any:
    from ..models.graph_models.graph_wavenet import create_model as _create_model

    return _create_model(config=config, graph_bundle=graph_bundle)


def resolve_config(protocol: Mapping[str, Any], *, run_mode: str = "formal", ma_window: int | None = None) -> dict[str, Any]:
    from ..configs.graph_wavenet import resolve_graph_wavenet_config

    return resolve_graph_wavenet_config(protocol, run_mode=run_mode)


def create_adapter(*, model_id: str, graph_bundle: Any, **_: Any) -> Any:
    from ..adapters.e3_c import NativeAdaptiveNodeAdapter

    return NativeAdaptiveNodeAdapter(model_id, graph_bundle)


def load_graph_bundle() -> Any:
    from ..graph import load_graph_bundle as _load_graph_bundle

    return _load_graph_bundle()


def apply_graph_identity(config: dict[str, Any], graph_bundle: Any, model_id: str) -> None:
    from ..models.graph_models.adaptive_common import e3_c_graph_identity

    config.update(e3_c_graph_identity(graph_bundle, model_id))
