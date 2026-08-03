from __future__ import annotations

from typing import Any, Mapping


def create_model(*, config: Mapping[str, Any], protocol: Mapping[str, Any], graph_bundle: Any, **_: Any) -> Any:
    from ..models.graph_models.dcrnn import create_model as _create_model

    return _create_model(config=config, protocol=protocol, graph_bundle=graph_bundle)


def resolve_config(protocol: Mapping[str, Any], *, run_mode: str = "formal", ma_window: int | None = None) -> dict[str, Any]:
    from ..configs.dcrnn import resolve_dcrnn_config

    return resolve_dcrnn_config(protocol, run_mode=run_mode)


def create_adapter(*, model_id: str, graph_bundle: Any, **_: Any) -> Any:
    from ..adapters.e3_b import NativeGraphAdapter

    return NativeGraphAdapter(model_id, graph_bundle)


def load_graph_bundle() -> Any:
    from ..graph import load_graph_bundle as _load_graph_bundle

    return _load_graph_bundle()


def apply_graph_identity(config: dict[str, Any], graph_bundle: Any) -> None:
    from ..models.graph_models.common import graph_identity_dict

    config.update(graph_identity_dict(graph_bundle, tuple(config["graph_support_names"])))
