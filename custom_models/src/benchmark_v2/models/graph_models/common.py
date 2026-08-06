from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ...graph import GraphBundle, GraphProtocolError


MODEL_SUPPORTS = {
    "gcn": ("A_gcn",),
    "stgcn": ("L_tilde",),
    "dcrnn": ("P_forward", "P_reverse"),
}


def graph_identity_dict(
    bundle: GraphBundle, support_names: Iterable[str]
) -> dict[str, Any]:
    names = tuple(str(name) for name in support_names)
    unknown = [name for name in names if name not in bundle.matrices]
    if unknown:
        raise GraphProtocolError(f"Unknown frozen graph supports: {unknown}")
    return {
        "graph_id": bundle.spec.graph_id,
        "node_count": bundle.spec.node_count,
        "ordered_node_ids": list(bundle.ordered_node_ids),
        "selected_k": bundle.spec.selected_k,
        "graph_support_names": list(names),
        "graph_support_shapes": [list(bundle.matrices[name].shape) for name in names],
        "graph_runtime_dtype": "torch.float32",
        "graph_frozen_dtype": "numpy.float64",
        "graph_support_persistent": False,
    }


def runtime_support(bundle: GraphBundle, name: str):
    import torch

    if name not in bundle.matrices:
        raise GraphProtocolError(f"Frozen GraphBundle has no support {name}.")
    return torch.from_numpy(
        np.array(getattr(bundle, name), dtype=np.float32, copy=True)
    )


def validate_model_graph_identity(
    model: Any,
    bundle: GraphBundle,
    support_names: Iterable[str],
    *,
    context: str,
) -> None:
    expected = graph_identity_dict(bundle, support_names)
    actual = getattr(model, "graph_identity", None)
    if not isinstance(actual, dict):
        raise GraphProtocolError(f"{context} model has no graph identity.")
    mismatches = {
        key: (actual.get(key), expected[key])
        for key in expected
        if actual.get(key) != expected[key]
    }
    if mismatches:
        raise GraphProtocolError(f"{context} graph identity mismatch: {mismatches}")
