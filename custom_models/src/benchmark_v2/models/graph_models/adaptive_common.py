from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ...graph import GraphBundle, GraphProtocolError
from .common import graph_identity_dict


E3_C_POLICIES: dict[str, dict[str, Any]] = {
    "graph_wavenet": {
        "uses_physical_support": True,
        "physical_support_names": ["P_forward", "P_reverse"],
        "adaptive_graph_policy": "learned_softmax_relu_low_rank",
        "node_identity_policy": "adaptive_low_rank_node_parameters",
        "temporal_identity_policy": "none",
        "adaptive_initialization_policy": "random_seeded",
    },
    "mtgnn": {
        "uses_physical_support": False,
        "physical_support_names": [],
        "adaptive_graph_policy": "directed_topk_graph_constructor",
        "node_identity_policy": "learned_directed_graph_node_embeddings",
        "temporal_identity_policy": "none",
        "adaptive_initialization_policy": "random_seeded",
    },
    "agcrn": {
        "uses_physical_support": False,
        "physical_support_names": [],
        "adaptive_graph_policy": "DAGG_from_node_embeddings",
        "node_identity_policy": "NAPL_shared_learned_node_embeddings",
        "temporal_identity_policy": "none",
        "adaptive_initialization_policy": "random_seeded",
    },
    "stid": {
        "uses_physical_support": False,
        "physical_support_names": [],
        "adaptive_graph_policy": "none",
        "node_identity_policy": "learned_node_embedding",
        "temporal_identity_policy": "historical_anchor_time_only",
        "adaptive_initialization_policy": "not_applicable",
    },
}


def e3_c_graph_identity(bundle: GraphBundle, model_id: str) -> dict[str, Any]:
    if model_id not in E3_C_POLICIES:
        raise GraphProtocolError(f"Unknown E3-C model policy: {model_id}")
    policy = E3_C_POLICIES[model_id]
    names = tuple(policy["physical_support_names"])
    identity = graph_identity_dict(bundle, names)
    identity.update(
        {
            "graph_context_id": bundle.spec.graph_id,
            "uses_physical_support": bool(policy["uses_physical_support"]),
            "physical_support_names": list(names),
            "physical_support_hashes": [
                bundle.matrix_hashes[name] for name in names
            ],
            "adaptive_graph_policy": policy["adaptive_graph_policy"],
            "node_identity_policy": policy["node_identity_policy"],
            "temporal_identity_policy": policy["temporal_identity_policy"],
            "adaptive_initialization_policy": policy[
                "adaptive_initialization_policy"
            ],
        }
    )
    return identity


def validate_e3_c_model_identity(
    model: Any, bundle: GraphBundle, model_id: str, *, context: str
) -> None:
    expected = e3_c_graph_identity(bundle, model_id)
    actual = getattr(model, "graph_identity", None)
    if not isinstance(actual, dict):
        raise GraphProtocolError(f"{context} model has no graph-context identity.")
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise GraphProtocolError(
            f"{context} graph-context/support policy mismatch: {mismatches}"
        )


def canonical_tensor_hash(value: Any) -> str:
    array = (
        value.detach().float().cpu().contiguous().numpy()
        if hasattr(value, "detach")
        else np.asarray(value, dtype=np.float32)
    )
    canonical = np.ascontiguousarray(array.astype("<f4", copy=False))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def learned_graph_summary(
    adjacency: Any,
    *,
    top_k: int | None = None,
    physical_supports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    array = adjacency.detach().float().cpu().numpy()
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"Adjacency must be square, got {array.shape}")
    finite = bool(np.isfinite(array).all())
    if not finite:
        raise ValueError("Learned adjacency contains NaN/Inf.")
    nonzero = np.abs(array) > 0.0
    rows = array.sum(axis=1)
    diagonal = np.diag(array)
    asymmetry = float(np.max(np.abs(array - array.T)))
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "canonical_content_hash": canonical_tensor_hash(array),
        "finite": finite,
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "density": float(nonzero.mean()),
        "edge_count": int(nonzero.sum()),
        "row_sum": {
            "min": float(rows.min()),
            "max": float(rows.max()),
            "mean": float(rows.mean()),
        },
        "self_loop": {
            "count": int((np.abs(diagonal) > 0.0).sum()),
            "mean": float(diagonal.mean()),
        },
        "asymmetry_max_abs": asymmetry,
        "symmetric": bool(asymmetry <= 1e-7),
        "out_degree": {
            "min": int(nonzero.sum(axis=1).min()),
            "max": int(nonzero.sum(axis=1).max()),
            "mean": float(nonzero.sum(axis=1).mean()),
        },
        "in_degree": {
            "min": int(nonzero.sum(axis=0).min()),
            "max": int(nonzero.sum(axis=0).max()),
            "mean": float(nonzero.sum(axis=0).mean()),
        },
    }
    if top_k is not None:
        order = np.argsort(-array, axis=1, kind="stable")[:, :top_k]
        summary["top_k"] = {
            "k": int(top_k),
            "indices": order.tolist(),
            "effective_nonzero_min": int(nonzero.sum(axis=1).min()),
            "effective_nonzero_max": int(nonzero.sum(axis=1).max()),
        }
    if physical_supports:
        learned_edges = nonzero.copy()
        if top_k is not None:
            learned_edges.fill(False)
            order = np.argsort(-array, axis=1, kind="stable")[:, :top_k]
            np.put_along_axis(learned_edges, order, True, axis=1)
        overlap: dict[str, Any] = {}
        for name, support in physical_supports.items():
            physical = (
                support.detach().float().cpu().numpy()
                if hasattr(support, "detach")
                else np.asarray(support)
            )
            physical_edges = np.abs(physical) > 0.0
            intersection = int(np.logical_and(learned_edges, physical_edges).sum())
            union = int(np.logical_or(learned_edges, physical_edges).sum())
            overlap[name] = {
                "intersection": intersection,
                "union": union,
                "jaccard": float(intersection / union) if union else 1.0,
            }
        summary["physical_graph_overlap_diagnostic_only"] = overlap
    return summary
