from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .transforms import (
    gcn_support,
    random_walk,
    scaled_laplacian_fixed_two,
    symmetric_normalized_laplacian,
)
from .validation import (
    GraphProtocolError,
    canonical_matrix,
    canonical_node_ids,
    connected_components,
    validate_matrices,
)


GRAPH_ID = "sdwpf_physical_knn_v1"
SCHEMA_VERSION = "graph_protocol_v1"
NODE_COUNT = 134
MATRIX_FILENAMES = {
    "A_binary_directed": "adjacency_binary_directed_v1.npy",
    "A_binary_undirected": "adjacency_binary_undirected_v1.npy",
    "A_directed": "adjacency_directed_v1.npy",
    "A_undirected": "adjacency_undirected_v1.npy",
    "A_gcn": "adjacency_gcn_v1.npy",
    "P_forward": "random_walk_forward_v1.npy",
    "P_reverse": "random_walk_reverse_v1.npy",
    "L_sym": "laplacian_symmetric_v1.npy",
    "L_tilde": "laplacian_scaled_v1.npy",
}


@dataclass(frozen=True)
class GraphBuild:
    ordered_node_ids: tuple[Any, ...]
    node_metadata: dict[str, Any]
    matrices: dict[str, np.ndarray]
    diagnostics: dict[str, Any]


def _stable_neighbors(distances: np.ndarray, k: int) -> np.ndarray:
    indices = np.arange(distances.shape[0], dtype=np.int64)
    return np.lexsort((indices, distances))[:k]


def _candidate_diagnostics(
    distance: np.ndarray, *, maximum_k: int
) -> tuple[int, list[dict[str, Any]]]:
    node_count = distance.shape[0]
    records: list[dict[str, Any]] = []
    selected: int | None = None
    for k in range(1, maximum_k + 1):
        directed = np.zeros((node_count, node_count), dtype=np.float64)
        for node in range(node_count):
            directed[node, _stable_neighbors(distance[node], k)] = 1.0
        undirected = np.maximum(directed, directed.T)
        components = connected_components(undirected)
        degree = np.count_nonzero(undirected, axis=1)
        records.append(
            {
                "k": k,
                "component_count": len(components),
                "component_sizes": [len(values) for values in components],
                "degree_min": int(degree.min()),
                "degree_mean": float(degree.mean()),
                "degree_max": int(degree.max()),
            }
        )
        if selected is None and len(components) == 1:
            selected = k
    if selected is None:
        raise GraphProtocolError(
            "BLOCKED_GRAPH_CONNECTIVITY: k=1..20 never becomes connected."
        )
    return selected, records


def _node_metadata_payload(
    ordered_node_ids: tuple[Any, ...],
    frame: Any,
) -> dict[str, Any]:
    records = [
        {
            "node_id": node_id,
            "x": float(row.x),
            "y": float(row.y),
            "elevation": float(row.Ele),
        }
        for node_id, row in zip(ordered_node_ids, frame.itertuples(index=False))
    ]
    return {
        "schema_version": "node_metadata_v1",
        "node_id_type": "integer"
        if isinstance(ordered_node_ids[0], int)
        else "string",
        "coordinate_system": "cartesian_xy",
        "coordinate_columns": ["x", "y"],
        "coordinate_units": "SOURCE_UNIT_UNSPECIFIED",
        "elevation_column": "Ele",
        "elevation_units": "SOURCE_UNIT_UNSPECIFIED",
        "elevation_used_in_edge_distance": False,
        "records": records,
    }


def build_physical_graph(
    ordered_node_ids: list[Any] | tuple[Any, ...],
    location_frame: Any,
    *,
    maximum_k: int = 20,
) -> GraphBuild:
    import pandas as pd

    ordered = canonical_node_ids(ordered_node_ids)
    if len(ordered) != NODE_COUNT:
        raise GraphProtocolError(
            f"Expected exactly {NODE_COUNT} nodes, got {len(ordered)}."
        )
    if len(set(ordered)) != len(ordered):
        raise GraphProtocolError("Duplicate canonical node IDs are not allowed.")
    frame = location_frame.copy()
    required = {"TurbID", "x", "y", "Ele"}
    if not required.issubset(frame.columns):
        raise GraphProtocolError(
            f"Location columns missing: {sorted(required - set(frame.columns))}"
        )
    if frame["TurbID"].isna().any() or frame["TurbID"].duplicated().any():
        raise GraphProtocolError("Location node IDs must be non-null and unique.")
    expected_type = int if isinstance(ordered[0], int) else str
    if expected_type is int:
        frame["TurbID"] = pd.to_numeric(
            frame["TurbID"], errors="raise"
        ).astype(int)
    else:
        frame["TurbID"] = frame["TurbID"].astype(str)
    location_set = set(frame["TurbID"].tolist())
    if location_set != set(ordered):
        raise GraphProtocolError(
            "Input/target canonical node set and location node set differ."
        )
    frame = frame.set_index("TurbID").loc[list(ordered)].reset_index()
    coordinates = frame[["x", "y"]].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float64)
    elevation = pd.to_numeric(frame["Ele"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(coordinates).all() or not np.isfinite(elevation).all():
        raise GraphProtocolError("Location coordinates/elevation contain NaN or Inf.")
    if pd.DataFrame(coordinates).duplicated().any():
        raise GraphProtocolError("BLOCKED_DUPLICATE_COORDINATE")
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.sqrt(np.sum(difference * difference, axis=-1, dtype=np.float64))
    np.fill_diagonal(distance, np.inf)
    selected_k, candidates = _candidate_diagnostics(
        distance, maximum_k=maximum_k
    )
    binary_directed = np.zeros((NODE_COUNT, NODE_COUNT), dtype=np.float64)
    selected_distances: list[float] = []
    for node in range(NODE_COUNT):
        neighbors = _stable_neighbors(distance[node], selected_k)
        binary_directed[node, neighbors] = 1.0
        selected_distances.extend(float(distance[node, value]) for value in neighbors)
    selected_distance_array = np.asarray(selected_distances, dtype=np.float64)
    if np.any(selected_distance_array <= 0.0):
        raise GraphProtocolError("BLOCKED_DUPLICATE_COORDINATE")
    sigma = float(np.median(selected_distance_array))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise GraphProtocolError("Graph sigma must be positive and finite.")
    directed = np.zeros_like(binary_directed)
    support = binary_directed > 0
    directed[support] = np.exp(-np.square(distance[support] / sigma))
    undirected = np.maximum(directed, directed.T)
    binary_undirected = (undirected > 0).astype(np.float64)
    laplacian = symmetric_normalized_laplacian(undirected)
    matrices = {
        "A_binary_directed": binary_directed,
        "A_binary_undirected": binary_undirected,
        "A_directed": directed,
        "A_undirected": undirected,
        "A_gcn": gcn_support(undirected),
        "P_forward": random_walk(directed),
        "P_reverse": random_walk(directed.T),
        "L_sym": laplacian,
        "L_tilde": scaled_laplacian_fixed_two(laplacian),
    }
    matrices = {name: canonical_matrix(value) for name, value in matrices.items()}
    matrix_diagnostics = validate_matrices(matrices, node_count=NODE_COUNT)
    metadata = _node_metadata_payload(ordered, frame)
    undirected_degree = np.count_nonzero(binary_undirected, axis=1)
    weighted_degree = undirected.sum(axis=1)
    undirected_edge_distances = distance[
        np.triu(binary_undirected.astype(bool), k=1)
    ]
    nonzero_weights = directed[directed > 0]
    diagnostics = {
        "node_count": NODE_COUNT,
        "coordinate_ranges": {
            "x": [float(coordinates[:, 0].min()), float(coordinates[:, 0].max())],
            "y": [float(coordinates[:, 1].min()), float(coordinates[:, 1].max())],
            "elevation": [float(elevation.min()), float(elevation.max())],
        },
        "coordinate_units": "SOURCE_UNIT_UNSPECIFIED",
        "selected_k": selected_k,
        "candidate_k_component_counts": candidates,
        "directed_edge_count": int(np.count_nonzero(binary_directed)),
        "undirected_edge_count": int(
            np.count_nonzero(np.triu(binary_undirected, k=1))
        ),
        "density": float(
            np.count_nonzero(binary_undirected) / (NODE_COUNT * (NODE_COUNT - 1))
        ),
        "connected_component_count": len(connected_components(undirected)),
        "isolated_node_count": int(np.count_nonzero(undirected_degree == 0)),
        "degree": {
            "min": int(undirected_degree.min()),
            "mean": float(undirected_degree.mean()),
            "median": float(np.median(undirected_degree)),
            "max": int(undirected_degree.max()),
        },
        "weighted_degree": {
            "min": float(weighted_degree.min()),
            "mean": float(weighted_degree.mean()),
            "max": float(weighted_degree.max()),
        },
        "edge_distance": {
            "min": float(undirected_edge_distances.min()),
            "median": float(np.median(undirected_edge_distances)),
            "max": float(undirected_edge_distances.max()),
        },
        "sigma": sigma,
        "weight": {
            "min": float(nonzero_weights.min()),
            "median": float(np.median(nonzero_weights)),
            "max": float(nonzero_weights.max()),
        },
        "matrix_diagnostics": matrix_diagnostics,
    }
    return GraphBuild(
        ordered_node_ids=ordered,
        node_metadata=metadata,
        matrices=matrices,
        diagnostics=diagnostics,
    )


def make_protocol_payload(build: GraphBuild) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "graph_protocol_status": "FROZEN",
        "graph_id": GRAPH_ID,
        "dataset": "SDWPF",
        "node_count": NODE_COUNT,
        "ordered_node_ids": list(build.ordered_node_ids),
        "coordinate_system": "cartesian_xy",
        "coordinate_columns": ["x", "y"],
        "coordinate_units": "SOURCE_UNIT_UNSPECIFIED",
        "coordinate_semantics_evidence": [
            "dataset/clean_sdwpf_data.py: build_knn_neighbors uses Euclidean sqrt over x/y",
            "custom_models/src/st_mgprompt/graph_prior.py: build_distance_prior_graph uses squared Euclidean x/y",
        ],
        "distance_metric": "euclidean_2d",
        "elevation_policy": {
            "column": "Ele",
            "semantics": "elevation",
            "units": "SOURCE_UNIT_UNSPECIFIED",
            "stored_as_metadata": True,
            "used_in_edge_distance": False,
        },
        "k_selection_rule": (
            "minimum k in 1..20 whose maximum-symmetrized kNN graph has "
            "one connected component; exact distance ties break by canonical "
            "node-order index"
        ),
        "selected_k": build.diagnostics["selected_k"],
        "directedness": {
            "base": "directed kNN",
            "undirected": "elementwise maximum of A_directed and transpose",
        },
        "self_loop_policy": {
            "A_directed": False,
            "A_undirected": False,
            "A_gcn": "explicit A_undirected + I before normalization",
            "diffusion_supports": False,
        },
        "weight_formula": "sigma=median(selected positive directed distances); exp(-(distance/sigma)^2)",
        "sigma": build.diagnostics["sigma"],
        "normalization_formulas": {
            "A_gcn": "D^(-1/2) (A_undirected + I) D^(-1/2)",
            "P_forward": "D_out^(-1) A_directed",
            "P_reverse": "D_out_reverse^(-1) A_directed.T",
            "L_sym": "I - D^(-1/2) A_undirected D^(-1/2)",
            "L_tilde": "L_sym - I; lambda_max fixed to 2.0 theoretical bound",
        },
        "numeric_canonicalization": {
            "build_device": "cpu",
            "calculation_dtype": "float64",
            "storage_dtype": "little-endian float64",
            "c_contiguous": True,
            "negative_zero_normalized": True,
            "finite_required": True,
            "round_decimals": 12,
        },
        "matrix_files": {
            name: {
                "filename": filename,
                "shape": [NODE_COUNT, NODE_COUNT],
            }
            for name, filename in MATRIX_FILENAMES.items()
        },
        "runtime_policy": {
            "load_frozen_only": True,
            "automatic_rebuild": False,
            "automatic_overwrite": False,
            "node_order_mismatch": "FAIL_CLOSED",
        },
        "leakage_policy": {
            "edge_inputs": ["canonical node ID", "x", "y", "protocol constants"],
            "elevation_edge_input": False,
            "feature_values_used": False,
            "target_values_used": False,
            "mask_values_used": False,
            "split_used": False,
        },
    }
    return payload
