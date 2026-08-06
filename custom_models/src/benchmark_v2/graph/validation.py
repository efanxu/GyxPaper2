from __future__ import annotations

from typing import Any, Iterable

import numpy as np

class GraphProtocolError(ValueError):
    pass


ROUND_DECIMALS = 12


def canonical_node_ids(node_ids: Iterable[Any]) -> tuple[Any, ...]:
    values = tuple(node_ids)
    if not values:
        raise ValueError("Node order must not be empty.")
    if any(isinstance(value, bool) for value in values):
        raise TypeError("Boolean node IDs are not permitted.")
    kinds = {type(value) for value in values}
    if kinds in ({int}, {str}):
        return values
    raise TypeError("Node IDs must be uniformly integer or uniformly string.")


def canonical_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Graph matrix must be rank two, got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Graph matrix contains NaN or Inf.")
    matrix = np.round(matrix, decimals=ROUND_DECIMALS)
    matrix[matrix == 0.0] = 0.0
    return np.ascontiguousarray(matrix, dtype="<f8")


def connected_components(adjacency: np.ndarray) -> list[list[int]]:
    support = np.asarray(adjacency) > 0
    if support.ndim != 2 or support.shape[0] != support.shape[1]:
        raise GraphProtocolError("Connected-components input must be square.")
    support = np.logical_or(support, support.T)
    remaining = set(range(support.shape[0]))
    components: list[list[int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = [
                int(value)
                for value in np.flatnonzero(support[node])
                if int(value) in remaining
            ]
            for neighbor in neighbors:
                remaining.remove(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda values: (-len(values), values[0]))


def validate_node_order(
    runtime_node_ids: Iterable[Any],
    frozen_node_ids: Iterable[Any],
) -> None:
    runtime = canonical_node_ids(runtime_node_ids)
    frozen = canonical_node_ids(frozen_node_ids)
    if runtime != frozen:
        raise GraphProtocolError(
            "Runtime node order does not match frozen canonical node order."
        )


def validate_matrices(
    matrices: dict[str, np.ndarray], *, node_count: int
) -> dict[str, Any]:
    expected_names = {
        "A_binary_directed",
        "A_binary_undirected",
        "A_directed",
        "A_undirected",
        "A_gcn",
        "P_forward",
        "P_reverse",
        "L_sym",
        "L_tilde",
    }
    if set(matrices) != expected_names:
        raise GraphProtocolError(
            f"Graph matrix names mismatch: {sorted(set(matrices) ^ expected_names)}"
        )
    diagnostics: dict[str, Any] = {}
    for name, raw in matrices.items():
        matrix = np.asarray(raw, dtype=np.float64)
        if matrix.shape != (node_count, node_count):
            raise GraphProtocolError(
                f"{name} shape must be {(node_count, node_count)}, got {matrix.shape}."
            )
        if not np.isfinite(matrix).all():
            raise GraphProtocolError(f"{name} contains NaN or Inf.")
        if name not in {"L_sym", "L_tilde"} and np.any(matrix < -1e-12):
            raise GraphProtocolError(f"{name} contains negative values.")
        diagnostics[name] = {
            "shape": [node_count, node_count],
            "finite": True,
            "min": float(matrix.min()),
            "max": float(matrix.max()),
            "self_loop_count": int(np.count_nonzero(np.diag(matrix))),
            "symmetry_error": float(np.max(np.abs(matrix - matrix.T))),
            "row_sum_min": float(matrix.sum(axis=1).min()),
            "row_sum_max": float(matrix.sum(axis=1).max()),
        }
    for name in ("A_directed", "A_undirected"):
        if np.any(np.diag(matrices[name]) != 0):
            raise GraphProtocolError(f"{name} must not contain self-loops.")
    if not np.allclose(matrices["A_undirected"], matrices["A_undirected"].T):
        raise GraphProtocolError("A_undirected must be symmetric.")
    if len(connected_components(matrices["A_undirected"])) != 1:
        raise GraphProtocolError("A_undirected must be one connected component.")
    if not np.allclose(matrices["A_gcn"], matrices["A_gcn"].T, atol=1e-12):
        raise GraphProtocolError("A_gcn must be symmetric.")
    for name in ("P_forward", "P_reverse"):
        if not np.allclose(matrices[name].sum(axis=1), 1.0, atol=1e-12):
            raise GraphProtocolError(f"{name} rows must sum to one.")
    return diagnostics
