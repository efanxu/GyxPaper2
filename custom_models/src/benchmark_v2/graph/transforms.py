from __future__ import annotations

import numpy as np


def _degree_inverse_sqrt(adjacency: np.ndarray) -> np.ndarray:
    degree = np.asarray(adjacency, dtype=np.float64).sum(axis=1)
    if np.any(degree <= 0.0):
        raise ValueError("Normalized support cannot contain a zero-degree node.")
    return np.power(degree, -0.5)


def gcn_support(adjacency_undirected: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency_undirected, dtype=np.float64)
    adjacency_self = adjacency + np.eye(adjacency.shape[0], dtype=np.float64)
    inv_sqrt = _degree_inverse_sqrt(adjacency_self)
    return inv_sqrt[:, None] * adjacency_self * inv_sqrt[None, :]


def random_walk(adjacency_directed: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency_directed, dtype=np.float64)
    degree = adjacency.sum(axis=1)
    if np.any(degree <= 0.0):
        raise ValueError("Random-walk support cannot contain a zero-degree row.")
    return adjacency / degree[:, None]


def symmetric_normalized_laplacian(
    adjacency_undirected: np.ndarray,
) -> np.ndarray:
    adjacency = np.asarray(adjacency_undirected, dtype=np.float64)
    inv_sqrt = _degree_inverse_sqrt(adjacency)
    normalized = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    return np.eye(adjacency.shape[0], dtype=np.float64) - normalized


def scaled_laplacian_fixed_two(laplacian_symmetric: np.ndarray) -> np.ndarray:
    laplacian = np.asarray(laplacian_symmetric, dtype=np.float64)
    return laplacian - np.eye(laplacian.shape[0], dtype=np.float64)
