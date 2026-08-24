from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def validate_matrix(matrix: np.ndarray, *, node_count: int = 134) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (node_count, node_count):
        raise ValueError(f"Expected {(node_count, node_count)}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Graph matrix contains non-finite values.")
    return value


def row_top_k(matrix: np.ndarray, k: int, *, exclude_self: bool = True) -> np.ndarray:
    value = validate_matrix(matrix)
    if not 1 <= int(k) < value.shape[0]:
        raise ValueError("k must be between 1 and N-1.")
    result = np.zeros_like(value)
    for row in range(value.shape[0]):
        candidates = [(float(value[row, col]), col) for col in range(value.shape[1]) if not (exclude_self and row == col)]
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for weight, col in candidates:
            if weight == 0.0:
                continue
            result[row, col] = weight
            if np.count_nonzero(result[row]) == int(k):
                break
    return result


def edge_mask(matrix: np.ndarray, *, exclude_self: bool = False) -> np.ndarray:
    value = validate_matrix(matrix)
    mask = value != 0
    if exclude_self:
        np.fill_diagonal(mask, False)
    return mask


def matrix_statistics(matrix: np.ndarray, *, model_id: str, matrix_name: str) -> dict[str, Any]:
    value = validate_matrix(matrix)
    mask = edge_mask(value)
    nonself = edge_mask(value, exclude_self=True)
    weights = value[mask]
    reciprocal = nonself & nonself.T
    symmetric_equal = nonself & np.isclose(value, value.T, rtol=1e-10, atol=1e-12)
    out_degree = nonself.sum(axis=1)
    in_degree = nonself.sum(axis=0)
    weighted_out = np.where(nonself, value, 0.0).sum(axis=1)
    weighted_in = np.where(nonself, value, 0.0).sum(axis=0)
    top_nodes = sorted(
        (
            {"TurbID": index + 1, "weighted_in_degree": float(weighted_in[index]), "weighted_out_degree": float(weighted_out[index])}
            for index in range(value.shape[0])
        ),
        key=lambda row: (-row["weighted_in_degree"], row["TurbID"]),
    )[:10]
    edge_rows = np.argwhere(nonself)
    top_edges = sorted(
        (
            {"source_TurbID": int(i + 1), "target_TurbID": int(j + 1), "weight": float(value[i, j])}
            for i, j in edge_rows
        ),
        key=lambda row: (-row["weight"], row["source_TurbID"], row["target_TurbID"]),
    )[:20]
    quantiles = np.quantile(weights, [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]).tolist() if weights.size else [None] * 7
    return {
        "model_id": model_id,
        "matrix_name": matrix_name,
        "node_count": int(value.shape[0]),
        "edge_count": int(mask.sum()),
        "nonself_edge_count": int(nonself.sum()),
        "density": float(mask.mean()),
        "self_loop_count": int(np.count_nonzero(np.diag(value))),
        "symmetry_ratio": float(symmetric_equal.sum() / nonself.sum()) if nonself.any() else None,
        "reciprocal_edge_ratio": float(reciprocal.sum() / nonself.sum()) if nonself.any() else None,
        "weight_quantiles": dict(zip(("q0", "q25", "q50", "q75", "q90", "q99", "q100"), quantiles)),
        "out_degree_min": int(out_degree.min()),
        "out_degree_mean": float(out_degree.mean()),
        "out_degree_max": int(out_degree.max()),
        "in_degree_min": int(in_degree.min()),
        "in_degree_mean": float(in_degree.mean()),
        "in_degree_max": int(in_degree.max()),
        "top_influence_nodes": top_nodes,
        "top_weighted_edges": top_edges,
    }


def matched_edge_overlap(prior: np.ndarray, learned: np.ndarray, *, k: int = 4) -> dict[str, Any]:
    prior_top = edge_mask(row_top_k(prior, k), exclude_self=True)
    learned_top = edge_mask(row_top_k(learned, k), exclude_self=True)
    shared = prior_top & learned_top
    union = prior_top | learned_top
    prior_count = int(prior_top.sum())
    learned_count = int(learned_top.sum())
    shared_count = int(shared.sum())
    return {
        "matched_k": int(k),
        "prior_edges": prior_count,
        "learned_edges": learned_count,
        "shared": shared_count,
        "prior_only": int((prior_top & ~learned_top).sum()),
        "learned_only": int((learned_top & ~prior_top).sum()),
        "jaccard": float(shared_count / union.sum()) if union.any() else None,
        "precision": float(shared_count / learned_count) if learned_count else None,
        "recall": float(shared_count / prior_count) if prior_count else None,
        "row_wise_top_k_overlap": float(shared.sum(axis=1).mean() / k),
        "reciprocal_shared": int((shared & shared.T).sum()),
    }


def load_locations(path: str | Path, ordered_node_ids: list[int]) -> np.ndarray:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {int(row["TurbID"]): row for row in csv.DictReader(handle)}
    if set(rows) != set(ordered_node_ids):
        raise ValueError("Location TurbID set does not match node order.")
    return np.asarray([[float(rows[node]["x"]), float(rows[node]["y"])] for node in ordered_node_ids], dtype=np.float64)


def distance_weight_analysis(matrix: np.ndarray, coordinates: np.ndarray, *, model_id: str, matrix_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = validate_matrix(matrix)
    if coordinates.shape != (134, 2):
        raise ValueError("Coordinates must be (134,2).")
    distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    mask = edge_mask(value, exclude_self=True)
    i, j = np.where(mask)
    weights = value[i, j]
    scale = float(np.max(np.abs(weights))) if weights.size else 1.0
    normalized_matrix = value / scale if scale > 0 else value.copy()
    distances = distance[i, j]
    from scipy.stats import spearmanr

    correlation = spearmanr(distances, weights).statistic if len(weights) > 1 else None
    pair_distances = distance[np.triu_indices(134, 1)]
    boundaries = np.quantile(pair_distances, [0, 0.25, 0.5, 0.75, 1])
    bins = np.digitize(distance, boundaries[1:-1], right=True)
    bin_rows = []
    possible = ~np.eye(134, dtype=bool)
    for index in range(4):
        possible_bin = possible & (bins == index)
        edge_bin = mask & (bins == index)
        bin_weights = normalized_matrix[edge_bin]
        bin_rows.append(
            {
                "model_id": model_id,
                "matrix_name": matrix_name,
                "distance_bin": index + 1,
                "distance_min": float(boundaries[index]),
                "distance_max": float(boundaries[index + 1]),
                "possible_pairs": int(possible_bin.sum()),
                "edge_count": int(edge_bin.sum()),
                "edge_probability": float(edge_bin.sum() / possible_bin.sum()) if possible_bin.any() else None,
                "mean_normalized_weight": float(bin_weights.mean()) if bin_weights.size else None,
                "median_normalized_weight": float(np.median(bin_weights)) if bin_weights.size else None,
            }
        )
    return {
        "model_id": model_id,
        "matrix_name": matrix_name,
        "edge_count": int(len(weights)),
        "spearman_weight_vs_distance": None if correlation is None or not np.isfinite(correlation) else float(correlation),
        "comparison_policy": "within-model normalized weight/rank/top-k only",
    }, bin_rows
