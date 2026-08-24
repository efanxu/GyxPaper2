from __future__ import annotations

from typing import Any

import numpy as np


def deterministic_align(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Align [samples,time,features] with one declared mean-bin rule."""
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 3 or right.ndim != 3 or left.shape[0] != right.shape[0] or left.shape[2] != right.shape[2]:
        raise ValueError("Representations must be [samples,time,features] with matching samples/features.")
    target = min(left.shape[1], right.shape[1])
    if left.shape[1] == right.shape[1]:
        return left, right, {"method": "identity", "target_length": target}

    def pool(array: np.ndarray) -> np.ndarray:
        edges = np.linspace(0, array.shape[1], target + 1, dtype=int)
        return np.stack([array[:, edges[i]:edges[i + 1]].mean(axis=1) for i in range(target)], axis=1)

    return pool(left), pool(right), {"method": "deterministic_equal_bins_mean", "target_length": target}


def pooled_cosine(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    left, right, _ = deterministic_align(left, right)
    a, b = left.mean(axis=1), right.mean(axis=1)
    return (a * b).sum(axis=-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), eps)


def linear_cka(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> float:
    left, right, _ = deterministic_align(left, right)
    x, y = left.reshape(-1, left.shape[-1]).astype(np.float64), right.reshape(-1, right.shape[-1]).astype(np.float64)
    x, y = x - x.mean(axis=0, keepdims=True), y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denom = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / max(denom, eps))


def representation_statistics(fine_pre, coarse_pre, fine_post, coarse_post) -> dict[str, Any]:
    arrays = [np.asarray(value, dtype=np.float64) for value in (fine_pre, coarse_pre, fine_post, coarse_post)]
    for value in arrays:
        if value.ndim != 3:
            raise ValueError("Representation statistics expect [samples,time,features].")
    fp, cp, fa, ca = arrays
    cosine_pre, cosine_post = pooled_cosine(fp, cp), pooled_cosine(fa, ca)
    fine_shift, coarse_shift = pooled_cosine(fp, fa), pooled_cosine(cp, ca)
    return {
        "fine_coarse_cosine_pre_mean": float(cosine_pre.mean()),
        "fine_coarse_cosine_post_mean": float(cosine_post.mean()),
        "fine_coarse_cosine_shift_mean": float((cosine_post - cosine_pre).mean()),
        "linear_cka_pre": linear_cka(fp, cp), "linear_cka_post": linear_cka(fa, ca),
        "fine_norm_mean": float(np.linalg.norm(fa, axis=-1).mean()),
        "coarse_norm_mean": float(np.linalg.norm(ca, axis=-1).mean()),
        "norm_ratio_fine_over_coarse": float(np.linalg.norm(fa, axis=-1).mean() / max(np.linalg.norm(ca, axis=-1).mean(), 1e-12)),
        "fine_relative_l2_shift": float(np.linalg.norm(fa - fp) / max(np.linalg.norm(fp), 1e-12)),
        "coarse_relative_l2_shift": float(np.linalg.norm(ca - cp) / max(np.linalg.norm(cp), 1e-12)),
        "fine_pre_post_cosine_mean": float(fine_shift.mean()),
        "coarse_pre_post_cosine_mean": float(coarse_shift.mean()),
        "sample_cosine_pre": cosine_pre.tolist(), "sample_cosine_post": cosine_post.tolist(),
    }
