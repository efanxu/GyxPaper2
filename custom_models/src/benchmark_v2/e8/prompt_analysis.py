from __future__ import annotations

from typing import Any

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (a * b).sum(axis=-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), eps)


def macro_prompt_statistics(raw_prompt, fine, coarse, *, has_physical_trend_head: bool = False) -> dict[str, Any]:
    prompt = np.asarray(raw_prompt, dtype=np.float64)
    if prompt.ndim < 3:
        raise ValueError("Macro Prompt needs [...,prompt_len,hidden].")
    pooled = prompt.mean(axis=-2)
    fine_pooled = np.asarray(fine, dtype=np.float64).mean(axis=-2)
    coarse_pooled = np.asarray(coarse, dtype=np.float64).mean(axis=-2)
    result = {
        "raw_prompt_shape": list(prompt.shape), "pooled_prompt_shape": list(pooled.shape),
        "prompt_norm_mean": float(np.linalg.norm(prompt, axis=-1).mean()),
        "prompt_variance": float(prompt.var()),
        "prompt_fine_similarity_mean": float(_cosine(pooled, fine_pooled).mean()),
        "prompt_coarse_similarity_mean": float(_cosine(pooled, coarse_pooled).mean()),
        "physical_trend_semantics": "AVAILABLE" if has_physical_trend_head else "NOT_APPLICABLE",
    }
    if not has_physical_trend_head:
        result.update({"trend_sign_accuracy": "NOT_APPLICABLE", "trend_correlation": "NOT_APPLICABLE", "trend_confusion_matrix": "NOT_APPLICABLE"})
    return result


def st_prompt_horizon_statistics(st_prompt) -> dict[str, Any]:
    value = np.asarray(st_prompt, dtype=np.float64)
    if value.ndim != 4 or value.shape[0] != 1:
        raise ValueError("ST Prompt must be [1,H,N,D].")
    horizon = value[0].mean(axis=1)
    normed = horizon / np.maximum(np.linalg.norm(horizon, axis=-1, keepdims=True), 1e-12)
    cosine = normed @ normed.T
    adjacent = np.diag(cosine, 1)
    far_mask = np.abs(np.arange(len(horizon))[:, None] - np.arange(len(horizon))[None, :]) >= max(2, len(horizon) // 2)
    return {
        "pairwise_cosine_matrix": cosine.tolist(),
        "adjacent_horizon_similarity": float(adjacent.mean()) if len(adjacent) else None,
        "far_horizon_similarity": float(cosine[far_mask].mean()) if far_mask.any() else None,
        "inter_horizon_variance": float(horizon.var(axis=0).mean()),
        "prompt_norm_by_horizon": np.linalg.norm(horizon, axis=-1).tolist(),
        "representation_distance_by_horizon": np.linalg.norm(horizon[:, None] - horizon[None, :], axis=-1).tolist(),
    }
