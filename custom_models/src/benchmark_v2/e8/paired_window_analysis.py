from __future__ import annotations

from typing import Any

import numpy as np


def align_window_identity(left_ids, right_ids, left_values, right_values, left_mask=None, right_mask=None):
    left_ids, right_ids = list(left_ids), list(right_ids)
    if len(left_ids) != len(set(left_ids)) or len(right_ids) != len(set(right_ids)):
        raise ValueError("Window IDs must be unique.")
    if left_ids != right_ids:
        raise ValueError("WINDOW_IDENTITY_MISMATCH")
    left_values, right_values = np.asarray(left_values), np.asarray(right_values)
    if left_values.shape != right_values.shape or left_values.shape[0] != len(left_ids):
        raise ValueError("Prediction arrays do not align with window IDs.")
    if left_mask is not None or right_mask is not None:
        if left_mask is None or right_mask is None or not np.array_equal(left_mask, right_mask):
            raise ValueError("MASK_IDENTITY_MISMATCH")
    return left_values, right_values, None if left_mask is None else np.asarray(left_mask)


def _ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return None
    rx, ry = _ranks(x[valid]), _ranks(y[valid])
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def quantile_groups(signal, improvement, quantiles: int = 4) -> list[dict[str, Any]]:
    signal, improvement = np.asarray(signal, dtype=np.float64), np.asarray(improvement, dtype=np.float64)
    if signal.shape != improvement.shape:
        raise ValueError("Signal and improvement shapes must match.")
    boundaries = np.quantile(signal, np.linspace(0, 1, quantiles + 1))
    rows = []
    for index in range(quantiles):
        selected = (signal >= boundaries[index]) & (signal <= boundaries[index + 1] if index == quantiles - 1 else signal < boundaries[index + 1])
        values = improvement[selected]
        rows.append({"quantile": index + 1, "lower": float(boundaries[index]), "upper": float(boundaries[index + 1]),
                     "count": int(selected.sum()), "mean": float(values.mean()) if len(values) else None,
                     "median": float(np.median(values)) if len(values) else None})
    return rows


def paired_signal_analysis(signal, improvement) -> dict[str, Any]:
    return {"spearman": spearman(signal, improvement), "quantile_groups": quantile_groups(signal, improvement)}
