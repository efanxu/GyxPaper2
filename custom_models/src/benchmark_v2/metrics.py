from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def _arrays(pred: Any, target: Any, mask: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    to_np = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    p, y, m = to_np(pred).astype(np.float64), to_np(target).astype(np.float64), to_np(mask).astype(bool)
    if p.shape != y.shape or y.shape != m.shape or p.ndim != 3:
        raise ValueError(f"pred/target/mask must align as (B,N,H), got {p.shape}, {y.shape}, {m.shape}")
    return p, y, m & np.isfinite(p) & np.isfinite(y)


def official_score(pred: Any, target: Any, mask: Any, num_nodes: int) -> float:
    p, y, m = _arrays(pred, target, mask)
    # Invalid raw targets may be NaN. Mask before aggregation because
    # NumPy's NaN * False remains NaN and would poison an otherwise valid score.
    err_mw = np.where(m, (p - y) / 1000.0, 0.0)
    valid_h = m.sum(axis=2).astype(float)
    valid_node = valid_h > 0
    denom = np.maximum(valid_h, 1.0)
    mae = (np.abs(err_mw) * m).sum(axis=2) / denom
    rmse = np.sqrt((np.square(err_mw) * m).sum(axis=2) / denom)
    node_score = 0.5 * (mae + rmse)
    valid_count = valid_node.sum(axis=1).astype(float)
    valid_sample = valid_count > 0
    if not np.any(valid_sample):
        return float("nan")
    per_sample = float(num_nodes) * (node_score * valid_node).sum(axis=1) / np.maximum(valid_count, 1e-12)
    return float(per_sample[valid_sample].mean())


def regression_metrics(pred: Any, target: Any, mask: Any, *, num_nodes: int, physical_clip_applied: bool = False) -> dict[str, Any]:
    p, y, m = _arrays(pred, target, mask)
    pv, yv = p[m], y[m]
    total = int(m.size)
    base = {"valid_target_count": int(len(yv)), "total_target_count": total, "valid_target_ratio": float(len(yv) / total) if total else float("nan"), "physical_clip_applied": bool(physical_clip_applied), "score_lower_is_better": True}
    if len(yv) == 0:
        return {**base, "MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "Score": float("nan"), "score": float("nan"), "status": "NO_VALID_TARGET"}
    err = pv - yv
    sst = float(np.square(yv - yv.mean()).sum())
    score = official_score(p, y, m, num_nodes)
    return {**base, "MAE": float(np.abs(err).mean()), "RMSE": float(math.sqrt(np.square(err).mean())), "R2": float("nan") if sst <= 1e-6 else float(1 - np.square(err).sum() / sst), "Score": score, "score": score, "status": "OK", "score_compute_unit": "MW_after_kW_to_MW_conversion"}


def evaluate_horizons(pred: Any, target: Any, mask: Any, horizons: Iterable[int] = (3, 6, 10), *, num_nodes: int, physical_clip: tuple[float, float] | None = (0.0, 1500.0)) -> list[dict[str, Any]]:
    p, y, m = _arrays(pred, target, mask)
    if physical_clip is not None:
        p = np.clip(p, physical_clip[0], physical_clip[1])
    return [{"horizon": int(h), **regression_metrics(p[:, :, :int(h)], y[:, :, :int(h)], m[:, :, :int(h)], num_nodes=num_nodes, physical_clip_applied=physical_clip is not None)} for h in horizons]
