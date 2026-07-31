from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np


def bhn_to_bnh(values: np.ndarray) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError(f"Expected 3D array, got {values.shape}.")
    return np.transpose(values, (0, 2, 1))


def bnh_to_bhn(values: np.ndarray) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError(f"Expected 3D array, got {values.shape}.")
    return np.transpose(values, (0, 2, 1))


def _to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def _validate_bhn(target: Any, pred: Any, mask: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _to_numpy(target).astype(np.float64)
    p = _to_numpy(pred).astype(np.float64)
    m = _to_numpy(mask).astype(bool)
    if p.shape != y.shape or y.shape != m.shape:
        raise ValueError(f"Shapes must match [B,H,N], got pred={p.shape}, target={y.shape}, mask={m.shape}.")
    if p.ndim != 3:
        raise ValueError(f"Expected [B,H,N], got {p.shape}.")
    finite = np.isfinite(p) & np.isfinite(y)
    return y, p, m & finite


def _masked_flat(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pred.shape != target.shape or target.shape != mask.shape:
        raise ValueError(f"Shapes must match, got {pred.shape}, {target.shape}, {mask.shape}.")
    valid = mask.astype(bool) & np.isfinite(pred) & np.isfinite(target)
    return pred[valid].astype(np.float64), target[valid].astype(np.float64)


def masked_official_align_score_kw(
    y_true_kw: Any,
    y_pred_kw: Any,
    mask_bhn: Any,
    num_nodes: int,
    eps: float = 1e-12,
) -> float:
    """Official masked SDWPF alignment score.

    Inputs are strictly [B,H,N] in kW. Errors are converted to MW, aggregated
    over each node's valid horizon positions, then scaled from valid nodes back
    to the protocol node count per sample.
    """

    y, p, mask = _validate_bhn(y_true_kw, y_pred_kw, mask_bhn)
    err_mw = (p - y) / 1000.0
    valid_horizon_count_bn = mask.sum(axis=1).astype(np.float64)
    valid_node_bn = valid_horizon_count_bn > 0
    denom = np.maximum(valid_horizon_count_bn, 1.0)
    mae_bn = (np.abs(err_mw) * mask).sum(axis=1) / denom
    rmse_bn = np.sqrt(((err_mw**2) * mask).sum(axis=1) / denom)
    node_score_bn = 0.5 * (mae_bn + rmse_bn)
    valid_node_count_b = valid_node_bn.sum(axis=1).astype(np.float64)
    valid_sample_b = valid_node_count_b > 0
    if not np.any(valid_sample_b):
        return float("nan")
    sample_score_b = (
        float(num_nodes)
        * (node_score_bn * valid_node_bn).sum(axis=1)
        / np.maximum(valid_node_count_b, eps)
    )
    return float(sample_score_b[valid_sample_b].mean())


def official_align_score(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, num_nodes: int = 134) -> float:
    """Backward-compatible alias for the corrected kW-input official score."""

    return masked_official_align_score_kw(target, pred, mask, num_nodes=num_nodes)


def regression_metrics_kw(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    num_nodes: int = 134,
    eps: float = 1e-6,
    physical_clip_applied: bool = False,
) -> dict[str, float]:
    p, y = _masked_flat(pred, target, mask)
    total_count = int(np.asarray(mask).size)
    valid_count = int((np.asarray(mask).astype(bool) & np.isfinite(pred) & np.isfinite(target)).sum())
    if len(y) == 0:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "R2": float("nan"),
            "SMAPE": float("nan"),
            "MAPE": float("nan"),
            "official_align_score": float("nan"),
            "score": float("nan"),
            "Score": float("nan"),
            "valid_target_count": 0,
            "total_target_count": total_count,
            "valid_target_ratio": 0.0,
            "num_valid_points": 0,
            "score_compute_unit": "MW_after_kW_to_MW_conversion",
            "score_lower_is_better": True,
            "physical_clip_applied": bool(physical_clip_applied),
        }
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err**2)))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if sst <= eps else float(1.0 - np.sum(err**2) / sst)
    smape = float(np.mean(2.0 * np.abs(err) / np.maximum(np.abs(p) + np.abs(y), eps)))
    nonzero = np.abs(y) > eps
    mape = float(np.mean(np.abs(err[nonzero] / y[nonzero]))) if np.any(nonzero) else float("nan")
    official = masked_official_align_score_kw(target, pred, mask, num_nodes=num_nodes)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "SMAPE": smape,
        "MAPE": mape,
        "official_align_score": official,
        "score": official,
        "Score": official,
        "valid_target_count": valid_count,
        "total_target_count": total_count,
        "valid_target_ratio": float(valid_count / total_count) if total_count else float("nan"),
        "num_valid_points": valid_count,
        "score_compute_unit": "MW_after_kW_to_MW_conversion",
        "score_lower_is_better": True,
        "physical_clip_applied": bool(physical_clip_applied),
    }


def evaluate_prefix_horizons(
    pred_bhn_kw: np.ndarray,
    target_bhn_kw: np.ndarray,
    mask_bhn: np.ndarray,
    horizons: Iterable[int],
    num_nodes: int = 134,
    physical_clip_applied: bool = False,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for horizon in horizons:
        metrics = regression_metrics_kw(
            pred_bhn_kw[:, :horizon, :],
            target_bhn_kw[:, :horizon, :],
            mask_bhn[:, :horizon, :],
            num_nodes=num_nodes,
            physical_clip_applied=physical_clip_applied,
        )
        _assert_score_aliases(metrics)
        rows.append({"horizon": int(horizon), **metrics})
    return rows


def evaluate_per_step(
    pred_bhn_kw: np.ndarray,
    target_bhn_kw: np.ndarray,
    mask_bhn: np.ndarray,
    num_nodes: int = 134,
    raw_pred_bhn_kw: np.ndarray | None = None,
    physical_max_kw: float | None = None,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for step in range(pred_bhn_kw.shape[1]):
        metrics = regression_metrics_kw(
            pred_bhn_kw[:, step : step + 1, :],
            target_bhn_kw[:, step : step + 1, :],
            mask_bhn[:, step : step + 1, :],
            num_nodes=num_nodes,
            physical_clip_applied=raw_pred_bhn_kw is not None,
        )
        raw_step = raw_pred_bhn_kw[:, step : step + 1, :] if raw_pred_bhn_kw is not None else pred_bhn_kw[:, step : step + 1, :]
        raw_valid = mask_bhn[:, step : step + 1, :].astype(bool)
        denom = max(int(raw_valid.sum()), 1)
        metrics.update(
            {
                "step": step + 1,
                "bias_kw": float(np.mean((pred_bhn_kw[:, step : step + 1, :] - target_bhn_kw[:, step : step + 1, :])[raw_valid]))
                if raw_valid.any()
                else float("nan"),
                "negative_prediction_ratio_raw": float(((raw_step < 0) & raw_valid).sum() / denom),
                "above_capacity_ratio_raw": (
                    float(((raw_step > physical_max_kw) & raw_valid).sum() / denom)
                    if physical_max_kw is not None
                    else float("nan")
                ),
            }
        )
        rows.append(metrics)
    return rows


def evaluate_per_turbine(
    pred_bhn_kw: np.ndarray,
    target_bhn_kw: np.ndarray,
    mask_bhn: np.ndarray,
    horizons: Iterable[int],
    turbine_ids: Iterable[int] | None = None,
) -> list[dict[str, float | int]]:
    ids = list(turbine_ids or range(pred_bhn_kw.shape[2]))
    rows: list[dict[str, float | int]] = []
    for node_idx, turbine_id in enumerate(ids):
        row: dict[str, float | int] = {"TurbID": int(turbine_id), "node_index": int(node_idx)}
        for horizon in horizons:
            metrics = regression_metrics_kw(
                pred_bhn_kw[:, :horizon, node_idx : node_idx + 1],
                target_bhn_kw[:, :horizon, node_idx : node_idx + 1],
                mask_bhn[:, :horizon, node_idx : node_idx + 1],
                num_nodes=1,
            )
            row[f"MAE_H{horizon}"] = metrics["MAE"]
            row[f"RMSE_H{horizon}"] = metrics["RMSE"]
            row[f"R2_H{horizon}"] = metrics["R2"]
            row[f"Score_H{horizon}"] = metrics["Score"]
        row["valid_target_count"] = int(mask_bhn[:, :, node_idx].astype(bool).sum())
        rows.append(row)
    return rows


def evaluate_by_group(
    pred_bhn_kw: np.ndarray,
    target_bhn_kw: np.ndarray,
    mask_bhn: np.ndarray,
    group_labels: np.ndarray,
    horizons: Iterable[int],
    num_nodes: int = 134,
    gate_mean_by_sample: np.ndarray | None = None,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    labels = np.asarray(group_labels)
    for label in ["low", "medium", "high"]:
        sample_mask = labels == label
        if not np.any(sample_mask):
            continue
        for horizon in horizons:
            metrics = regression_metrics_kw(
                pred_bhn_kw[sample_mask, :horizon, :],
                target_bhn_kw[sample_mask, :horizon, :],
                mask_bhn[sample_mask, :horizon, :],
                num_nodes=num_nodes,
            )
            metrics["volatility_group"] = label
            metrics["horizon"] = int(horizon)
            metrics["gate_mean"] = (
                float(np.nanmean(gate_mean_by_sample[sample_mask])) if gate_mean_by_sample is not None else float("nan")
            )
            rows.append(metrics)
    return rows


def _assert_score_aliases(metrics: dict[str, Any], tol: float = 1e-9) -> None:
    official = metrics.get("official_align_score")
    score = metrics.get("score")
    score_cap = metrics.get("Score")
    if not (np.isfinite(official) and np.isfinite(score) and np.isfinite(score_cap)):
        return
    if abs(float(official) - float(score)) >= tol or abs(float(official) - float(score_cap)) >= tol:
        raise AssertionError("official_align_score, score, and Score must be identical aliases.")
