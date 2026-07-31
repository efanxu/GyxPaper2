from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import STMGPromptConfig
from .data import STMGPromptDataBundle
from .metrics import evaluate_by_group, evaluate_per_step, evaluate_per_turbine, evaluate_prefix_horizons
from .train import _autocast_context, resolve_device


@torch.inference_mode()
def collect_predictions(
    model,
    loader,
    target_scaler,
    device,
    config: STMGPromptConfig | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    pred_norm_parts = []
    pred_kw_raw_parts = []
    pred_kw_eval_parts = []
    y_norm_parts = []
    y_kw_parts = []
    mask_parts = []
    sample_volatility_parts = []
    gate_parts = []
    start_indices = []
    for batch in loader:
        x = batch["x"].to(device).float()
        with _autocast_context(device, bool(getattr(config, "amp_enabled", False))):
            out = model(x)
        pred_norm = out["pred"].detach().float().cpu().numpy()
        y_norm = batch["y"].numpy()
        y_kw = batch["y_raw"].numpy()
        mask = batch["valid_target_mask"].numpy()
        pred_kw_raw = target_scaler.inverse_transform(pred_norm)
        pred_kw_eval = apply_physical_clip(pred_kw_raw, config)
        pred_norm_parts.append(pred_norm)
        pred_kw_raw_parts.append(pred_kw_raw)
        pred_kw_eval_parts.append(pred_kw_eval)
        y_norm_parts.append(y_norm)
        y_kw_parts.append(y_kw)
        mask_parts.append(mask)
        sample_volatility_parts.append(_historical_window_volatility(batch["x"].numpy()))
        if "dynamic_patch_gate" in out["aux"]:
            gate_parts.append(out["aux"]["dynamic_patch_gate"].detach().cpu().numpy().mean(axis=(1, 2, 3)))
        start_indices.extend(batch["prediction_start_index"].numpy().tolist())
    pred_kw_eval_bhn = np.concatenate(pred_kw_eval_parts, axis=0)
    arrays = {
        "pred_norm_bhn": np.concatenate(pred_norm_parts, axis=0),
        "pred_kw_raw_bhn": np.concatenate(pred_kw_raw_parts, axis=0),
        "pred_kw_eval_bhn": pred_kw_eval_bhn,
        "pred_kw_bhn": pred_kw_eval_bhn,
        "y_norm_bhn": np.concatenate(y_norm_parts, axis=0),
        "y_kw_bhn": np.concatenate(y_kw_parts, axis=0),
        "mask_bhn": np.concatenate(mask_parts, axis=0),
        "sample_history_volatility": np.concatenate(sample_volatility_parts, axis=0),
        "prediction_start_index": np.asarray(start_indices, dtype=np.int64),
    }
    if gate_parts:
        arrays["gate_mean_by_sample"] = np.concatenate(gate_parts, axis=0)
    return arrays


def apply_physical_clip(pred_kw: np.ndarray, config: STMGPromptConfig | None) -> np.ndarray:
    if config is None or not getattr(config, "enable_physical_clip_eval", True):
        return pred_kw.astype(np.float32, copy=True)
    min_kw = getattr(config, "physical_power_min_kw", None)
    max_kw = getattr(config, "physical_power_max_kw", None)
    if min_kw is None or max_kw is None:
        raise ValueError("physical_power_min_kw and physical_power_max_kw are required when eval clipping is enabled.")
    return np.clip(pred_kw, float(min_kw), float(max_kw)).astype(np.float32)


def _historical_window_volatility(x_blnc: np.ndarray) -> np.ndarray:
    if x_blnc.shape[1] <= 1:
        return np.zeros((x_blnc.shape[0],), dtype=np.float32)
    delta = np.abs(np.diff(x_blnc.astype(np.float64), axis=1))
    return np.nanmean(delta, axis=(1, 2, 3)).astype(np.float32)


def volatility_groups(sample_volatility: np.ndarray) -> np.ndarray:
    values = np.asarray(sample_volatility, dtype=np.float64)
    if values.size == 0:
        return np.asarray([], dtype=object)
    q33, q67 = np.nanquantile(values, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.full(values.shape, "medium", dtype=object)
    labels[values <= q33] = "low"
    labels[values > q67] = "high"
    return labels


def prediction_range_diagnostics(
    pred_kw_raw_bhn: np.ndarray,
    pred_kw_eval_bhn: np.ndarray,
    mask_bhn: np.ndarray,
    config: STMGPromptConfig,
) -> dict[str, Any]:
    valid = mask_bhn.astype(bool)
    denom = max(int(valid.sum()), 1)
    min_kw = float(config.physical_power_min_kw)
    max_kw = float(config.physical_power_max_kw)
    changed = np.abs(pred_kw_eval_bhn - pred_kw_raw_bhn) > 1e-7
    out: dict[str, Any] = {
        "raw_negative_prediction_count": int(((pred_kw_raw_bhn < min_kw) & valid).sum()),
        "raw_negative_prediction_ratio": float(((pred_kw_raw_bhn < min_kw) & valid).sum() / denom),
        "raw_above_capacity_count": int(((pred_kw_raw_bhn > max_kw) & valid).sum()),
        "raw_above_capacity_ratio": float(((pred_kw_raw_bhn > max_kw) & valid).sum() / denom),
        "clip_changed_count": int((changed & valid).sum()),
        "clip_changed_ratio": float((changed & valid).sum() / denom),
        "raw_prediction_min_kw": float(np.nanmin(pred_kw_raw_bhn)),
        "raw_prediction_max_kw": float(np.nanmax(pred_kw_raw_bhn)),
        "eval_prediction_min_kw": float(np.nanmin(pred_kw_eval_bhn)),
        "eval_prediction_max_kw": float(np.nanmax(pred_kw_eval_bhn)),
        "physical_power_min_kw": min_kw,
        "physical_power_max_kw": max_kw,
    }
    per_step = []
    for step in range(pred_kw_raw_bhn.shape[1]):
        step_valid = valid[:, step, :]
        step_denom = max(int(step_valid.sum()), 1)
        per_step.append(
            {
                "step": step + 1,
                "raw_negative_prediction_count": int(((pred_kw_raw_bhn[:, step, :] < min_kw) & step_valid).sum()),
                "raw_negative_prediction_ratio": float(((pred_kw_raw_bhn[:, step, :] < min_kw) & step_valid).sum() / step_denom),
                "raw_above_capacity_count": int(((pred_kw_raw_bhn[:, step, :] > max_kw) & step_valid).sum()),
                "raw_above_capacity_ratio": float(((pred_kw_raw_bhn[:, step, :] > max_kw) & step_valid).sum() / step_denom),
                "clip_changed_count": int((changed[:, step, :] & step_valid).sum()),
                "clip_changed_ratio": float((changed[:, step, :] & step_valid).sum() / step_denom),
            }
        )
    out["per_step"] = per_step
    return out


def save_metrics(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "horizon",
        "step",
        "volatility_group",
        "MAE",
        "RMSE",
        "R2",
        "SMAPE",
        "MAPE",
        "official_align_score",
        "score",
        "Score",
        "bias_kw",
        "valid_target_count",
        "total_target_count",
        "valid_target_ratio",
        "num_valid_points",
        "score_compute_unit",
        "score_lower_is_better",
        "physical_clip_applied",
        "negative_prediction_ratio_raw",
        "above_capacity_ratio_raw",
        "gate_mean",
    ]
    extra = sorted({key for row in rows for key in row if key not in preferred})
    fieldnames = [key for key in preferred if any(key in row for row in rows)] + extra
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_csv_value(row.get(k)) for k in fieldnames})


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.3f}"
    return value


@torch.inference_mode()
def evaluate_model_streaming(
    config: STMGPromptConfig,
    data: STMGPromptDataBundle,
    model,
    run_dir: Path,
    stage_callback=None,
) -> list[dict]:
    device = resolve_device(config.device)
    model.to(device)
    model.eval()
    accum = {int(h): _new_stream_accumulator() for h in config.eval_horizons}
    sample_count = 0
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    infer_start = time.perf_counter()
    for batch in data.test_loader:
        x = batch["x"].to(device).float()
        with _autocast_context(device, bool(getattr(config, "amp_enabled", False))):
            out = model(x)
        pred_norm = out["pred"].detach().float().cpu().numpy()
        pred_kw = data.scalers["target"].inverse_transform(pred_norm).astype(np.float32)
        pred_kw = apply_physical_clip(pred_kw, config).astype(np.float32)
        y_kw = batch["y_raw"].numpy().astype(np.float32)
        mask = batch["valid_target_mask"].numpy().astype(bool)
        sample_count += int(pred_kw.shape[0])
        for horizon in config.eval_horizons:
            _update_stream_accumulator(
                accum[int(horizon)],
                pred_kw[:, : int(horizon), :],
                y_kw[:, : int(horizon), :],
                mask[:, : int(horizon), :],
                num_nodes=config.num_nodes,
            )
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    inference_time_sec = time.perf_counter() - infer_start
    rows = [
        {"horizon": horizon, **_finalize_stream_accumulator(item, physical_clip_applied=bool(config.enable_physical_clip_eval))}
        for horizon, item in accum.items()
    ]
    if stage_callback:
        stage_callback("TEST_METRICS_WRITE_STARTED")
    save_metrics(rows, run_dir / "metrics.csv")
    for row in rows:
        horizon = int(row["horizon"])
        (run_dir / f"metrics_eval_h{horizon}.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if stage_callback:
        stage_callback("TEST_METRICS_WRITE_FINISHED")
    metadata = {
        "metrics_compute_space": "kW inputs; official Score converts error to MW internally",
        "target_inverse_transform_before_metrics": True,
        "masked_metrics": True,
        "physical_clip_applied": bool(config.enable_physical_clip_eval),
        "physical_clip_protocol": config.physical_clip_protocol,
        "score_compute_unit": "MW_after_kW_to_MW_conversion",
        "score_lower_is_better": True,
        "artifact_source_checkpoint": "best_checkpoint.pt",
        "prediction_accumulation": "streaming",
        "inference_time_sec_full_test": inference_time_sec,
        "inference_time_ms_per_window": float(1000.0 * inference_time_sec / max(sample_count, 1)),
        "inference_windows_per_sec": float(sample_count / max(inference_time_sec, 1e-12)),
    }
    if stage_callback:
        stage_callback("PREDICTION_METADATA_WRITE_STARTED")
    (run_dir / "prediction_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if stage_callback:
        stage_callback("PREDICTION_METADATA_WRITE_FINISHED")
    _write_evaluation_complete(config, run_dir, stage_callback=stage_callback)
    return rows


def _new_stream_accumulator() -> dict[str, float | int]:
    return {
        "abs_err_sum": 0.0,
        "sq_err_sum": 0.0,
        "y_sum": 0.0,
        "y_sq_sum": 0.0,
        "smape_sum": 0.0,
        "mape_sum": 0.0,
        "mape_count": 0,
        "valid_count": 0,
        "total_count": 0,
        "official_sum": 0.0,
        "official_count": 0,
    }


def _update_stream_accumulator(acc: dict, pred: np.ndarray, target: np.ndarray, mask: np.ndarray, num_nodes: int) -> None:
    valid = mask.astype(bool) & np.isfinite(pred) & np.isfinite(target)
    acc["total_count"] += int(mask.size)
    acc["valid_count"] += int(valid.sum())
    if not np.any(valid):
        return
    p = pred.astype(np.float64, copy=False)
    y = target.astype(np.float64, copy=False)
    err = p - y
    acc["abs_err_sum"] += float(np.abs(err[valid]).sum())
    acc["sq_err_sum"] += float((err[valid] ** 2).sum())
    acc["y_sum"] += float(y[valid].sum())
    acc["y_sq_sum"] += float((y[valid] ** 2).sum())
    acc["smape_sum"] += float((2.0 * np.abs(err[valid]) / np.maximum(np.abs(p[valid]) + np.abs(y[valid]), 1e-6)).sum())
    nonzero = valid & (np.abs(y) > 1e-6)
    acc["mape_sum"] += float((np.abs(err[nonzero] / y[nonzero])).sum()) if np.any(nonzero) else 0.0
    acc["mape_count"] += int(nonzero.sum())

    err_mw = err / 1000.0
    valid_horizon_count_bn = valid.sum(axis=1).astype(np.float64)
    valid_node_bn = valid_horizon_count_bn > 0
    denom = np.maximum(valid_horizon_count_bn, 1.0)
    mae_bn = (np.abs(err_mw) * valid).sum(axis=1) / denom
    rmse_bn = np.sqrt(((err_mw**2) * valid).sum(axis=1) / denom)
    node_score_bn = 0.5 * (mae_bn + rmse_bn)
    valid_node_count_b = valid_node_bn.sum(axis=1).astype(np.float64)
    valid_sample_b = valid_node_count_b > 0
    sample_score_b = float(num_nodes) * (node_score_bn * valid_node_bn).sum(axis=1) / np.maximum(valid_node_count_b, 1e-12)
    acc["official_sum"] += float(sample_score_b[valid_sample_b].sum())
    acc["official_count"] += int(valid_sample_b.sum())


def _finalize_stream_accumulator(acc: dict, physical_clip_applied: bool) -> dict[str, float | int | bool | str]:
    valid_count = int(acc["valid_count"])
    total_count = int(acc["total_count"])
    if valid_count == 0:
        mae = rmse = r2 = smape = mape = official = float("nan")
    else:
        mae = float(acc["abs_err_sum"] / valid_count)
        rmse = float(math.sqrt(acc["sq_err_sum"] / valid_count))
        mean_y = float(acc["y_sum"] / valid_count)
        sst = float(acc["y_sq_sum"] - valid_count * mean_y * mean_y)
        r2 = float("nan") if sst <= 1e-6 else float(1.0 - acc["sq_err_sum"] / sst)
        smape = float(acc["smape_sum"] / valid_count)
        mape = float(acc["mape_sum"] / max(int(acc["mape_count"]), 1)) if int(acc["mape_count"]) else float("nan")
        official = float(acc["official_sum"] / max(int(acc["official_count"]), 1))
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


def _write_evaluation_complete(config: STMGPromptConfig, run_dir: Path, stage_callback=None) -> None:
    if stage_callback:
        stage_callback("EVALUATION_COMPLETE_WRITE_STARTED")
    (run_dir / "evaluation_complete.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "checkpoint": "best_checkpoint.pt",
                "eval_horizons": list(config.eval_horizons),
                "diagnostics_level": getattr(config, "diagnostics_level", "standard"),
                "prediction_accumulation": getattr(config, "prediction_accumulation", "full"),
                "amp_enabled": bool(getattr(config, "amp_enabled", False)),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if stage_callback:
        stage_callback("EVALUATION_COMPLETE_WRITE_FINISHED")


def evaluate_model(config: STMGPromptConfig, data: STMGPromptDataBundle, model, run_dir: Path, stage_callback=None) -> list[dict]:
    if getattr(config, "prediction_accumulation", "full") == "streaming":
        return evaluate_model_streaming(config, data, model, run_dir, stage_callback=stage_callback)
    device = resolve_device(config.device)
    model.to(device)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    infer_start = time.perf_counter()
    arrays = collect_predictions(model, data.test_loader, data.scalers["target"], device, config=config)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    inference_time_sec = time.perf_counter() - infer_start
    rows = evaluate_prefix_horizons(
        arrays["pred_kw_eval_bhn"],
        arrays["y_kw_bhn"],
        arrays["mask_bhn"],
        config.eval_horizons,
        num_nodes=config.num_nodes,
        physical_clip_applied=bool(config.enable_physical_clip_eval),
    )
    raw_rows = evaluate_prefix_horizons(
        arrays["pred_kw_raw_bhn"],
        arrays["y_kw_bhn"],
        arrays["mask_bhn"],
        config.eval_horizons,
        num_nodes=config.num_nodes,
        physical_clip_applied=False,
    )
    if stage_callback:
        stage_callback("TEST_METRICS_WRITE_STARTED")
    save_metrics(rows, run_dir / "metrics.csv")
    save_metrics(raw_rows, run_dir / "metrics_raw.csv")
    for row in rows:
        horizon = int(row["horizon"])
        (run_dir / f"metrics_eval_h{horizon}.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if stage_callback:
        stage_callback("TEST_METRICS_WRITE_FINISHED")
    minimal_eval = getattr(config, "diagnostics_level", "standard") in {"none", "minimal"} or getattr(
        config, "prediction_accumulation", "full"
    ) == "streaming"
    if not minimal_eval:
        save_metrics(
            evaluate_per_step(
                arrays["pred_kw_eval_bhn"],
                arrays["y_kw_bhn"],
                arrays["mask_bhn"],
                num_nodes=config.num_nodes,
                raw_pred_bhn_kw=arrays["pred_kw_raw_bhn"],
                physical_max_kw=float(config.physical_power_max_kw),
            ),
            run_dir / "metrics_per_step.csv",
        )
        save_metrics(
            evaluate_per_turbine(
                arrays["pred_kw_eval_bhn"],
                arrays["y_kw_bhn"],
                arrays["mask_bhn"],
                config.eval_horizons,
                turbine_ids=data.turbine_ids,
            ),
            run_dir / "metrics_per_turbine.csv",
        )
        save_metrics(
            evaluate_by_group(
                arrays["pred_kw_eval_bhn"],
                arrays["y_kw_bhn"],
                arrays["mask_bhn"],
                volatility_groups(arrays["sample_history_volatility"]),
                config.eval_horizons,
                num_nodes=config.num_nodes,
                gate_mean_by_sample=arrays.get("gate_mean_by_sample"),
            ),
            run_dir / "metrics_by_volatility.csv",
        )
        np.savez_compressed(run_dir / "predictions.npz", **arrays)
    range_diag = prediction_range_diagnostics(arrays["pred_kw_raw_bhn"], arrays["pred_kw_eval_bhn"], arrays["mask_bhn"], config)
    metadata = {
        "shape_protocol": {
            "model_pred": "[B,H,N]",
            "convertible_wrapper_pred": "[B,N,H]",
            "saved_arrays": {
                "pred_norm_bhn": list(arrays["pred_norm_bhn"].shape),
                "pred_kw_raw_bhn": list(arrays["pred_kw_raw_bhn"].shape),
                "pred_kw_eval_bhn": list(arrays["pred_kw_eval_bhn"].shape),
                "pred_kw_bhn": "compatibility alias of pred_kw_eval_bhn",
                "y_kw_bhn": list(arrays["y_kw_bhn"].shape),
                "mask_bhn": list(arrays["mask_bhn"].shape),
            },
        },
        "metrics_compute_space": "kW inputs; official Score converts error to MW internally",
        "target_inverse_transform_before_metrics": True,
        "masked_metrics": True,
        "physical_clip_applied": bool(config.enable_physical_clip_eval),
        "physical_clip_protocol": config.physical_clip_protocol,
        "physical_range_diagnostic": range_diag,
        "score_compute_unit": "MW_after_kW_to_MW_conversion",
        "score_lower_is_better": True,
        "artifact_source_checkpoint": "best_checkpoint.pt",
        "inference_time_sec_full_test": inference_time_sec,
        "inference_time_ms_per_window": float(1000.0 * inference_time_sec / max(arrays["pred_kw_eval_bhn"].shape[0], 1)),
        "inference_windows_per_sec": float(arrays["pred_kw_eval_bhn"].shape[0] / max(inference_time_sec, 1e-12)),
        "inference_valid_points_per_sec": float(
            arrays["mask_bhn"].astype(bool).sum() / max(inference_time_sec, 1e-12)
        ),
    }
    if stage_callback:
        stage_callback("PREDICTION_METADATA_WRITE_STARTED")
    (run_dir / "prediction_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if stage_callback:
        stage_callback("PREDICTION_METADATA_WRITE_FINISHED")
    _write_evaluation_complete(config, run_dir, stage_callback=stage_callback)
    return rows
