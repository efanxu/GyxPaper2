from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "custom_models" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from st_mgprompt.config import STMGPromptConfig
from st_mgprompt.data import (
    PreparedArrays,
    _make_synthetic_arrays,
    _pivot_arrays,
    build_window_start_indices,
    compute_split_indices,
    load_eval_target,
    load_model_input,
)
from st_mgprompt.experiment_protocol import canonical_config
from st_mgprompt.graph_prior import (
    build_distance_prior_graph,
    build_macro_trend_graph,
    build_micro_local_graph,
)
from st_mgprompt.metrics import evaluate_per_step, evaluate_prefix_horizons, regression_metrics_kw


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "custom_models" / "results" / "empirical_analysis_v1" / "data_evidence"
DEFAULT_BASELINE_EXPORT = (
    PROJECT_ROOT
    / "custom_models"
    / "results"
    / "benchmark_v2_uniform_bs4"
    / "prediction_exports"
    / "movingaverage_predictions.npz"
)
CAUSAL_WINDOWS = (6, 18, 36)
EVAL_HORIZONS = (3, 6, 10)
EPS = 1e-12


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="")


def causal_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Causal NaN-aware mean at t over [max(0,t-window+1), t]."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"values must be [T,N], got {array.shape}")
    if window <= 0:
        raise ValueError("window must be positive")
    finite = np.isfinite(array)
    sums = np.concatenate(
        [np.zeros((1, array.shape[1])), np.cumsum(np.where(finite, array, 0.0), axis=0)], axis=0
    )
    counts = np.concatenate(
        [np.zeros((1, array.shape[1])), np.cumsum(finite.astype(np.int64), axis=0)], axis=0
    )
    end = np.arange(1, array.shape[0] + 1)
    start = np.maximum(end - int(window), 0)
    rolling_sum = sums[end] - sums[start]
    rolling_count = counts[end] - counts[start]
    result = rolling_sum / np.maximum(rolling_count, 1)
    result[rolling_count == 0] = np.nan
    return result.astype(np.float32)


def causal_downsample_upsample(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal block aggregation followed by hold-upsample with source provenance.

    After the first block, every position in block k uses only the completed
    block k-1.  The first block uses the causal prefix available at each t.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"values must be [T,N], got {array.shape}")
    if window <= 0:
        raise ValueError("window must be positive")
    output = np.full(array.shape, np.nan, dtype=np.float32)
    source_start = np.zeros(array.shape[0], dtype=np.int64)
    source_end = np.zeros(array.shape[0], dtype=np.int64)
    first_end = min(int(window), array.shape[0])
    output[:first_end] = causal_rolling_mean(array[:first_end], int(window))
    source_end[:first_end] = np.arange(first_end, dtype=np.int64)
    for block_start in range(int(window), array.shape[0], int(window)):
        block_end = min(block_start + int(window), array.shape[0])
        previous_start = block_start - int(window)
        previous_end = block_start
        block = array[previous_start:previous_end]
        finite = np.isfinite(block)
        count = finite.sum(axis=0)
        block_value = np.where(
            count > 0,
            np.where(finite, block, 0.0).sum(axis=0) / np.maximum(count, 1),
            np.nan,
        )
        output[block_start:block_end] = block_value
        source_start[block_start:block_end] = previous_start
        source_end[block_start:block_end] = previous_end - 1
    return output, source_start, source_end


def _first_difference_std(values: np.ndarray) -> np.ndarray:
    current = np.asarray(values, dtype=np.float64)
    left, right = current[:-1], current[1:]
    valid = np.isfinite(left) & np.isfinite(right)
    delta = np.where(valid, right - left, np.nan)
    return np.nanstd(delta, axis=0)


def _column_fill_and_center(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    count = finite.sum(axis=0)
    mean = np.divide(
        np.where(finite, array, 0.0).sum(axis=0),
        np.maximum(count, 1),
        out=np.zeros(array.shape[1], dtype=np.float64),
    )
    filled = np.where(finite, array, mean)
    return filled - filled.mean(axis=0, keepdims=True)


def spectral_acf_metrics(
    values: np.ndarray,
    *,
    high_frequency_cutoff: float = 1.0 / 18.0,
    max_lag: int = 144,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-node high-frequency energy share and |ACF| 1/e decay lag."""

    centered = _column_fill_and_center(values)
    if centered.shape[0] < 3:
        size = centered.shape[1]
        return np.full(size, np.nan), np.full(size, np.nan)
    spectrum = np.fft.rfft(centered, axis=0)
    energy = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(centered.shape[0])
    non_dc = frequencies > 0
    high = frequencies >= float(high_frequency_cutoff)
    total = energy[non_dc].sum(axis=0)
    high_share = energy[high].sum(axis=0) / np.maximum(total, EPS)

    autocov = np.fft.irfft(np.abs(spectrum) ** 2, n=centered.shape[0], axis=0)
    variance = np.maximum(autocov[0], EPS)
    lag_limit = min(int(max_lag), centered.shape[0] - 1)
    acf = autocov[: lag_limit + 1] / variance
    decay = np.full(centered.shape[1], float(lag_limit), dtype=np.float64)
    below = np.abs(acf[1:]) <= math.exp(-1.0)
    for node_index in range(centered.shape[1]):
        hits = np.flatnonzero(below[:, node_index])
        if hits.size:
            decay[node_index] = float(hits[0] + 1)
    return high_share, decay


def _pearson_columns(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if left.shape != right.shape:
        raise ValueError(f"association shapes differ: {left.shape} != {right.shape}")
    result = np.full(left.shape[1], np.nan, dtype=np.float64)
    counts = np.zeros(left.shape[1], dtype=np.int64)
    for node_index in range(left.shape[1]):
        valid = np.isfinite(left[:, node_index]) & np.isfinite(right[:, node_index])
        counts[node_index] = int(valid.sum())
        if counts[node_index] < 3:
            continue
        a = left[valid, node_index].astype(np.float64)
        b = right[valid, node_index].astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom > EPS:
            result[node_index] = float(np.dot(a, b) / denom)
    return result, counts


def _future_target_mean(target: np.ndarray, mask: np.ndarray, horizon: int) -> np.ndarray:
    """At row t, average strictly future targets [t+1, t+horizon]."""

    valid = mask.astype(bool) & np.isfinite(target)
    sums = np.concatenate(
        [np.zeros((1, target.shape[1])), np.cumsum(np.where(valid, target, 0.0), axis=0)], axis=0
    )
    counts = np.concatenate(
        [np.zeros((1, target.shape[1])), np.cumsum(valid.astype(np.int64), axis=0)], axis=0
    )
    starts = np.arange(1, target.shape[0] - horizon + 1)
    ends = starts + int(horizon)
    total = sums[ends] - sums[starts]
    count = counts[ends] - counts[starts]
    result = total / np.maximum(count, 1)
    result[count == 0] = np.nan
    return result


def _descriptive(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "valid_count": 0,
            "mean": np.nan,
            "std": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q50": np.nan,
            "q75": np.nan,
            "q95": np.nan,
        }
    quantiles = np.quantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "valid_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
    }


def _load_arrays(config: STMGPromptConfig, smoke: bool) -> PreparedArrays:
    if smoke:
        return _make_synthetic_arrays(config)
    return _pivot_arrays(load_model_input(config), load_eval_target(config), config)


def _split_and_windows(
    prepared: PreparedArrays, config: STMGPromptConfig
) -> tuple[dict[str, tuple[int, int]], dict[str, list[int]]]:
    splits = compute_split_indices(len(prepared.timestamps), config.split_ratios)
    strides = {
        "train": config.train_sample_stride,
        "val": config.val_sample_stride,
        "test": config.test_sample_stride,
    }
    windows = {
        name: build_window_start_indices(start, end, config.lookback, config.max_pred_len, strides[name])
        for name, (start, end) in splits.items()
    }
    return splits, windows


def build_data_overview(
    prepared: PreparedArrays,
    splits: dict[str, tuple[int, int]],
    windows: dict[str, list[int]],
    config: STMGPromptConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in ("train", "val", "test"):
        start, end = splits[name]
        split_mask = prepared.mask[start:end].astype(bool)
        rows.append(
            {
                "split": name,
                "start_index_inclusive": start,
                "end_index_exclusive": end,
                "start_timestamp": prepared.timestamps[start],
                "end_timestamp": prepared.timestamps[end - 1],
                "time_points": end - start,
                "window_count": len(windows[name]),
                "prediction_start_stride": getattr(config, f"{name}_sample_stride"),
                "valid_target_count": int(split_mask.sum()),
                "total_target_count": int(split_mask.size),
                "valid_target_ratio": float(split_mask.mean()),
                "lookback": config.lookback,
                "max_pred_len": config.max_pred_len,
                "window_rule": "[t-lookback,t)->[t,t+max_pred_len)",
            }
        )
    return pd.DataFrame(rows)


def build_turbine_summary(
    prepared: PreparedArrays,
    train_end: int,
    config: STMGPromptConfig,
) -> pd.DataFrame:
    feature_index = {name: index for index, name in enumerate(prepared.feature_cols)}
    signals = {
        "Patv_raw": np.where(prepared.mask[:train_end].astype(bool), prepared.y_raw[:train_end], np.nan),
        "Wspd": prepared.x_raw[:train_end, :, feature_index["Wspd"]],
        "Patv_clean_for_input": prepared.x_raw[:train_end, :, feature_index[config.input_patv_col]],
    }
    rows: list[dict[str, Any]] = []
    for signal_name, values in signals.items():
        diff_std = _first_difference_std(values)
        for node_index, turbine_id in enumerate(prepared.turbine_ids):
            summary = _descriptive(values[:, node_index])
            rows.append(
                {
                    "TurbID": int(turbine_id),
                    "node_index": node_index,
                    "signal": signal_name,
                    "fit_split": "train",
                    "total_count": int(values.shape[0]),
                    "valid_ratio": float(np.isfinite(values[:, node_index]).mean()),
                    "first_difference_std": float(diff_std[node_index]),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def _temporal_representations(values: np.ndarray) -> dict[str, tuple[np.ndarray, int | None, str]]:
    result: dict[str, tuple[np.ndarray, int | None, str]] = {"raw": (values, None, "raw")}
    for window in CAUSAL_WINDOWS:
        trend = causal_rolling_mean(values, window)
        result[f"trend_ma{window}"] = (trend, window, "trend")
        result[f"local_residual_ma{window}"] = (values - trend, window, "local_residual")
    return result


def build_temporal_evidence(
    prepared: PreparedArrays,
    train_end: int,
    config: STMGPromptConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, tuple[np.ndarray, int | None, str]]]]:
    feature_index = {name: index for index, name in enumerate(prepared.feature_cols)}
    target = np.where(prepared.mask[:train_end].astype(bool), prepared.y_raw[:train_end], np.nan)
    signals = {
        "Patv_raw": target,
        "Wspd": prepared.x_raw[:train_end, :, feature_index["Wspd"]],
    }
    all_representations = {name: _temporal_representations(values) for name, values in signals.items()}
    statistic_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    for signal_name, representations in all_representations.items():
        for representation_name, (values, window, component) in representations.items():
            high_share, decay = spectral_acf_metrics(values)
            diff_std = _first_difference_std(values)
            for node_index, turbine_id in enumerate(prepared.turbine_ids):
                statistic_rows.append(
                    {
                        "TurbID": int(turbine_id),
                        "node_index": node_index,
                        "signal": signal_name,
                        "representation": representation_name,
                        "component": component,
                        "causal_window": window,
                        "fit_split": "train",
                        "valid_count": int(np.isfinite(values[:, node_index]).sum()),
                        "high_frequency_energy_ratio": float(high_share[node_index]),
                        "high_frequency_cutoff_cycles_per_step": 1.0 / 18.0,
                        "acf_decay_length_steps": float(decay[node_index]),
                        "first_difference_std": float(diff_std[node_index]),
                    }
                )
            for horizon in EVAL_HORIZONS:
                future_mean = _future_target_mean(target, prepared.mask[:train_end], horizon)
                feature_values = values[: future_mean.shape[0]]
                correlations, valid_counts = _pearson_columns(feature_values, future_mean)
                for node_index, turbine_id in enumerate(prepared.turbine_ids):
                    association_rows.append(
                        {
                            "TurbID": int(turbine_id),
                            "node_index": node_index,
                            "signal": signal_name,
                            "representation": representation_name,
                            "component": component,
                            "causal_window": window,
                            "future_horizon": horizon,
                            "association_method": "pearson",
                            "association": float(correlations[node_index]),
                            "absolute_association": float(abs(correlations[node_index])),
                            "valid_pair_count": int(valid_counts[node_index]),
                            "feature_information_end_offset": 0,
                            "target_start_offset": 1,
                            "target_end_offset": horizon,
                            "fit_split": "train",
                        }
                    )
    return pd.DataFrame(statistic_rows), pd.DataFrame(association_rows), all_representations


def _farm_mean(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    total = np.where(finite, values, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _select_ramp_events(series: np.ndarray, *, count: int, radius: int) -> list[int]:
    ramp = np.abs(np.diff(series, prepend=np.nan))
    candidates = np.arange(radius, len(series) - radius)
    ordered = candidates[np.argsort(np.nan_to_num(ramp[candidates], nan=-np.inf))[::-1]]
    selected: list[int] = []
    for index in ordered:
        if not np.isfinite(ramp[index]):
            continue
        if all(abs(int(index) - other) > radius for other in selected):
            selected.append(int(index))
        if len(selected) >= count:
            break
    return selected


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    a, b = left[valid].astype(np.float64), right[valid].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > EPS else float("nan")


def _peak_offset(reference: np.ndarray, candidate: np.ndarray) -> int:
    ref_delta = np.abs(np.diff(reference))
    candidate_delta = np.abs(np.diff(candidate))
    if not np.isfinite(ref_delta).any() or not np.isfinite(candidate_delta).any():
        return 0
    return int(np.nanargmax(candidate_delta) - np.nanargmax(ref_delta))


def build_alignment_evidence(
    prepared: PreparedArrays,
    train_end: int,
    config: STMGPromptConfig,
    *,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    patv_index = prepared.feature_cols.index(config.input_patv_col)
    model_input_power = prepared.x_raw[:train_end, :, patv_index]
    raw = _farm_mean(model_input_power)[:, None]
    coarse = {window: causal_rolling_mean(raw, window)[:, 0] for window in CAUSAL_WINDOWS}
    resampled, source_start, source_end = causal_downsample_upsample(raw, 36)
    resampled_farm = resampled[:, 0]
    radius = 36
    events = _select_ramp_events(raw[:, 0], count=4 if smoke else 24, radius=radius)
    if not events:
        raise RuntimeError("No finite training ramp event was available for alignment analysis.")
    representations = {
        "coaxial_ma6": coarse[6],
        "coaxial_ma18": coarse[18],
        "coaxial_ma36": coarse[36],
        "causal_downsample_upsample_ma36": resampled_farm,
    }
    metric_rows: list[dict[str, Any]] = []
    ramp = np.diff(raw[:, 0], prepend=np.nan)
    for rank, event_index in enumerate(events, start=1):
        start, end = event_index - radius, event_index + radius + 1
        reference = raw[start:end, 0]
        for representation_name, series in representations.items():
            candidate = series[start:end]
            valid = np.isfinite(reference) & np.isfinite(candidate)
            rmse = float(np.sqrt(np.mean((candidate[valid] - reference[valid]) ** 2))) if valid.any() else np.nan
            metric_rows.append(
                {
                    "event_rank_by_abs_train_ramp": rank,
                    "event_index": event_index,
                    "event_timestamp": prepared.timestamps[event_index],
                    "ramp_kw_per_10min": float(ramp[event_index]),
                    "absolute_ramp_kw_per_10min": float(abs(ramp[event_index])),
                    "representation": representation_name,
                    "alignment_mode": "coaxial" if representation_name.startswith("coaxial") else "causal_downsample_upsample",
                    "peak_timing_error_steps": _peak_offset(reference, candidate),
                    "local_correlation": _safe_corr(reference, candidate),
                    "reconstruction_rmse_kw": rmse,
                    "window_radius_steps": radius,
                    "source_signal": config.input_patv_col,
                    "fit_split": "train",
                }
            )

    typical = events[0]
    curve_rows: list[dict[str, Any]] = []
    ma6 = coarse[6]
    for time_index in range(typical - radius, typical + radius + 1):
        curve_rows.append(
            {
                "event_index": typical,
                "event_timestamp": prepared.timestamps[typical],
                "time_index": time_index,
                "relative_step": time_index - typical,
                "timestamp": prepared.timestamps[time_index],
                "Patv_clean_for_input_farm_mean_kw": float(raw[time_index, 0]),
                "Fine_local_residual_ma6_kw": float(raw[time_index, 0] - ma6[time_index]),
                "Fine_first_difference_kw": float(ramp[time_index]),
                "Coarse_ma6_kw": float(coarse[6][time_index]),
                "Coarse_ma18_kw": float(coarse[18][time_index]),
                "Coarse_ma36_kw": float(coarse[36][time_index]),
                "Downsample_upsample_ma36_kw": float(resampled_farm[time_index]),
                "resampled_source_start_index": int(source_start[time_index]),
                "resampled_source_end_index": int(source_end[time_index]),
                "resampled_source_is_causal": bool(source_end[time_index] <= time_index),
            }
        )
    audit = {
        "event_selection": "descending absolute one-step farm-mean Patv_clean_for_input ramp on train, separated by >36 steps",
        "alignment_source_signal": config.input_patv_col,
        "typical_event_rule": "rank 1 maximum absolute eligible training ramp",
        "selected_event_count": len(events),
        "typical_event_index": typical,
        "max_resampled_source_end_index": int(source_end.max()),
        "resampled_all_sources_causal": bool(np.all(source_end <= np.arange(train_end))),
    }
    return pd.DataFrame(metric_rows), pd.DataFrame(curve_rows), audit


def build_spectrum_acf_source(
    prepared: PreparedArrays,
    train_end: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = np.where(prepared.mask[:train_end].astype(bool), prepared.y_raw[:train_end], np.nan)
    volatility = _first_difference_std(target)
    node_index = int(np.nanargmax(volatility))
    turbine_id = int(prepared.turbine_ids[node_index])
    raw = target[:, node_index : node_index + 1]
    representations = {
        "raw": raw,
        "trend_ma36": causal_rolling_mean(raw, 36),
        "local_residual_ma6": raw - causal_rolling_mean(raw, 6),
    }
    rows: list[dict[str, Any]] = []
    for name, values in representations.items():
        centered = _column_fill_and_center(values)[:, 0]
        spectrum = np.fft.rfft(centered)
        frequencies = np.fft.rfftfreq(centered.size)
        power = np.abs(spectrum) ** 2
        power /= max(float(power[1:].sum()), EPS)
        autocov = np.fft.irfft(np.abs(spectrum) ** 2, n=centered.size)
        acf = autocov / max(float(autocov[0]), EPS)
        bin_edges = np.linspace(0.0, 0.5, 257)
        bin_ids = np.digitize(frequencies, bin_edges, right=False) - 1
        for bin_index in range(256):
            members = (bin_ids == bin_index) & (frequencies > 0)
            if not np.any(members):
                continue
            rows.append(
                {
                    "TurbID": turbine_id,
                    "representation": name,
                    "curve_type": "spectrum",
                    "x": float((bin_edges[bin_index] + bin_edges[bin_index + 1]) / 2.0),
                    "y": float(power[members].sum()),
                    "x_unit": "cycles_per_10min_step",
                    "y_unit": "share_of_non_dc_energy_per_frequency_bin",
                }
            )
        for lag in range(0, min(144, len(acf) - 1) + 1):
            rows.append(
                {
                    "TurbID": turbine_id,
                    "representation": name,
                    "curve_type": "acf",
                    "x": lag,
                    "y": float(acf[lag]),
                    "x_unit": "10min_steps",
                    "y_unit": "autocorrelation",
                }
            )
    return pd.DataFrame(rows), {
        "selected_turbine_id": turbine_id,
        "selection_rule": "maximum train-only Patv_raw first-difference standard deviation",
    }


def _location_coordinates(config: STMGPromptConfig, turbine_ids: list[int], smoke: bool) -> np.ndarray:
    if smoke:
        index = np.arange(len(turbine_ids), dtype=np.float64)
        return np.column_stack([index, np.sin(index / 2.0)])
    location = pd.read_csv(config.resolve_path(config.location_path))
    lower = {name.lower(): name for name in location.columns}
    id_col = lower.get("turbid", lower.get("turbine_id"))
    if id_col is None:
        raise KeyError("Location table must contain TurbID.")
    location[id_col] = pd.to_numeric(location[id_col], errors="raise").astype(int)
    location = location.set_index(id_col).reindex(turbine_ids)
    coordinate_cols = [lower.get(name.lower()) for name in config.graph_coordinate_cols]
    coordinate_cols = [name for name in coordinate_cols if name is not None]
    if len(coordinate_cols) < 2:
        raise KeyError(f"Location table lacks configured coordinate columns {config.graph_coordinate_cols}.")
    coordinates = location[coordinate_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("Location coordinates contain missing or non-finite values.")
    return coordinates


def _cosine_similarity_matrix(values: np.ndarray) -> np.ndarray:
    centered = _column_fill_and_center(values)
    norm = np.linalg.norm(centered, axis=0)
    normalized = centered / np.maximum(norm, EPS)
    return np.clip(normalized.T @ normalized, -1.0, 1.0)


def _correlation_matrix(values: np.ndarray) -> np.ndarray:
    centered = _column_fill_and_center(values)
    std = centered.std(axis=0)
    normalized = centered / np.maximum(std, EPS)
    correlation = normalized.T @ normalized / max(centered.shape[0], 1)
    return np.clip(correlation, -1.0, 1.0)


def _support_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a, b = left > 1e-8, right > 1e-8
    np.fill_diagonal(a, False)
    np.fill_diagonal(b, False)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / max(int(union), 1))


def build_graph_evidence(
    prepared: PreparedArrays,
    train_end: int,
    config: STMGPromptConfig,
    *,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feature_index = {name: index for index, name in enumerate(prepared.feature_cols)}
    target = np.where(prepared.mask[:train_end].astype(bool), prepared.y_raw[:train_end], np.nan)
    wspd = prepared.x_raw[:train_end, :, feature_index["Wspd"]]
    coordinates = _location_coordinates(config, prepared.turbine_ids, smoke)
    location = pd.DataFrame(
        {
            "TurbID": prepared.turbine_ids,
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
        }
    )
    setattr(config, "_graph_num_nodes", len(prepared.turbine_ids))
    setattr(config, "_graph_turbine_ids", list(prepared.turbine_ids))
    distance = build_distance_prior_graph(location, config)
    macro = build_macro_trend_graph(target, prepared.mask[:train_end].astype(bool), location, config)
    micro = build_micro_local_graph(wspd, np.isfinite(wspd), location, config)
    graphs = {"Micro": micro, "Macro": macro, "Distance": distance}
    delta_wspd = np.diff(wspd, axis=0)
    ma36_patv = causal_rolling_mean(target, 36)
    delta_similarity = _cosine_similarity_matrix(delta_wspd)
    trend_similarity = _correlation_matrix(ma36_patv)
    semantic_matrices = {
        "delta_wspd_cosine": delta_similarity,
        "ma36_patv_correlation": trend_similarity,
    }
    global_jaccard = _support_jaccard(micro, macro)

    difference = coordinates[:, None, :] - coordinates[None, :, :]
    edge_distance = np.sqrt(np.sum(difference**2, axis=-1))
    neighbor_rows: list[dict[str, Any]] = []
    for node_index, turbine_id in enumerate(prepared.turbine_ids):
        micro_neighbors = set(np.flatnonzero(micro[node_index] > 1e-8).tolist()) - {node_index}
        macro_neighbors = set(np.flatnonzero(macro[node_index] > 1e-8).tolist()) - {node_index}
        union = micro_neighbors | macro_neighbors
        intersection = micro_neighbors & macro_neighbors
        row: dict[str, Any] = {
            "TurbID": int(turbine_id),
            "node_index": node_index,
            "micro_degree": len(micro_neighbors),
            "macro_degree": len(macro_neighbors),
            "neighbor_jaccard_micro_macro": len(intersection) / max(len(union), 1),
            "global_support_jaccard_micro_macro": global_jaccard,
            "fit_split": "train",
        }
        for graph_name, adjacency in graphs.items():
            neighbors = np.flatnonzero(adjacency[node_index] > 1e-8)
            neighbors = neighbors[neighbors != node_index]
            distances = edge_distance[node_index, neighbors]
            prefix = graph_name.lower()
            row[f"{prefix}_edge_length_mean"] = float(np.mean(distances)) if distances.size else np.nan
            row[f"{prefix}_edge_length_q90"] = float(np.quantile(distances, 0.9)) if distances.size else np.nan
        neighbor_rows.append(row)

    edge_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for graph_name, adjacency in graphs.items():
        support = adjacency > 1e-8
        np.fill_diagonal(support, False)
        sources, targets = np.where(support)
        weights = adjacency[sources, targets].astype(np.float64)
        for source, target_index, weight in zip(sources, targets, weights):
            edge_rows.append(
                {
                    "graph": graph_name,
                    "source_TurbID": int(prepared.turbine_ids[source]),
                    "target_TurbID": int(prepared.turbine_ids[target_index]),
                    "source_node_index": int(source),
                    "target_node_index": int(target_index),
                    "weight": float(weight),
                    "edge_length": float(edge_distance[source, target_index]),
                    "delta_wspd_cosine": float(delta_similarity[source, target_index]),
                    "ma36_patv_correlation": float(trend_similarity[source, target_index]),
                    "fit_split": "train",
                }
            )
        for semantic_name, matrix in semantic_matrices.items():
            values = matrix[sources, targets]
            weighted = float(np.average(values, weights=np.maximum(weights, EPS))) if values.size else np.nan
            assignment = "reference"
            if graph_name == "Micro":
                assignment = "matched" if semantic_name == "delta_wspd_cosine" else "crossed"
            elif graph_name == "Macro":
                assignment = "matched" if semantic_name == "ma36_patv_correlation" else "crossed"
            fit_rows.append(
                {
                    "graph": graph_name,
                    "semantic": semantic_name,
                    "assignment": assignment,
                    "edge_count": int(values.size),
                    "mean_semantic_fit": float(np.mean(values)) if values.size else np.nan,
                    "weighted_mean_semantic_fit": weighted,
                    "mean_edge_length": float(np.mean(edge_distance[sources, targets])) if values.size else np.nan,
                    "global_support_jaccard_micro_macro": global_jaccard,
                    "fit_split": "train",
                    "interpretation_scope": "statistical_association_not_physical_causality",
                }
            )
    fit_frame = pd.DataFrame(fit_rows)
    distance_reference = (
        fit_frame[fit_frame["graph"] == "Distance"]
        .set_index("semantic")["weighted_mean_semantic_fit"]
        .to_dict()
    )
    fit_frame["distance_reference_fit"] = fit_frame["semantic"].map(distance_reference)
    fit_frame["semantic_fit_lift_vs_distance"] = (
        fit_frame["weighted_mean_semantic_fit"] - fit_frame["distance_reference_fit"]
    )
    fit_frame["semantic_fit_score"] = fit_frame["semantic_fit_lift_vs_distance"]
    graph_audit = {
        "official_builders_reused": [
            "build_micro_local_graph",
            "build_macro_trend_graph",
            "build_distance_prior_graph",
        ],
        "global_support_jaccard_micro_macro": global_jaccard,
        "macro_row_sum_range": [float(macro.sum(axis=1).min()), float(macro.sum(axis=1).max())],
        "micro_row_sum_range": [float(micro.sum(axis=1).min()), float(micro.sum(axis=1).max())],
        "distance_row_sum_range": [float(distance.sum(axis=1).min()), float(distance.sum(axis=1).max())],
        "semantic_fit_score_definition": "edge-weighted semantic fit minus the same semantic measured on the official distance graph",
    }
    return fit_frame, pd.DataFrame(neighbor_rows), pd.DataFrame(edge_rows), graph_audit


def _window_targets(
    prepared: PreparedArrays, starts: Iterable[int], horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    starts = list(starts)
    target = np.stack([prepared.y_raw[index : index + horizon] for index in starts], axis=0)
    mask = np.stack([prepared.mask[index : index + horizon] for index in starts], axis=0).astype(bool)
    return target.astype(np.float32), mask


def _moving_average_predictions(
    prepared: PreparedArrays,
    starts: Iterable[int],
    config: STMGPromptConfig,
) -> np.ndarray:
    feature_index = prepared.feature_cols.index(config.input_patv_col)
    patv_clean = prepared.x_raw[:, :, feature_index]
    predictions = []
    for prediction_start in starts:
        history = patv_clean[prediction_start - config.lookback : prediction_start]
        if history.shape[0] != config.lookback or not np.isfinite(history).all():
            raise ValueError("MovingAverage validation reference requires a finite full lookback window.")
        mean = history.mean(axis=0)
        predictions.append(np.repeat(mean[None, :], config.max_pred_len, axis=0))
    result = np.stack(predictions, axis=0).astype(np.float32)
    return np.clip(result, config.physical_power_min_kw, config.physical_power_max_kw)


def _load_formal_test_predictions(
    path: Path,
    prepared: PreparedArrays,
    test_starts: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        prediction = archive["y_pred_eval_snh"].astype(np.float32)
        target = archive["y_true_snh"].astype(np.float32)
        mask = archive["valid_target_mask_snh"].astype(bool)
        starts = archive["prediction_start_index"].astype(np.int64)
        node_ids = archive["node_id"].astype(np.int64)
        metadata = json.loads(str(archive["export_metadata_json"].item()))
    expected_starts = np.asarray(test_starts, dtype=np.int64)
    expected_nodes = np.asarray(prepared.turbine_ids, dtype=np.int64)
    if not np.array_equal(starts, expected_starts):
        raise ValueError("Formal prediction_start_index does not match the official test windows.")
    if not np.array_equal(node_ids, expected_nodes):
        raise ValueError("Formal prediction node order does not match the official data order.")
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("Formal prediction, target, and mask shapes differ.")
    return (
        np.transpose(prediction, (0, 2, 1)),
        np.transpose(target, (0, 2, 1)),
        np.transpose(mask, (0, 2, 1)),
        metadata,
    )


def _difficulty_summary_rows(
    split: str,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    num_nodes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in evaluate_prefix_horizons(prediction, target, mask, EVAL_HORIZONS, num_nodes=num_nodes):
        rows.append(
            {
                "split": split,
                "aggregation_level": "prefix_horizon",
                "horizon": int(result["horizon"]),
                "future_step": np.nan,
                "Score": result["Score"],
                "MAE": result["MAE"],
                "RMSE": result["RMSE"],
                "valid_target_count": result["valid_target_count"],
                "valid_target_ratio": result["valid_target_ratio"],
            }
        )
    for result in evaluate_per_step(prediction, target, mask, num_nodes=num_nodes):
        rows.append(
            {
                "split": split,
                "aggregation_level": "future_step",
                "horizon": np.nan,
                "future_step": int(result["step"]),
                "Score": result["Score"],
                "MAE": result["MAE"],
                "RMSE": result["RMSE"],
                "valid_target_count": result["valid_target_count"],
                "valid_target_ratio": result["valid_target_ratio"],
            }
        )
    return rows


def build_difficulty_evidence(
    prepared: PreparedArrays,
    windows: dict[str, list[int]],
    config: STMGPromptConfig,
    *,
    smoke: bool,
    prediction_export: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    val_prediction = _moving_average_predictions(prepared, windows["val"], config)
    val_target, val_mask = _window_targets(prepared, windows["val"], config.max_pred_len)
    val_mask &= np.isfinite(val_target) & np.isfinite(val_prediction)
    val_target = np.where(val_mask, val_target, 0.0).astype(np.float32)
    val_prediction = np.where(val_mask, val_prediction, 0.0).astype(np.float32)
    formal_metadata: dict[str, Any]
    if smoke:
        test_prediction = _moving_average_predictions(prepared, windows["test"], config)
        test_target, test_mask = _window_targets(prepared, windows["test"], config.max_pred_len)
        formal_metadata = {
            "display_name": "MovingAverage smoke reference",
            "prediction_source": "generated_from_synthetic_history",
            "validation_pass": True,
        }
    else:
        test_prediction, test_target, test_mask, formal_metadata = _load_formal_test_predictions(
            prediction_export, prepared, windows["test"]
        )
    test_mask &= np.isfinite(test_target) & np.isfinite(test_prediction)
    test_target = np.where(test_mask, test_target, 0.0).astype(np.float32)
    test_prediction = np.where(test_mask, test_prediction, 0.0).astype(np.float32)

    horizon_rows = []
    horizon_rows.extend(
        _difficulty_summary_rows("val", val_prediction, val_target, val_mask, len(prepared.turbine_ids))
    )
    horizon_rows.extend(
        _difficulty_summary_rows("test", test_prediction, test_target, test_mask, len(prepared.turbine_ids))
    )
    formal_metric_comparison: list[dict[str, Any]] = []
    if not smoke:
        for horizon in EVAL_HORIZONS:
            computed = next(
                row
                for row in horizon_rows
                if row["split"] == "test"
                and row["aggregation_level"] == "prefix_horizon"
                and row["horizon"] == horizon
            )
            official = formal_metadata[f"official_metrics_h{horizon}"]
            for metric in ("Score", "MAE", "RMSE"):
                formal_metric_comparison.append(
                    {
                        "horizon": horizon,
                        "metric": metric,
                        "computed": float(computed[metric]),
                        "official": float(official[metric]),
                        "absolute_difference": abs(float(computed[metric]) - float(official[metric])),
                    }
                )

    val_h10_score = np.full(len(prepared.turbine_ids), np.nan, dtype=np.float64)
    for node_index in range(len(prepared.turbine_ids)):
        metrics = regression_metrics_kw(
            val_prediction[:, :10, node_index : node_index + 1],
            val_target[:, :10, node_index : node_index + 1],
            val_mask[:, :10, node_index : node_index + 1],
            num_nodes=1,
        )
        val_h10_score[node_index] = float(metrics["Score"])
    hard_count = max(1, int(math.ceil(0.10 * len(prepared.turbine_ids))))
    valid_order = np.argsort(np.nan_to_num(val_h10_score, nan=-np.inf))[::-1]
    hard_indices = set(int(value) for value in valid_order[:hard_count])
    threshold = float(np.nanmin(val_h10_score[list(hard_indices)]))
    ranks = np.empty(len(prepared.turbine_ids), dtype=np.int64)
    ranks[valid_order] = np.arange(1, len(prepared.turbine_ids) + 1)

    turbine_rows: list[dict[str, Any]] = []
    for split, prediction, target, mask in (
        ("val", val_prediction, val_target, val_mask),
        ("test", test_prediction, test_target, test_mask),
    ):
        for node_index, turbine_id in enumerate(prepared.turbine_ids):
            for horizon in EVAL_HORIZONS:
                metrics = regression_metrics_kw(
                    prediction[:, :horizon, node_index : node_index + 1],
                    target[:, :horizon, node_index : node_index + 1],
                    mask[:, :horizon, node_index : node_index + 1],
                    num_nodes=1,
                )
                turbine_rows.append(
                    {
                        "split": split,
                        "TurbID": int(turbine_id),
                        "node_index": node_index,
                        "horizon": horizon,
                        "Score": metrics["Score"],
                        "MAE": metrics["MAE"],
                        "RMSE": metrics["RMSE"],
                        "valid_target_count": metrics["valid_target_count"],
                        "reference_val_score_h10": float(val_h10_score[node_index]),
                        "reference_val_rank_desc": int(ranks[node_index]),
                        "hard_turbine_threshold_val_score_h10": threshold,
                        "is_top10pct_hard_turbine": node_index in hard_indices,
                        "hard_group_rule": f"top_{hard_count}_by_validation_H10_Score",
                    }
                )
    audit = {
        "baseline": "MovingAverage",
        "formal_test_prediction_path": None if smoke else str(prediction_export.resolve()),
        "formal_export_metadata": formal_metadata,
        "validation_prediction_rule": "mean of the preceding 144 Patv_clean_for_input values, repeated for H10 and physically clipped",
        "hard_turbine_reference_split": "validation",
        "hard_turbine_metric": "validation H10 Score",
        "hard_turbine_count": hard_count,
        "hard_turbine_threshold": threshold,
        "hard_turbine_ids": [int(prepared.turbine_ids[index]) for index in sorted(hard_indices)],
        "test_targets_used_for_threshold": False,
        "formal_metric_comparison": formal_metric_comparison,
        "formal_metric_max_absolute_difference": (
            max(row["absolute_difference"] for row in formal_metric_comparison)
            if formal_metric_comparison
            else None
        ),
        "formal_metric_match_tolerance": 1e-5,
        "formal_metric_match_pass": (
            all(row["absolute_difference"] <= 1e-5 for row in formal_metric_comparison)
            if formal_metric_comparison
            else True
        ),
    }
    return pd.DataFrame(horizon_rows), pd.DataFrame(turbine_rows), audit


def _evidence_summary(
    temporal_statistics: pd.DataFrame,
    graph_semantic_fit: pd.DataFrame,
    alignment_metrics: pd.DataFrame,
    turbine_difficulty: pd.DataFrame,
) -> dict[str, Any]:
    temporal = (
        temporal_statistics.groupby(["component"], dropna=False)["high_frequency_energy_ratio"].mean().to_dict()
    )
    graph = graph_semantic_fit[graph_semantic_fit["assignment"].isin(["matched", "crossed"])]
    graph_summary = graph.groupby("assignment")["semantic_fit_score"].mean().to_dict()
    alignment = alignment_metrics.groupby("alignment_mode")["reconstruction_rmse_kw"].mean().to_dict()
    test_h10 = turbine_difficulty[
        (turbine_difficulty["split"] == "test") & (turbine_difficulty["horizon"] == 10)
    ]
    hard_mean = test_h10[test_h10["is_top10pct_hard_turbine"]]["Score"].mean()
    other_mean = test_h10[~test_h10["is_top10pct_hard_turbine"]]["Score"].mean()
    return {
        "fixed_dual_motivation": {
            "mean_high_frequency_energy_ratio_by_component": temporal,
            "scope": "descriptive train-only evidence for distinct local and trend scales",
        },
        "coaxial_alignment_motivation": {
            "mean_event_reconstruction_rmse_kw_by_alignment_mode": alignment,
            "scope": "descriptive train-only event alignment evidence",
        },
        "dual_graph_motivation": {
            "mean_semantic_fit_lift_vs_distance_by_assignment": graph_summary,
            "scope": "statistical semantic matching only; no physical causality claim",
        },
        "dynamic_difficulty_motivation": {
            "test_h10_score_mean_hard_group": float(hard_mean),
            "test_h10_score_mean_other_group": float(other_mean),
            "scope": "validation-fixed grouping applied to formal test predictions",
        },
    }


def build_evidence(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    smoke: bool = False,
    make_plots: bool = True,
    prediction_export: Path = DEFAULT_BASELINE_EXPORT,
) -> dict[str, Any]:
    config = canonical_config(STMGPromptConfig())
    config.smoke = bool(smoke)
    config.smoke_use_synthetic = bool(smoke)
    if smoke:
        # Keep the formal 144-step history intact.  Each chronological 10%
        # validation/test split therefore needs at least 154 points.
        config.smoke_num_time_steps = max(config.smoke_num_time_steps, 1_800)
        config.graph_coordinate_cols = ["x", "y"]
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = _load_arrays(config, smoke)
    splits, windows = _split_and_windows(prepared, config)
    train_start, train_end = splits["train"]
    if train_start != 0:
        raise AssertionError("The official chronological training split must start at index zero.")
    if list(config.coarse_windows) != list(CAUSAL_WINDOWS):
        raise AssertionError(f"Formal coarse windows changed: {config.coarse_windows}")

    overview = build_data_overview(prepared, splits, windows, config)
    turbine_summary = build_turbine_summary(prepared, train_end, config)
    temporal_statistics, associations, _ = build_temporal_evidence(prepared, train_end, config)
    alignment_metrics, typical_curve, alignment_audit = build_alignment_evidence(
        prepared, train_end, config, smoke=smoke
    )
    spectrum_acf, spectrum_audit = build_spectrum_acf_source(prepared, train_end)
    graph_fit, graph_overlap, graph_edges, graph_audit = build_graph_evidence(
        prepared, train_end, config, smoke=smoke
    )
    horizon_difficulty, turbine_difficulty, difficulty_audit = build_difficulty_evidence(
        prepared,
        windows,
        config,
        smoke=smoke,
        prediction_export=Path(prediction_export),
    )

    outputs = {
        "data_overview.csv": overview,
        "turbine_summary.csv": turbine_summary,
        "temporal_statistics.csv": temporal_statistics,
        "horizon_feature_association.csv": associations,
        "alignment_event_metrics.csv": alignment_metrics,
        "alignment_typical_event_curve.csv": typical_curve,
        "spectrum_acf_source.csv": spectrum_acf,
        "graph_semantic_fit.csv": graph_fit,
        "graph_neighbor_overlap.csv": graph_overlap,
        "graph_edge_lengths.csv": graph_edges,
        "baseline_horizon_difficulty.csv": horizon_difficulty,
        "baseline_turbine_difficulty.csv": turbine_difficulty,
    }
    for filename, frame in outputs.items():
        _write_csv(frame, output_dir / filename)

    metadata = {
        "analysis_id": "data_evidence_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if smoke else "full",
        "training_performed": False,
        "data_loader_reuse": {
            "model_input_loader": "st_mgprompt.data.load_model_input",
            "target_loader": "st_mgprompt.data.load_eval_target",
            "alignment_and_pivot": "st_mgprompt.data._pivot_arrays",
            "split_function": "st_mgprompt.data.compute_split_indices",
            "window_function": "st_mgprompt.data.build_window_start_indices",
        },
        "dataset": {
            "time_points": len(prepared.timestamps),
            "num_nodes": len(prepared.turbine_ids),
            "feature_count": len(prepared.feature_cols),
            "feature_names": prepared.feature_cols,
            "split_indices": splits,
            "window_counts": {key: len(value) for key, value in windows.items()},
        },
        "train_only_protocol": {
            "statistics_fit_split": "train",
            "statistics_source_index_range": [train_start, train_end - 1],
            "train_end_index_exclusive": train_end,
            "validation_or_test_targets_used_for_training_statistics": False,
            "graph_statistics_fit_split": "train",
            "event_selection_fit_split": "train",
            "future_feature_access": False,
            "association_feature_rule": "feature(t) uses values <=t; target uses t+1..t+H",
        },
        "temporal_protocol": {
            "coarse_windows": list(CAUSAL_WINDOWS),
            "formal_model_coarse_windows": list(config.coarse_windows),
            "rolling_alignment": "coaxial causal trailing mean including current index",
            "high_frequency_cutoff_cycles_per_step": 1.0 / 18.0,
            "acf_decay_rule": "first lag with absolute ACF <= exp(-1), capped at 144",
        },
        "alignment_audit": alignment_audit,
        "spectrum_audit": spectrum_audit,
        "graph_audit": graph_audit,
        "difficulty_audit": difficulty_audit,
        "evidence_summary": _evidence_summary(
            temporal_statistics, graph_fit, alignment_metrics, turbine_difficulty
        ),
        "output_files": sorted(outputs),
        "interpretation_limits": [
            "Graph results quantify statistical matching and do not establish physical causality.",
            "Data evidence motivates later controlled experiments but is not itself an ablation result.",
        ],
    }
    metadata_path = output_dir / "analysis_metadata.json"
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False), encoding="utf-8")

    figure_paths: list[str] = []
    if make_plots:
        try:
            from .plot_data_evidence import plot_all
        except ImportError:
            from plot_data_evidence import plot_all
        figure_paths = [str(path) for path in plot_all(output_dir)]
        metadata["figure_files"] = [str(Path(path).relative_to(output_dir)) for path in figure_paths]
        metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "output_dir": str(output_dir.resolve()),
        "mode": metadata["mode"],
        "csv_count": len(outputs),
        "figure_count": len(figure_paths),
        "metadata_path": str(metadata_path.resolve()),
        "train_end_index_exclusive": train_end,
        "num_nodes": len(prepared.turbine_ids),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only SDWPF data evidence for empirical step 2.")
    parser.add_argument("--smoke", action="store_true", help="Use the repository synthetic smoke data path.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prediction-export", type=Path, default=DEFAULT_BASELINE_EXPORT)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (DEFAULT_OUTPUT_DIR / "smoke" if args.smoke else DEFAULT_OUTPUT_DIR)
    result = build_evidence(
        output_dir=output_dir,
        smoke=args.smoke,
        make_plots=not args.no_plots,
        prediction_export=args.prediction_export,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
