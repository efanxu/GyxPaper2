from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_16_FEATURES, STMGPromptConfig
from .volatility_patching import compute_vadsp_robust_statistics_from_timeline


@dataclass
class StandardScaler:
    mean: np.ndarray | float | None = None
    std: np.ndarray | float | None = None

    def fit(self, values: np.ndarray, mask: np.ndarray | None = None) -> "StandardScaler":
        data = values.astype(np.float64)
        if mask is not None:
            data = data[mask.astype(bool)]
        if data.size == 0:
            raise ValueError("Cannot fit scaler with no valid values.")
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        self.std = np.where(np.isfinite(self.std) & (self.std > 1e-8), self.std, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler is not fit.")
        scaled = (values - self.mean) / self.std
        return np.where(np.isfinite(scaled), scaled, 0.0).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler is not fit.")
        return (values * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        def _tolist(v: Any) -> Any:
            return v.tolist() if isinstance(v, np.ndarray) else float(v)

        return {"mean": _tolist(self.mean), "std": _tolist(self.std)}


@dataclass
class PreparedArrays:
    x_raw: np.ndarray
    y_raw: np.ndarray
    mask: np.ndarray
    timestamps: list[str]
    turbine_ids: list[int]
    feature_cols: list[str]


@dataclass
class STMGPromptDataBundle:
    train_loader: object
    val_loader: object
    test_loader: object
    scalers: dict[str, StandardScaler]
    split_indices: dict[str, tuple[int, int]]
    window_indices: dict[str, list[int]]
    feature_cols: list[str]
    turbine_ids: list[int]
    input_dim: int
    num_nodes: int
    metadata: dict[str, Any]
    vadsp_robust_statistics: dict[str, Any] | None = None


def _read_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path, columns=columns)
        except Exception:
            if columns is None:
                raise
            return pd.read_parquet(path)
    if suffix == ".csv":
        if columns is None:
            return pd.read_csv(path)
        selected = set(columns)
        return pd.read_csv(path, usecols=lambda c: c in selected)
    raise ValueError(f"Unsupported file format: {path}")


def load_model_input(config: STMGPromptConfig) -> pd.DataFrame:
    cols = [config.timestamp_col, config.turbine_id_col, *config.feature_cols]
    return _read_table(config.resolve_path(config.model_input_path), columns=cols)


def load_eval_target(config: STMGPromptConfig) -> pd.DataFrame:
    cols = [config.timestamp_col, config.turbine_id_col, config.target_col, config.target_mask_col]
    return _read_table(config.resolve_path(config.eval_target_path), columns=cols)


def validate_alignment(model_df: pd.DataFrame, target_df: pd.DataFrame, config: STMGPromptConfig) -> None:
    key_cols = [config.timestamp_col, config.turbine_id_col]
    missing_model = [c for c in [*key_cols, *config.feature_cols] if c not in model_df.columns]
    missing_target = [c for c in [*key_cols, config.target_col, config.target_mask_col] if c not in target_df.columns]
    if missing_model:
        raise KeyError(f"model input missing columns: {missing_model}")
    if missing_target:
        raise KeyError(f"eval target missing columns: {missing_target}")
    if model_df.duplicated(key_cols).any() or target_df.duplicated(key_cols).any():
        raise ValueError("Duplicate (timestamp, turbine) rows are not allowed.")
    model_index = pd.MultiIndex.from_frame(model_df[key_cols])
    target_index = pd.MultiIndex.from_frame(target_df[key_cols])
    if not model_index.equals(target_index):
        raise ValueError("model_input and eval_target are not aligned by (Tmstamp, TurbID).")
    mask_values = target_df[config.target_mask_col].dropna().unique()
    if not set(pd.Series(mask_values).astype(int).unique()).issubset({0, 1}):
        raise ValueError("valid_target_mask must be bool or 0/1.")


def compute_split_indices(num_time_steps: int, split_ratios: list[float]) -> dict[str, tuple[int, int]]:
    train_ratio, val_ratio, _ = split_ratios
    train_end = int(num_time_steps * train_ratio)
    val_end = train_end + int(num_time_steps * val_ratio)
    return {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, num_time_steps)}


def fit_train_only_scalers(
    train_data: np.ndarray,
    train_target: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, StandardScaler]:
    input_scaler = StandardScaler().fit(train_data.reshape(-1, train_data.shape[-1]))
    valid_target = train_mask.astype(bool) & np.isfinite(train_target)
    target_scaler = StandardScaler().fit(train_target.reshape(-1), mask=valid_target.reshape(-1))
    return {"input": input_scaler, "target": target_scaler}


def build_window_start_indices(
    S: int,
    E: int,
    lookback: int,
    max_pred_len: int,
    sample_stride: int,
) -> list[int]:
    first_t = S + lookback
    last_t_exclusive = E - max_pred_len + 1
    if first_t >= last_t_exclusive:
        return []
    return list(range(first_t, last_t_exclusive, sample_stride))


def _pivot_arrays(model_df: pd.DataFrame, target_df: pd.DataFrame, config: STMGPromptConfig) -> PreparedArrays:
    validate_alignment(model_df, target_df, config)
    key_cols = [config.timestamp_col, config.turbine_id_col]
    merged = model_df.merge(target_df, on=key_cols, how="inner", validate="one_to_one")
    merged[config.timestamp_col] = pd.to_datetime(merged[config.timestamp_col], errors="raise")
    merged[config.turbine_id_col] = pd.to_numeric(merged[config.turbine_id_col], errors="raise").astype(int)
    merged = merged.sort_values([config.timestamp_col, config.turbine_id_col])
    timestamps = list(pd.Index(merged[config.timestamp_col].unique()).sort_values())
    turbine_ids = sorted(merged[config.turbine_id_col].unique().astype(int).tolist())
    if len(turbine_ids) != config.num_nodes and not config.smoke:
        raise ValueError(f"Expected {config.num_nodes} turbines, got {len(turbine_ids)}.")

    feature_arrays = []
    for col in config.feature_cols:
        pivot = merged.pivot(index=config.timestamp_col, columns=config.turbine_id_col, values=col)
        pivot = pivot.reindex(index=timestamps, columns=turbine_ids)
        feature_arrays.append(pivot.to_numpy(dtype=np.float32))
    x_raw = np.stack(feature_arrays, axis=-1)
    y_raw = (
        merged.pivot(index=config.timestamp_col, columns=config.turbine_id_col, values=config.target_col)
        .reindex(index=timestamps, columns=turbine_ids)
        .to_numpy(dtype=np.float32)
    )
    mask = (
        merged.pivot(index=config.timestamp_col, columns=config.turbine_id_col, values=config.target_mask_col)
        .reindex(index=timestamps, columns=turbine_ids)
        .fillna(False)
        .to_numpy()
        .astype(bool)
    )
    mask &= np.isfinite(y_raw)
    return PreparedArrays(
        x_raw=x_raw,
        y_raw=y_raw,
        mask=mask.astype(np.float32),
        timestamps=[str(ts) for ts in timestamps],
        turbine_ids=turbine_ids,
        feature_cols=list(config.feature_cols),
    )


def _make_synthetic_arrays(config: STMGPromptConfig) -> PreparedArrays:
    T = max(config.smoke_num_time_steps, 3 * (config.lookback + config.max_pred_len + 4))
    N = min(config.num_nodes, 8)
    C = len(config.feature_cols)
    rng = np.random.default_rng(config.seed)
    t = np.arange(T, dtype=np.float32)[:, None]
    node = np.linspace(-1.0, 1.0, N, dtype=np.float32)[None, :]
    patv = 850.0 + 180.0 * np.sin(t / 18.0) + 70.0 * node
    patv = np.maximum(0.0, patv + rng.normal(0.0, 20.0, size=patv.shape)).astype(np.float32)
    x = np.zeros((T, N, C), dtype=np.float32)
    for idx, name in enumerate(config.feature_cols):
        if name == config.input_patv_col:
            base = patv
        elif name in {"Wspd", "Wspd_w"}:
            base = 8.0 + 2.0 * np.sin(t / 18.0) + 0.2 * node
        elif name in {"Wdir", "Wdir_w", "Ndir"}:
            base = 180.0 + 30.0 * np.cos(t / 24.0) + 5.0 * node
        else:
            base = (idx + 1) * 0.1 + np.sin(t / (10.0 + idx)) + node * 0.1
        x[:, :, idx] = (base + rng.normal(0.0, 0.05, size=(T, N))).astype(np.float32)
    mask = np.ones_like(patv, dtype=np.float32)
    mask[::19, 0] = 0.0
    timestamps = [f"synthetic_{i:05d}" for i in range(T)]
    return PreparedArrays(x, patv, mask, timestamps, list(range(1, N + 1)), list(config.feature_cols))


class STMGPromptDataset:
    def __init__(
        self,
        x: np.ndarray,
        y_norm: np.ndarray,
        y_raw: np.ndarray,
        mask: np.ndarray,
        timestamps: list[str],
        turbine_ids: list[int],
        starts: list[int],
        lookback: int,
        max_pred_len: int,
    ) -> None:
        self.x = x.astype(np.float32)
        self.y_norm = np.where(mask.astype(bool), y_norm, 0.0).astype(np.float32)
        self.y_raw = np.where(mask.astype(bool), y_raw, 0.0).astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.timestamps = timestamps
        self.turbine_ids = turbine_ids
        self.starts = starts
        self.lookback = lookback
        self.max_pred_len = max_pred_len

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        t = self.starts[idx]
        x_slice = self.x[t - self.lookback : t]
        target_slice = self.y_norm[t : t + self.max_pred_len]
        raw_slice = self.y_raw[t : t + self.max_pred_len]
        mask_slice = self.mask[t : t + self.max_pred_len]
        return {
            "x": torch.from_numpy(x_slice),
            "y": torch.from_numpy(target_slice),
            "y_raw": torch.from_numpy(raw_slice),
            "valid_target_mask": torch.from_numpy(mask_slice),
            "timestamps": self.timestamps[t - self.lookback : t],
            "target_timestamps": self.timestamps[t : t + self.max_pred_len],
            "turbine_ids": self.turbine_ids,
            "prediction_start_index": int(t),
        }


def make_dataloaders(config: STMGPromptConfig) -> dict[str, Any]:
    config.validate()
    if config.smoke and config.smoke_use_synthetic:
        prepared = _make_synthetic_arrays(config)
    else:
        prepared = _pivot_arrays(load_model_input(config), load_eval_target(config), config)

    split_indices = compute_split_indices(len(prepared.timestamps), config.split_ratios)
    train_s, train_e = split_indices["train"]
    scalers = fit_train_only_scalers(
        prepared.x_raw[train_s:train_e],
        prepared.y_raw[train_s:train_e],
        prepared.mask[train_s:train_e],
    )
    x_norm = scalers["input"].transform(prepared.x_raw)
    y_norm = scalers["target"].transform(prepared.y_raw)
    vadsp_robust_statistics = None
    if config.use_vadsp and config.use_train_robust_volatility:
        vadsp_robust_statistics = compute_vadsp_robust_statistics_from_timeline(
            timeline=x_norm[train_s:train_e],
            feature_names=prepared.feature_cols,
            volatility_source_cols=config.volatility_source_cols,
            volatility_mode=config.volatility_mode,
            robust_eps=config.volatility_scale_eps,
            fit_split="train",
            fit_source="unique_train_timeline",
            feature_space="model_input_scaled",
        )
    strides = {
        "train": config.train_sample_stride,
        "val": config.val_sample_stride,
        "test": config.test_sample_stride,
    }
    window_indices = {
        split: build_window_start_indices(S, E, config.lookback, config.max_pred_len, strides[split])
        for split, (S, E) in split_indices.items()
    }
    empty = [split for split, starts in window_indices.items() if not starts]
    if empty:
        raise ValueError(f"No valid split-contained windows for splits: {empty}")

    from torch.utils.data import DataLoader

    datasets = {
        split: STMGPromptDataset(
            x=x_norm,
            y_norm=y_norm,
            y_raw=prepared.y_raw,
            mask=prepared.mask,
            timestamps=prepared.timestamps,
            turbine_ids=prepared.turbine_ids,
            starts=starts,
            lookback=config.lookback,
            max_pred_len=config.max_pred_len,
        )
        for split, starts in window_indices.items()
    }
    loaders = {}
    effective_batch_sizes = {
        "train": config.train_batch_size or config.batch_size,
        "val": config.val_batch_size or config.eval_batch_size,
        "test": config.test_batch_size or config.eval_batch_size,
    }
    for split, dataset in datasets.items():
        batch_size = effective_batch_sizes[split]
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
        }
        if config.num_workers > 0:
            loader_kwargs["persistent_workers"] = config.persistent_workers
        loaders[split] = DataLoader(dataset, **loader_kwargs)
    valid_ratios = {
        split: float(prepared.mask[S:E].mean()) if E > S else 0.0
        for split, (S, E) in split_indices.items()
    }
    window_expanded_counts: dict[str, dict[str, int | float]] = {}
    for split, starts in window_indices.items():
        valid_count = 0
        total_count = 0
        for t in starts:
            window_mask = prepared.mask[t : t + config.max_pred_len]
            valid_count += int(window_mask.astype(bool).sum())
            total_count += int(window_mask.size)
        window_expanded_counts[split] = {
            "valid_target_count": valid_count,
            "total_target_count": total_count,
            "window_expanded_valid_ratio": float(valid_count / total_count) if total_count else 0.0,
        }
    metadata = {
        "input_feature_set": "base_official_scada_era5",
        "feature_names": prepared.feature_cols,
        "num_features": len(prepared.feature_cols),
        "target_col": config.target_col,
        "target_mask_col": config.target_mask_col,
        "target_mask_used_as_model_input": False,
        "final_prediction_target_is_power": True,
        "split_indices": split_indices,
        "window_counts": {k: len(v) for k, v in window_indices.items()},
        "effective_batch_sizes": effective_batch_sizes,
        "valid_target_ratio": valid_ratios,
        "raw_split_valid_ratio": valid_ratios,
        "raw_split_valid_count": {
            split: int(prepared.mask[S:E].astype(bool).sum())
            for split, (S, E) in split_indices.items()
        },
        "raw_split_total_count": {
            split: int(prepared.mask[S:E].size)
            for split, (S, E) in split_indices.items()
        },
        "window_expanded_valid_ratio": {
            split: float(item["window_expanded_valid_ratio"])
            for split, item in window_expanded_counts.items()
        },
        "window_expanded_valid_count": {
            split: int(item["valid_target_count"])
            for split, item in window_expanded_counts.items()
        },
        "window_expanded_total_count": {
            split: int(item["total_target_count"])
            for split, item in window_expanded_counts.items()
        },
        "input_scaler_fit_split": "train_only",
        "target_scaler_fit_split": "train_only",
        "target_scaler_fit_mask": config.target_mask_col,
        "window_rule": "[t-lookback,t)->[t,t+max_pred_len)",
        "sample_stride_is_prediction_start_stride": True,
        "uses_on_the_fly_features": bool(config.use_vadsp),
        "on_the_fly_feature_names": (
            [
                "volatility",
                "rolling_std",
                "dynamic_patch_gate",
                "x_fine",
                "x_coarse",
            ]
            if config.use_vadsp
            else []
        ),
        "on_the_fly_features_history_only": True,
    }
    if vadsp_robust_statistics is not None:
        metadata.update(
            {
                "vadsp_robust_statistics_enabled": True,
                "vadsp_statistics_fit_split": vadsp_robust_statistics["fit_split"],
                "vadsp_statistics_source": vadsp_robust_statistics["fit_source"],
                "vadsp_statistics_exact": vadsp_robust_statistics["exact"],
                "vadsp_statistics_device": vadsp_robust_statistics["device"],
                "vadsp_statistics_mode": vadsp_robust_statistics["mode"],
                "vadsp_statistics_source_cols": list(vadsp_robust_statistics["source_cols"]),
                "vadsp_statistics_num_time_points": int(vadsp_robust_statistics["num_time_points"]),
                "vadsp_statistics_num_deltas": int(vadsp_robust_statistics["num_time_deltas"]),
                "vadsp_statistics_shape": dict(vadsp_robust_statistics["shape"]),
                "vadsp_statistics_nonfinite_filtered": int(vadsp_robust_statistics["nonfinite_filtered"]),
                "vadsp_statistics_degenerate_iqr_count": int(vadsp_robust_statistics["degenerate_iqr_count"]),
                "vadsp_statistics_depends_on_window_stride": False,
                "vadsp_statistics_feature_space": vadsp_robust_statistics["feature_space"],
            }
        )
    else:
        metadata["vadsp_robust_statistics_enabled"] = False
    return {
        "train": loaders["train"],
        "val": loaders["val"],
        "test": loaders["test"],
        "bundle": STMGPromptDataBundle(
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            test_loader=loaders["test"],
            scalers=scalers,
            split_indices=split_indices,
            window_indices=window_indices,
            feature_cols=prepared.feature_cols,
            turbine_ids=prepared.turbine_ids,
            input_dim=len(prepared.feature_cols),
            num_nodes=len(prepared.turbine_ids),
            metadata=metadata,
            vadsp_robust_statistics=vadsp_robust_statistics,
        ),
    }
