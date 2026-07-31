from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import BenchmarkBatch
from .scaling import StandardScaler
from .signatures import data_signature
from .windows import split_indices, window_start_indices


FEATURES = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "T2m", "Sp", "RelH", "Wspd_w", "Wdir_w", "Tp", "Patv_clean_for_input"]


@dataclass
class _Window:
    x: np.ndarray
    target: np.ndarray
    target_raw: np.ndarray
    mask: np.ndarray
    sample_id: str
    end_index: int
    history_end_timestamp: Any
    history_end_time_of_day_id: int
    history_end_day_of_week_id: int


class SDWPFDataProvider:
    """Aligned SDWPF provider; it owns the only benchmark data/window path."""

    def __init__(self, x_raw: np.ndarray, target_raw: np.ndarray, mask: np.ndarray, *, timestamps: list[Any], node_ids: list[Any], feature_names: list[str] = FEATURES, protocol: Any | None = None):
        self.x_raw = np.asarray(x_raw, dtype=np.float32)
        self.target_raw = np.asarray(target_raw, dtype=np.float32)
        self.mask = np.asarray(mask).astype(bool)
        self.timestamps = list(timestamps)
        self.node_ids = list(node_ids)
        self.feature_names = list(feature_names)
        self.protocol = protocol
        if self.x_raw.ndim != 3 or self.target_raw.shape != self.mask.shape or self.target_raw.shape != self.x_raw.shape[:2]:
            raise ValueError(f"Aligned arrays must be x=(T,N,C), target/mask=(T,N); got {self.x_raw.shape}, {self.target_raw.shape}, {self.mask.shape}")
        if len(self.feature_names) != self.x_raw.shape[-1] or len(self.node_ids) != self.x_raw.shape[1]:
            raise ValueError("Feature or node order does not match aligned arrays.")
        self.scalers = self._fit_train_only_scalers()
        self.x_model = self.scalers["input"].transform(self.x_raw)
        self.target_model = self.scalers["target"].transform(self.target_raw)
        self.split_indices = split_indices(len(self.timestamps), list(self._p("split_ratio", [0.8, 0.1, 0.1])))
        self.starts = {
            split: window_start_indices(s, e, int(self._p("lookback", 144)), int(self._p("max_pred_len", 10)), int(self._p(f"{split}_sample_stride", {"train": 6, "val": 3, "test": 1}[split])))
            for split, (s, e) in self.split_indices.items()
        }

    @classmethod
    def from_arrays(cls, x_raw: np.ndarray, target_raw: np.ndarray, mask: np.ndarray, *, node_ids: list[Any] | None = None, timestamps: list[Any] | None = None, protocol: Any | None = None) -> "SDWPFDataProvider":
        return cls(x_raw, target_raw, mask, timestamps=list(timestamps or range(len(x_raw))), node_ids=list(node_ids or range(x_raw.shape[1])), protocol=protocol)

    @classmethod
    def from_files(cls, input_path: str | Path, target_path: str | Path, *, protocol: Any | None = None) -> "SDWPFDataProvider":
        import pandas as pd
        input_path, target_path = Path(input_path), Path(target_path)
        features = list((protocol or {}).get("ordered_input_features", FEATURES))
        target_col = (protocol or {}).get("target_column", "Patv_raw")
        mask_col = (protocol or {}).get("mask_column", "valid_target_mask")
        timestamp_col = (protocol or {}).get("timestamp_column", "Tmstamp")
        node_col = (protocol or {}).get("node_id_column", "TurbID")
        model_df = pd.read_parquet(input_path, columns=[timestamp_col, node_col, *features])
        target_df = pd.read_parquet(target_path, columns=[timestamp_col, node_col, target_col, mask_col])
        keys = [timestamp_col, node_col]
        if model_df.duplicated(keys).any() or target_df.duplicated(keys).any():
            raise ValueError("Duplicate (timestamp,node) rows are not allowed.")
        if not model_df[keys].reset_index(drop=True).equals(target_df[keys].reset_index(drop=True)):
            raise ValueError("Model input and target are not aligned in row order.")
        timestamps = sorted(model_df[timestamp_col].unique().tolist())
        node_ids = sorted(model_df[node_col].astype(int).unique().tolist())
        def pivot(df, col):
            return df.pivot(index=timestamp_col, columns=node_col, values=col).reindex(index=timestamps, columns=node_ids).to_numpy()
        x = np.stack([pivot(model_df, col) for col in features], axis=-1).astype(np.float32)
        target = pivot(target_df, target_col).astype(np.float32)
        mask = np.isfinite(target) & pivot(target_df, mask_col).astype(bool)
        return cls(x, target, mask, timestamps=timestamps, node_ids=node_ids, feature_names=features, protocol=protocol)

    def _p(self, key: str, default: Any) -> Any:
        return self.protocol.get(key, default) if self.protocol is not None else default

    def _fit_train_only_scalers(self) -> dict[str, StandardScaler]:
        train_end = int(len(self.timestamps) * float(self._p("split_ratio", [0.8, 0.1, 0.1])[0]))
        input_scaler = StandardScaler().fit(self.x_raw[:train_end].reshape(-1, self.x_raw.shape[-1]))
        mask = self.mask[:train_end]
        target_scaler = StandardScaler().fit(self.target_raw[:train_end], mask=mask)
        return {"input": input_scaler, "target": target_scaler}

    def windows(self, split: str) -> list[_Window]:
        result = []
        lookback, horizon = int(self._p("lookback", 144)), int(self._p("max_pred_len", 10))
        for end_index in self.starts[split]:
            raw_mask = self.mask[end_index:end_index + horizon]
            import pandas as pd

            history_end_timestamp = self.timestamps[end_index - 1]
            anchor = pd.Timestamp(history_end_timestamp)
            time_of_day_id = int(anchor.hour * 6 + anchor.minute // 10)
            day_of_week_id = int(anchor.dayofweek)
            result.append(_Window(
                self.x_model[end_index - lookback:end_index],
                self.target_model[end_index:end_index + horizon].T,
                self.target_raw[end_index:end_index + horizon].T,
                raw_mask.T,
                f"SDWPF:{split}:{end_index}",
                end_index,
                history_end_timestamp,
                time_of_day_id,
                day_of_week_id,
            ))
        return result

    def batches(self, split: str, batch_size: int) -> list[BenchmarkBatch]:
        windows = self.windows(split)
        batches = []
        for offset in range(0, len(windows), batch_size):
            chunk = windows[offset:offset + batch_size]
            import torch
            batch = BenchmarkBatch(
                x=torch.from_numpy(np.stack([w.x for w in chunk])),
                target=torch.from_numpy(np.stack([w.target for w in chunk])),
                mask=torch.from_numpy(np.stack([w.mask for w in chunk])),
                target_raw_or_inverse_transform=torch.from_numpy(np.stack([w.target_raw for w in chunk])),
                sample_ids=[w.sample_id for w in chunk],
                window_end_indices=[w.end_index for w in chunk],
                node_ids=list(self.node_ids), split=split,
                metadata={
                    "contains_future_target": False,
                    "target_model_space": "standard_zscore",
                    "target_raw_space": "kW",
                    "history_end_timestamp": [
                        w.history_end_timestamp for w in chunk
                    ],
                    "history_end_time_of_day_id": [
                        w.history_end_time_of_day_id for w in chunk
                    ],
                    "history_end_day_of_week_id": [
                        w.history_end_day_of_week_id for w in chunk
                    ],
                    "timezone_policy": "SOURCE_NAIVE_UNCHANGED",
                },
            )
            batch.validate(expected_nodes=len(self.node_ids))
            batches.append(batch)
        return batches

    def signature(self) -> dict[str, Any]:
        return data_signature(dataset_id=str(self._p("dataset_id", "SDWPF")), feature_names=self.feature_names, node_ids=self.node_ids, timestamps=self.timestamps, shape=self.x_raw.shape, scalers_fit_split="train_only")
