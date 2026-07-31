from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StandardScaler:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, values: np.ndarray, mask: np.ndarray | None = None) -> "StandardScaler":
        data = np.asarray(values, dtype=np.float64)
        if mask is not None:
            data = data[np.asarray(mask).astype(bool)]
        if data.size == 0:
            raise ValueError("Cannot fit scaler with no valid train values.")
        self.mean = np.asarray(np.nanmean(data, axis=0), dtype=np.float64)
        self.std = np.asarray(np.nanstd(data, axis=0), dtype=np.float64)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        self.std = np.where(np.isfinite(self.std) & (self.std > 1e-8), self.std, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler is not fit.")
        result = (np.asarray(values) - self.mean) / self.std
        return np.where(np.isfinite(result), result, 0.0).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler is not fit.")
        return (np.asarray(values) * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist() if self.mean is not None else None, "std": self.std.tolist() if self.std is not None else None}

