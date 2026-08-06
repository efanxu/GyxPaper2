from __future__ import annotations

from typing import Any


def data_signature(
    *,
    dataset_id: str,
    feature_names: list[str],
    node_ids: list[Any],
    timestamps: list[Any],
    shape: tuple[int, ...],
    scalers_fit_split: str,
    input_path: str | None = None,
    target_path: str | None = None,
    split: Any = None,
    lookback: int = 144,
    horizon: int = 10,
    stride: Any = None,
    seed: int = 2026,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "input_path": input_path,
        "target_path": target_path,
        "shape": list(shape),
        "feature_names": list(feature_names),
        "node_count": len(node_ids),
        "node_ids": list(node_ids),
        "timestamp_count": len(timestamps),
        "first_timestamp": str(timestamps[0]) if timestamps else None,
        "last_timestamp": str(timestamps[-1]) if timestamps else None,
        "normalization_fit_scope": scalers_fit_split,
        "split": split,
        "lookback": int(lookback),
        "horizon": int(horizon),
        "stride": stride,
        "seed": int(seed),
    }
