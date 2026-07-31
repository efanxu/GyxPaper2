from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def _hash(items: Iterable[Any]) -> str:
    payload = json.dumps(list(items), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_order_hash(features: Iterable[str]) -> str:
    return _hash(features)


def node_order_hash(node_ids: Iterable[Any]) -> str:
    return _hash(node_ids)


def data_signature(*, dataset_id: str, feature_names: list[str], node_ids: list[Any], timestamps: list[Any], shape: tuple[int, ...], scalers_fit_split: str) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "shape": list(shape),
        "feature_names": list(feature_names),
        "feature_order_hash": feature_order_hash(feature_names),
        "node_count": len(node_ids),
        "node_order_hash": node_order_hash(node_ids),
        "timestamp_count": len(timestamps),
        "first_timestamp": str(timestamps[0]) if timestamps else None,
        "last_timestamp": str(timestamps[-1]) if timestamps else None,
        "normalization_fit_scope": scalers_fit_split,
    }

