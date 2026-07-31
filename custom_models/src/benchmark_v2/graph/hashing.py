from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROUND_DECIMALS = 12


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_node_ids(node_ids: Iterable[Any]) -> tuple[Any, ...]:
    values = tuple(node_ids)
    if not values:
        raise ValueError("Node order must not be empty.")
    if any(isinstance(value, bool) for value in values):
        raise TypeError("Boolean node IDs are not permitted.")
    kinds = {type(value) for value in values}
    if kinds == {int}:
        return values
    if kinds == {str}:
        return values
    raise TypeError(
        "Node IDs must use one canonical JSON type: all integer or all string."
    )


def node_order_hash(node_ids: Iterable[Any]) -> str:
    return stable_hash(list(canonical_node_ids(node_ids)))


def canonical_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Graph matrix must be rank two, got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Graph matrix contains NaN or Inf.")
    matrix = np.round(matrix, decimals=ROUND_DECIMALS)
    matrix[matrix == 0.0] = 0.0
    return np.ascontiguousarray(matrix, dtype="<f8")


def matrix_hash(value: Any) -> str:
    matrix = canonical_matrix(value)
    header = canonical_json(
        {
            "dtype": "little-endian-float64",
            "shape": list(matrix.shape),
            "round_decimals": ROUND_DECIMALS,
        }
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\n")
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def graph_bundle_hash(
    *,
    graph_id: str,
    node_count: int,
    node_order_digest: str,
    matrix_hashes: dict[str, str],
) -> str:
    return stable_hash(
        {
            "graph_id": graph_id,
            "node_count": int(node_count),
            "node_order_hash": node_order_digest,
            "matrix_hashes": dict(sorted(matrix_hashes.items())),
        }
    )


def graph_protocol_hash(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("graph_protocol_hash", None)
    return stable_hash(clean)
