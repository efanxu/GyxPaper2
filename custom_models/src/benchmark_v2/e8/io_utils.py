from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fields=None) -> Path:
    target = Path(path)
    materialized = list(rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(materialized[0]) if materialized else []
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, target)
    return target


def finite_number(value: Any) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError("Metric value is missing or boolean.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Metric value is non-finite.")
    return result


def read_metrics_csv(path: str | Path) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            horizon = int(row["horizon"])
            result[horizon] = {metric: finite_number(row.get(metric)) for metric in ("Score", "MAE", "RMSE", "R2")}
    if set(result) != {3, 6, 10}:
        raise ValueError(f"Metrics must contain exactly H3/H6/H10: {path}")
    return result


def forbidden_source(path: str | Path) -> str | None:
    normalized = str(path).replace("\\", "/").lower()
    from .constants import FORBIDDEN_SOURCE_TOKENS
    return next((token for token in FORBIDDEN_SOURCE_TOKENS if token.replace("\\", "/") in normalized), None)


def file_identity(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    stat = target.stat()
    return {"path": str(target.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
