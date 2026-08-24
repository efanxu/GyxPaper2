from __future__ import annotations

from typing import Iterable

import numpy as np


def spearman_relationships(rows: Iterable[dict], *, outcome: str, mechanisms: Iterable[str]) -> list[dict]:
    materialized = list(rows)
    from scipy.stats import spearmanr

    result = []
    for mechanism in mechanisms:
        pairs = [(float(row[mechanism]), float(row[outcome])) for row in materialized if row.get(mechanism) is not None and row.get(outcome) is not None]
        if len(pairs) < 3:
            result.append({"outcome": outcome, "mechanism": mechanism, "count": len(pairs), "spearman": None, "status": "INSUFFICIENT_DATA"})
            continue
        left, right = np.asarray(pairs).T
        value = spearmanr(left, right).statistic
        result.append({"outcome": outcome, "mechanism": mechanism, "count": len(pairs), "spearman": None if not np.isfinite(value) else float(value), "status": "OK"})
    return result
