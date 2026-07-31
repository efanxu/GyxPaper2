from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def protocol_hash(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("protocol_hash", None)
    return stable_hash(clean)

