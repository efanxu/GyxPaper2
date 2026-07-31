from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..errors import ProtocolError
from .hashing import protocol_hash


PROTOCOL_PATH = Path(__file__).with_name("benchmark_protocol_v1.json")


@dataclass(frozen=True)
class BenchmarkProtocol(Mapping[str, Any]):
    payload: dict[str, Any]
    protocol_hash: str

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __iter__(self):
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.payload)
        result["protocol_hash"] = self.protocol_hash
        return result


def load_protocol(path: str | Path | None = None) -> BenchmarkProtocol:
    source = Path(path) if path else PROTOCOL_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("Protocol JSON root must be an object.")
    calculated = protocol_hash(payload)
    declared = payload.get("protocol_hash")
    if declared not in (None, calculated):
        raise ProtocolError(f"Protocol hash mismatch: declared={declared}, calculated={calculated}")
    payload["protocol_hash"] = calculated
    return BenchmarkProtocol(payload, calculated)
