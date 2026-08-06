from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..errors import ProtocolError


PROTOCOL_PATH = Path(__file__).with_name("benchmark_protocol_v1.json")


@dataclass(frozen=True)
class BenchmarkProtocol(Mapping[str, Any]):
    payload: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __iter__(self):
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def load_protocol(path: str | Path | None = None) -> BenchmarkProtocol:
    source = Path(path) if path else PROTOCOL_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("Protocol JSON root must be an object.")
    return BenchmarkProtocol(payload)
