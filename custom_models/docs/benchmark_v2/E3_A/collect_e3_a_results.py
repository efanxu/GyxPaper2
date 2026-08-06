"""Legacy E3-A report entry point for explicit graph validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = PROJECT_ROOT / "custom_models" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmark_v2.graph.provider import graph_protocol_check


def main() -> None:
    print(json.dumps(graph_protocol_check(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
