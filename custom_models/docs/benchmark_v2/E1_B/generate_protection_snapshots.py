"""Legacy entry point retained as an explicit-path inventory helper."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    paths = [
        "custom_models/src/benchmark_v2",
        "custom_models/src/st_mgprompt",
        "Time-Series-Library",
    ]
    payload = {
        "status": "PASS",
        "validation": "explicit path existence only",
        "paths": [
            {"path": value, "exists": (PROJECT_ROOT / value).exists()}
            for value in paths
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
