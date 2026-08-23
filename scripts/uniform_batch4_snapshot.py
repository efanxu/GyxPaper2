from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def file_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("before", "after"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = [
        ROOT / "custom_models/src/benchmark_v2/protocol/benchmark_protocol_v1.json",
        ROOT / "custom_models/src/benchmark_v2/training_profiles/uniform_train_batch4_v1.json",
        ROOT / "custom_models/docs/benchmark_v2/BATCH4/CURRENT_BATCH4_SCOPE26_MANIFEST.json",
    ]
    payload = {
        "schema_version": "uniform_batch4_file_inventory_v1",
        "phase": args.phase,
        "files": [file_record(path) for path in paths if path.is_file()],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
