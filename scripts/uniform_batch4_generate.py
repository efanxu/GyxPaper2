from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.uniform_batch4_machine_gate import current_freeze_identity


ROOT = Path(__file__).resolve().parents[1]


def collect_preflight(suite: str) -> dict[str, object]:
    root = ROOT / "custom_models/logs/uniform_bs4/preflight" / suite
    records = []
    if root.is_dir():
        for path in sorted(root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            records.append({"path": str(path), "status": payload.get("status"), "model_id": payload.get("model_id")})
    return {"schema_version": "uniform_batch4_preflight_inventory_v1", "suite": suite, "records": records}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--output", type=Path)
    collect = sub.add_parser("collect-preflight")
    collect.add_argument("--suite", choices=("known_oom6", "all26"), required=True)
    args = parser.parse_args()
    payload = current_freeze_identity() if args.command == "generate" else collect_preflight(args.suite)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
