"""Legacy E5 documentation entry point for explicit manifest validation."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MANIFEST = PROJECT_ROOT / "custom_models/docs/benchmark_v2/E5/E5_SCOPE27_VARIANT_MANIFEST.json"


def main() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = list(payload.get("entries", []))
    counts = {
        "trainable": sum(row.get("entry_type") == "TRAIN_COMMON_LOSS" for row in entries),
        "evaluate_only": sum(
            row.get("entry_type") == "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC"
            for row in entries
        ),
        "a8_reference": sum(row.get("model_id") == "st_mgprompt_a8" for row in entries),
        "total": len(entries),
    }
    status = "PASS" if counts == {
        "trainable": 24,
        "evaluate_only": 2,
        "a8_reference": 1,
        "total": 27,
    } else "FAIL"
    print(json.dumps({
        "status": status,
        "scope_id": payload.get("scope_id"),
        "counts": counts,
    }, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
