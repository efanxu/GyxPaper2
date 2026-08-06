from __future__ import annotations

import json

from scripts.uniform_batch4_machine_gate import precheck, verify


def main() -> int:
    payload = {
        "schema_version": "uniform_batch4_final_status_v1",
        "protocol": precheck(),
        "original": verify("original"),
        "a8": verify("a8"),
        "e5": verify("e5"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
