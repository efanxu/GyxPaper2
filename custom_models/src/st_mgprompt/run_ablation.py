from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from st_mgprompt.formal_runner import run_family


def main() -> None:
    try:
        result = run_family("component_ablation")
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
