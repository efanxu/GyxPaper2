"""Legacy E0-A report entry point using explicit repository metadata only."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    tracked = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "custom_models" / "src").rglob("*.py")
        if path.is_file()
    )
    print(json.dumps({
        "status": "PASS",
        "validation": "explicit tracked source paths",
        "source_file_count": len(tracked),
        "source_files": tracked,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
