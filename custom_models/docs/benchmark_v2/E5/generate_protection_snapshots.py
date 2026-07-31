from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DOC_ROOT = Path(__file__).resolve().parent
EXCLUDED_NAMES = {"__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else path.rglob("*")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            relative_parts = candidate.relative_to(PROJECT_ROOT).parts
            if any(part in EXCLUDED_NAMES for part in relative_parts):
                continue
            if candidate.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            key = candidate.resolve().as_posix().lower()
            if key not in seen:
                seen.add(key)
                yield candidate


def _snapshot(name: str, paths: Iterable[Path], phase: str) -> dict:
    files = []
    for path in sorted(_iter_files(paths), key=lambda item: item.as_posix().lower()):
        stat = path.stat()
        files.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "size": stat.st_size,
                "sha256": _sha256(path),
            }
        )
    logical = hashlib.sha256(
        json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "e5_protection_snapshot_v1",
        "category": name,
        "phase": phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "file_count": len(files),
        "logical_hash": logical,
        "files": files,
    }


def _stage_docs() -> list[Path]:
    base = PROJECT_ROOT / "custom_models" / "docs" / "benchmark_v2"
    return [base / name for name in ("E1_A", "E1_B", "E2_A", "E2_B", "E2_C", "E2_D", "E3_A", "E3_B", "E3_C")]


def categories() -> dict[str, list[Path]]:
    benchmark = PROJECT_ROOT / "custom_models" / "src" / "benchmark_v2"
    return {
        "core_hashes": [
            benchmark / "protocol" / "benchmark_protocol_v1.json",
            PROJECT_ROOT / "custom_models" / "results" / "st_mgprompt_canonical",
        ],
        "st_mgprompt_hashes": [PROJECT_ROOT / "custom_models" / "src" / "st_mgprompt"],
        "ablation_hashes": [
            PROJECT_ROOT / "custom_models" / "results" / "st_mgprompt_component_ablation",
            PROJECT_ROOT / "custom_models" / "results" / "st_mgprompt_method_full_ablation",
            PROJECT_ROOT / "custom_models" / "results" / "st_mgprompt_method_full_test_ablation",
        ],
        "tslib_hashes": [PROJECT_ROOT / "Time-Series-Library"],
        "graph_protocol_hashes": [
            benchmark / "graph",
            PROJECT_ROOT / "custom_models" / "graphs",
        ],
        "benchmark_model_hashes": [
            benchmark / "registry",
            benchmark / "configs",
            benchmark / "models",
            benchmark / "adapters",
            benchmark / "upstream",
        ],
        "stage_docs_hashes": _stage_docs()
        + [
            PROJECT_ROOT / "RUNBOOK_FIXED_DUAL_REFACTOR.md",
            PROJECT_ROOT / "HANDOFF_FIXED_DUAL_REFACTOR.md",
        ],
        "formal_results_inventory": [PROJECT_ROOT / "custom_models" / "results"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    summaries = {}
    for category, paths in categories().items():
        payload = _snapshot(category, paths, args.phase)
        output = DOC_ROOT / f"protected_{category}_{args.phase}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries[category] = {
            "path": str(output),
            "file_count": payload["file_count"],
            "logical_hash": payload["logical_hash"],
        }
    print(json.dumps({"status": "PASS", "phase": args.phase, "snapshots": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
