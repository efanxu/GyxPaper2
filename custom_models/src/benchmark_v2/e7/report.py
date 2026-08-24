from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import ANALYSIS_ROOT
from .io_utils import write_json


def build_artifact_manifest(root: str | Path = ANALYSIS_ROOT) -> dict[str, Any]:
    base = Path(root)
    files = []
    if base.is_dir():
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            stat = path.stat()
            files.append({"path": str(path.relative_to(base)), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {
        "schema_version": "e7_artifact_manifest_v1",
        "root": str(base.resolve()),
        "files": files,
        "content_hashes": "NOT_COMPUTED_BY_REPOSITORY_POLICY",
        "training_executed": False,
        "source_results_modified": False,
    }


def write_artifact_manifest(root: str | Path = ANALYSIS_ROOT) -> Path:
    return write_json(Path(root) / "E7_ARTIFACT_MANIFEST.json", build_artifact_manifest(root))
