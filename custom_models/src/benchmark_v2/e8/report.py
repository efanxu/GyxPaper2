from __future__ import annotations

from pathlib import Path

from .io_utils import write_json


def write_artifact_manifest(root: str | Path) -> Path:
    root = Path(root)
    files = []
    if root.is_dir():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            stat = path.stat()
            files.append({"relative_path": path.relative_to(root).as_posix(), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return write_json(root / "E8_ARTIFACT_MANIFEST.json", {
        "schema_version": "e8_artifact_manifest_v1", "analysis_root": str(root.resolve()),
        "identity_policy": "path_size_mtime; content hashes are intentionally not generated", "files": files,
    })


def write_markdown_report(path: str | Path, readiness: dict, internal_rows: list[dict], external_rows: list[dict]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# E8 Prompt and Cross-Fusion Analysis", "",
        "Internal Ablation is the primary component-level causal evidence.", "",
        "Original26 External Context is native-training-system architecture and end-to-end competitiveness context only; it is not a unified-loss structure-controlled comparison.", "",
        "E5 is cancelled. No common-loss external table exists. A8 is not E8 evidence.", "",
        "Score/MAE/RMSE are lower-is-better; R2 is higher-is-better. Correlation diagnostics are not causal proof. Missing values are never replaced with zero.", "",
        f"CORE_E8_READY: `{readiness['CORE_E8_READY']}`", "",
        f"Internal rows: {len(internal_rows)}; external context rows: {len(external_rows)}. All formal numbers were loaded automatically.", "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
