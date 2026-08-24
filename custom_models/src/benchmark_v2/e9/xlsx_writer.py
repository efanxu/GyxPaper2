from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _artifact_runtime() -> tuple[Path, Path]:
    node = os.environ.get("E9_ARTIFACT_NODE")
    modules = os.environ.get("E9_ARTIFACT_NODE_MODULES")
    if not node or not modules:
        raise RuntimeError("ARTIFACT_TOOL_RUNTIME_MISSING: set E9_ARTIFACT_NODE and E9_ARTIFACT_NODE_MODULES.")
    node_path, modules_path = Path(node), Path(modules)
    if not node_path.is_file() or not modules_path.is_dir():
        raise RuntimeError("ARTIFACT_TOOL_RUNTIME_INVALID")
    return node_path, modules_path


def _ensure_module_link(work_dir: Path, modules: Path) -> None:
    link = work_dir / "node_modules"
    if link.exists():
        if link.resolve() != modules.resolve():
            raise RuntimeError("ARTIFACT_TOOL_NODE_MODULE_LINK_CONFLICT")
        return
    try:
        os.symlink(modules, link, target_is_directory=True)
    except OSError:
        completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(modules)], capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"ARTIFACT_TOOL_NODE_MODULE_LINK_FAILED:{completed.stderr.strip()}")


def write_workbook(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    node, modules = _artifact_runtime()
    work_dir = target.parent / ".e9_artifact_runtime"
    work_dir.mkdir(parents=True, exist_ok=True)
    _ensure_module_link(work_dir, modules)
    input_path = work_dir / "workbook_input.json"
    input_path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    builder = work_dir / "build_workbook.mjs"
    shutil.copyfile(Path(__file__).with_name("build_workbook.mjs"), builder)
    completed = subprocess.run([str(node), str(builder), str(input_path), str(target)], cwd=work_dir, capture_output=True, text=True, encoding="utf-8", check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ARTIFACT_TOOL_EXPORT_FAILED:{completed.stderr.strip() or completed.stdout.strip()}")
    if not target.is_file():
        raise RuntimeError("ARTIFACT_TOOL_EXPORT_MISSING")
    return target
