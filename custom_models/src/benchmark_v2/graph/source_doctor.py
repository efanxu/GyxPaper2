"""Fail-closed diagnostics and repair for the frozen graph location source.

The graph protocol hashes the location CSV as raw bytes.  This module keeps
that identity check explicit: it never normalizes bytes for acceptance and it
only offers an apply operation when the difference is proven to be line
endings on the exact frozen path.  The CSV is pinned to
``233a120f4db2b52402bf4dc7c90631ac9301d6af236a56c6a245e46beec1f236`` and
must remain LF; Excel, an IDE, or Git automatic conversion must not rewrite
it.  The graph protocol intentionally does not normalize line endings.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[4]
LOCATION_RELATIVE_PATH = "dataset/sdwpf_turb_location_elevation.csv"
FROZEN_LOCATION_SHA256 = (
    "233a120f4db2b52402bf4dc7c90631ac9301d6af236a56c6a245e46beec1f236"
)
LOCATION_EOL_RULE = f"{LOCATION_RELATIVE_PATH} text eol=lf"
FROZEN_GRAPH_PROTOCOL_HASH = (
    "f8224287f2a41be0e67e808445ce9a43b9973e7b075d1e1b4768396d6f98e1ef"
)
FROZEN_GRAPH_BUNDLE_HASH = (
    "388a87dcba4fa0b897ab9b40b546d844433a26c799d50f4a009d832c6e335f0e"
)
RECEIPT_SCHEMA_VERSION = "graph_source_repair_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _normalise_line_endings(value: bytes) -> bytes:
    return value.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _run_git(project_root: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=False,
        capture_output=True,
        text=text,
        encoding="utf-8" if text else None,
        errors="replace" if text else None,
    )


def _git_text(project_root: Path, *args: str) -> str | None:
    completed = _run_git(project_root, *args)
    if completed.returncode != 0:
        return None
    return str(completed.stdout).strip()


def _git_blob_bytes(project_root: Path) -> bytes | None:
    completed = _run_git(
        project_root,
        "cat-file",
        "blob",
        f"HEAD:{LOCATION_RELATIVE_PATH}",
        text=False,
    )
    if completed.returncode != 0:
        return None
    return bytes(completed.stdout)


def _git_attribute_lines(project_root: Path) -> list[str]:
    value = _git_text(project_root, "check-attr", "-a", "--", LOCATION_RELATIVE_PATH)
    return [] if value is None else [line for line in value.splitlines() if line]


def _local_attributes(project_root: Path) -> list[str]:
    path = project_root / ".git" / "info" / "attributes"
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _has_exact_attribute_rule(project_root: Path) -> bool:
    path = project_root / ".gitattributes"
    if not path.is_file():
        return False
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        fields = line.split()
        if not fields or fields[0] != LOCATION_RELATIVE_PATH:
            continue
        attributes = set(fields[1:])
        if "text" in attributes and "eol=lf" in attributes:
            return True
    return False


def _has_local_exact_override(project_root: Path) -> bool:
    for raw_line in _local_attributes(project_root):
        line = raw_line.split("#", 1)[0].strip()
        fields = line.split()
        if not fields or fields[0] != LOCATION_RELATIVE_PATH:
            continue
        if any(field in {"-text", "text=auto", "text=false"} for field in fields[1:]):
            return True
    return False


def _diff_present(project_root: Path, *, staged: bool, path: str | None = LOCATION_RELATIVE_PATH) -> bool | None:
    args = ["diff"]
    if staged:
        args.append("--cached")
    args.extend(["--quiet"])
    if path is not None:
        args.extend(["--", path])
    completed = _run_git(project_root, *args)
    if completed.returncode not in (0, 1):
        return None
    return completed.returncode == 1


def _global_status(project_root: Path) -> str | None:
    return _git_text(project_root, "status", "--porcelain", "--untracked-files=no")


def _graph_protocol_status(project_root: Path) -> dict[str, Any]:
    protocol_path = project_root / "custom_models/src/benchmark_v2/protocol/graph_protocol_v1.json"
    bundle_dir = project_root / "custom_models/src/benchmark_v2/protocol/graph_v1"
    location_path = project_root / LOCATION_RELATIVE_PATH
    try:
        from .provider import load_graph_bundle

        bundle = load_graph_bundle(
            protocol_path=protocol_path,
            bundle_dir=bundle_dir,
            location_path=location_path,
            validate_location_source=False,
        )
        if bundle.graph_protocol_hash != FROZEN_GRAPH_PROTOCOL_HASH:
            raise ValueError(
                f"graph_protocol_hash={bundle.graph_protocol_hash}"
            )
        if bundle.graph_bundle_hash != FROZEN_GRAPH_BUNDLE_HASH:
            raise ValueError(f"graph_bundle_hash={bundle.graph_bundle_hash}")
        return {
            "status": "PASS",
            "graph_protocol_hash": bundle.graph_protocol_hash,
            "graph_bundle_hash": bundle.graph_bundle_hash,
        }
    except Exception as exc:
        return {
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def inspect_graph_source(project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return one JSON-serialisable status object for the exact CSV path."""

    root = Path(project_root).resolve()
    source_path = root / LOCATION_RELATIVE_PATH
    worktree_bytes = source_path.read_bytes() if source_path.is_file() else None
    head_bytes = _git_blob_bytes(root)
    worktree_sha = None if worktree_bytes is None else _sha256_bytes(worktree_bytes)
    head_sha = None if head_bytes is None else _sha256_bytes(head_bytes)
    staged_diff = _diff_present(root, staged=True)
    worktree_diff = _diff_present(root, staged=False)
    tracked_status = _git_text(
        root,
        "status",
        "--short",
        "--untracked-files=no",
        "--",
        LOCATION_RELATIVE_PATH,
    )
    attributes = _git_attribute_lines(root)
    local_attributes = _local_attributes(root)
    graph_status = _graph_protocol_status(root)
    reasons: list[str] = []
    if not source_path.is_file():
        reasons.append("WORKTREE_FILE_MISSING")
    if head_bytes is None:
        reasons.append("HEAD_BLOB_MISSING")
    if head_sha != FROZEN_LOCATION_SHA256:
        reasons.append("HEAD_BLOB_HASH_MISMATCH")
    if worktree_sha != FROZEN_LOCATION_SHA256:
        if worktree_bytes is not None and head_bytes is not None and _normalise_line_endings(worktree_bytes) == head_bytes:
            reasons.append("WORKTREE_LINE_ENDING_MISMATCH")
        else:
            reasons.append("WORKTREE_CONTENT_MISMATCH")
    if staged_diff:
        reasons.append("STAGED_MODIFICATION")
    if worktree_diff:
        reasons.append("WORKTREE_MODIFICATION")
    if not _has_exact_attribute_rule(root):
        reasons.append("GITATTRIBUTES_EXACT_EOL_RULE_MISSING")
    if _has_local_exact_override(root):
        reasons.append("LOCAL_ATTRIBUTES_OVERRIDE_PRESENT")
    if graph_status.get("status") != "PASS":
        reasons.append("GRAPH_PROTOCOL_CHECK_FAILED")

    repository_inconsistent = "HEAD_BLOB_HASH_MISMATCH" in reasons
    only_line_endings = "WORKTREE_LINE_ENDING_MISMATCH" in reasons and not any(
        reason in reasons
        for reason in (
            "WORKTREE_CONTENT_MISMATCH",
            "STAGED_MODIFICATION",
            "WORKTREE_MODIFICATION",
        )
    )
    repair_safe = bool(
        not repository_inconsistent
        and head_bytes is not None
        and worktree_bytes is not None
        and not staged_diff
        and _has_exact_attribute_rule(root)
        and only_line_endings
    )
    blocking_reasons = {
        "WORKTREE_FILE_MISSING",
        "HEAD_BLOB_MISSING",
        "HEAD_BLOB_HASH_MISMATCH",
        "WORKTREE_LINE_ENDING_MISMATCH",
        "WORKTREE_CONTENT_MISMATCH",
        "STAGED_MODIFICATION",
        "WORKTREE_MODIFICATION",
        "GITATTRIBUTES_EXACT_EOL_RULE_MISSING",
        "GRAPH_PROTOCOL_CHECK_FAILED",
    }
    if repository_inconsistent:
        status = "REPOSITORY_PROTOCOL_INCONSISTENCY"
    elif any(reason in blocking_reasons for reason in reasons):
        status = "REPAIR_REQUIRED"
    else:
        status = "PASS"

    git_head_blob_sha1 = _git_text(
        root, "rev-parse", f"HEAD:{LOCATION_RELATIVE_PATH}"
    )
    index_entry = _git_text(root, "ls-files", "-s", "--", LOCATION_RELATIVE_PATH)
    index_blob_sha = None
    if index_entry:
        fields = index_entry.split()
        index_blob_sha = fields[1] if len(fields) > 1 else None
    worktree_git_blob_sha = _git_text(root, "hash-object", "--", LOCATION_RELATIVE_PATH)
    return {
        "status": status,
        "relative_path": LOCATION_RELATIVE_PATH,
        "expected_sha256": FROZEN_LOCATION_SHA256,
        "worktree_sha256": worktree_sha,
        "git_head_blob_sha256": head_sha,
        "index_blob_sha": index_blob_sha,
        "worktree_git_blob_sha": worktree_git_blob_sha,
        "contains_crlf": bool(worktree_bytes is not None and b"\r\n" in worktree_bytes),
        "git_attributes": {
            "check_attr": attributes,
            "info_attributes": local_attributes,
            "repository_rule_present": _has_exact_attribute_rule(root),
        },
        "git_ls_files_eol": _git_text(root, "ls-files", "--eol", "--", LOCATION_RELATIVE_PATH),
        "core_autocrlf": _git_text(root, "config", "--get", "core.autocrlf"),
        "core_eol": _git_text(root, "config", "--get", "core.eol"),
        "tracked_status": tracked_status or "",
        "staged_diff": bool(staged_diff),
        "worktree_diff": bool(worktree_diff),
        "repair_required": bool(any(reason in blocking_reasons for reason in reasons)),
        "repair_safe": repair_safe,
        "reasons": reasons,
        "graph_protocol_check": graph_status,
    }


def checkout_identity(project_root: str | Path = PROJECT_ROOT) -> dict[str, str]:
    root = Path(project_root).resolve()
    attributes_path = root / ".gitattributes"
    return {
        "gitattributes_path": ".gitattributes",
        "gitattributes_sha256": _sha256_file(attributes_path),
        "location_eol_policy": LOCATION_EOL_RULE,
        "location_raw_sha256": FROZEN_LOCATION_SHA256,
    }


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.{uuid.uuid4().hex}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_receipt(receipt_root: Path, payload: Mapping[str, Any]) -> Path:
    receipt_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = receipt_root / f"{timestamp}-{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _other_tracked_changes(project_root: Path) -> list[str]:
    values: list[str] = []
    for args in (("diff", "--cached", "--name-only"), ("diff", "--name-only")):
        output = _git_text(project_root, *args) or ""
        values.extend(line for line in output.splitlines() if line and line != LOCATION_RELATIVE_PATH)
    return sorted(set(values))


def repair_graph_source(
    project_root: str | Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    receipt_root: str | Path | None = None,
) -> dict[str, Any]:
    """Preview or apply a proven line-ending-only repair for the exact path."""

    root = Path(project_root).resolve()
    status = inspect_graph_source(root)
    source_path = root / LOCATION_RELATIVE_PATH
    before_bytes = source_path.read_bytes() if source_path.is_file() else b""
    head_bytes = _git_blob_bytes(root)
    receipt_directory = Path(receipt_root) if receipt_root is not None else root / "custom_models/logs/uniform_bs4/audit/graph_source_repair"
    base = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_text(root, "rev-parse", "HEAD"),
        "relative_path": LOCATION_RELATIVE_PATH,
        "expected_sha256": FROZEN_LOCATION_SHA256,
        "before_sha256": _sha256_bytes(before_bytes),
        "after_sha256": status.get("worktree_sha256"),
        "head_blob_sha": status.get("git_head_blob_sha256"),
        "head_blob_git_sha": _git_text(root, "rev-parse", f"HEAD:{LOCATION_RELATIVE_PATH}"),
        "before_contains_crlf": bool(b"\r\n" in before_bytes),
        "after_contains_crlf": status.get("contains_crlf"),
        "attributes_seen": status.get("git_attributes"),
        "core_autocrlf": status.get("core_autocrlf"),
        "action": "PREVIEW" if not apply else "APPLY",
        "status": "PREVIEW_ONLY" if not apply else "PENDING",
        "rollback_performed": False,
    }
    if status["status"] == "PASS":
        base["status"] = "NO_REPAIR_REQUIRED"
        receipt = _write_receipt(receipt_directory, base)
        return {**status, "status": "NO_REPAIR_REQUIRED", "action": base["action"], "receipt_path": str(receipt)}
    if status["status"] == "REPOSITORY_PROTOCOL_INCONSISTENCY":
        base["status"] = status["status"]
        base["reasons"] = status["reasons"]
        receipt = _write_receipt(receipt_directory, base)
        return {**status, "action": base["action"], "receipt_path": str(receipt)}
    if not status.get("repair_safe"):
        base["status"] = "REPAIR_REFUSED"
        base["reasons"] = status["reasons"]
        receipt = _write_receipt(receipt_directory, base)
        return {**status, "status": "REPAIR_REFUSED", "action": base["action"], "receipt_path": str(receipt)}
    if head_bytes is None:
        base["status"] = "REPOSITORY_PROTOCOL_INCONSISTENCY"
        base["reasons"] = ["HEAD_BLOB_MISSING"]
        receipt = _write_receipt(receipt_directory, base)
        return {**status, "status": base["status"], "action": base["action"], "receipt_path": str(receipt)}
    if not apply:
        receipt = _write_receipt(receipt_directory, base)
        return {**status, "status": "PREVIEW_ONLY", "action": "PREVIEW", "receipt_path": str(receipt)}
    if _other_tracked_changes(root):
        base["status"] = "REPAIR_REFUSED"
        base["reasons"] = ["OTHER_TRACKED_WORKTREE_CHANGES"]
        receipt = _write_receipt(receipt_directory, base)
        return {**status, "status": "REPAIR_REFUSED", "action": "APPLY", "receipt_path": str(receipt)}

    try:
        _atomic_write_bytes(source_path, head_bytes)
        after = inspect_graph_source(root)
        global_status = _global_status(root) or ""
        if (
            after["status"] != "PASS"
            or after["worktree_sha256"] != FROZEN_LOCATION_SHA256
            or after["git_head_blob_sha256"] != FROZEN_LOCATION_SHA256
            or after["staged_diff"]
            or after["worktree_diff"]
            or global_status
        ):
            raise RuntimeError(
                "Post-repair repository identity is not clean: "
                f"status={after['status']} global_status={global_status!r}"
            )
        base.update(
            {
                "after_sha256": after["worktree_sha256"],
                "after_contains_crlf": after["contains_crlf"],
                "attributes_seen": after["git_attributes"],
                "status": "PASS",
            }
        )
        receipt = _write_receipt(receipt_directory, base)
        return {**after, "status": "PASS", "action": "APPLY", "receipt_path": str(receipt)}
    except Exception as exc:
        rollback = False
        try:
            _atomic_write_bytes(source_path, before_bytes)
            rollback = True
        finally:
            base.update(
                {
                    "after_sha256": _sha256_file(source_path) if source_path.is_file() else None,
                    "after_contains_crlf": bool(source_path.is_file() and b"\r\n" in source_path.read_bytes()),
                    "status": "REPAIR_FAILED",
                    "rollback_performed": rollback,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        receipt = _write_receipt(receipt_directory, base)
        return {
            **inspect_graph_source(root),
            "status": "REPAIR_FAILED",
            "action": "APPLY",
            "receipt_path": str(receipt),
            "rollback_performed": rollback,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


__all__ = [
    "FROZEN_GRAPH_BUNDLE_HASH",
    "FROZEN_GRAPH_PROTOCOL_HASH",
    "FROZEN_LOCATION_SHA256",
    "LOCATION_EOL_RULE",
    "LOCATION_RELATIVE_PATH",
    "checkout_identity",
    "inspect_graph_source",
    "repair_graph_source",
]
