from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class ProcessLockError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_start_time(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return pid > 0 and psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except Exception:
        return False


def current_owner(scope_id: str) -> dict[str, Any]:
    pid = os.getpid()
    started = _process_start_time(pid)
    if started is None:
        raise ProcessLockError("Cannot determine the current process start time.")
    return {
        "schema_version": "explicit_process_lock_v1",
        "scope_id": scope_id,
        "hostname": socket.gethostname(),
        "pid": pid,
        "process_start_time": started,
        "created_at": _utc_now(),
    }


def lock_status(path: str | Path, *, scope_id: str | None = None) -> dict[str, Any]:
    selected = Path(path)
    if not selected.is_file():
        return {"status": "ABSENT", "path": str(selected)}
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "MALFORMED", "path": str(selected)}
    required = {
        "schema_version", "scope_id", "hostname", "pid",
        "process_start_time", "created_at",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        return {"status": "MALFORMED", "path": str(selected)}
    if payload.get("schema_version") != "explicit_process_lock_v1":
        return {"status": "MALFORMED", "path": str(selected), "owner": payload}
    if scope_id is not None and payload.get("scope_id") != scope_id:
        return {"status": "MALFORMED", "path": str(selected), "owner": payload}
    try:
        pid = int(payload["pid"])
        expected_start = float(payload["process_start_time"])
    except (TypeError, ValueError):
        return {"status": "MALFORMED", "path": str(selected), "owner": payload}
    if payload["hostname"] != socket.gethostname():
        return {"status": "ACTIVE", "path": str(selected), "owner": payload}
    if not _pid_alive(pid):
        return {"status": "STALE", "path": str(selected), "owner": payload}
    actual_start = _process_start_time(pid)
    if actual_start is None or abs(actual_start - expected_start) >= 1:
        return {"status": "STALE", "path": str(selected), "owner": payload}
    return {"status": "ACTIVE", "path": str(selected), "owner": payload}


def acquire_lock(path: str | Path, *, scope_id: str) -> dict[str, Any]:
    selected = Path(path)
    state = lock_status(selected, scope_id=scope_id)
    if state["status"] != "ABSENT":
        raise ProcessLockError(f"LOCK_{state['status']}:{selected}")
    owner = current_owner(scope_id)
    selected.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(selected, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raced = lock_status(selected, scope_id=scope_id)
        raise ProcessLockError(f"LOCK_{raced['status']}:{selected}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
        json.dump(owner, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return owner


def owned_by(owner: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    keys = ("scope_id", "hostname", "pid", "process_start_time")
    return all(candidate.get(key) == owner.get(key) for key in keys)


def release_lock(path: str | Path, *, owner: Mapping[str, Any]) -> bool:
    selected = Path(path)
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return False
    if not isinstance(payload, dict) or not owned_by(owner, payload):
        return False
    selected.unlink()
    return True


def clear_stale_lock(path: str | Path, *, scope_id: str) -> dict[str, Any]:
    selected = Path(path)
    state = lock_status(selected, scope_id=scope_id)
    if state["status"] == "STALE":
        selected.unlink()
        return {"status": "CLEARED", "path": str(selected)}
    return state


__all__ = [
    "ProcessLockError", "acquire_lock", "clear_stale_lock", "current_owner",
    "lock_status", "owned_by", "release_lock",
]
