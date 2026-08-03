from __future__ import annotations

import argparse
import json
import os
import subprocess
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOCK_SCHEMA_VERSION = "e5_scope27_lock_v1"


def _read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read lock JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Lock JSON root must be an object.")
    return payload


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return completed.returncode == 0 and str(pid) in completed.stdout
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def inspect_lock(path: Path) -> int:
    if not path.is_file():
        print(json.dumps({"status": "ABSENT", "path": str(path)}))
        return 0
    payload = _read(path)
    pid = int(payload.get("pid", 0))
    status = "ACTIVE" if _pid_alive(pid) else "STALE"
    print(json.dumps({"status": status, "path": str(path), **payload}))
    return 0 if status == "ACTIVE" else 74


def acquire_lock(path: Path, args: argparse.Namespace) -> int:
    if path.exists():
        existing = _read(path)
        pid = int(existing.get("pid", 0))
        if _pid_alive(pid):
            print(
                f"ACTIVE_LOCK: refusing concurrent scope27 run pid={pid} path={path}",
                file=sys.stderr,
            )
            return 73
        print(
            f"STALE_LOCK: explicit clear-stale is required before restart: {path}",
            file=sys.stderr,
        )
        return 74
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "scope_id": args.scope_id,
        "pid": int(args.pid),
        "hostname": args.hostname or socket.gethostname(),
        "started_at": args.started_at
        or datetime.now(timezone.utc).isoformat(),
        "git_commit": args.git_commit,
        "python_executable": args.python_executable,
        "manifest_sha256": args.manifest_sha256,
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError:
        print(f"ACTIVE_LOCK: concurrent lock creation won: {path}", file=sys.stderr)
        return 73
    print(json.dumps({"status": "ACQUIRED", "path": str(path), **payload}))
    return 0


def clear_stale_lock(path: Path) -> int:
    if not path.is_file():
        print(f"ABSENT_LOCK: {path}")
        return 0
    payload = _read(path)
    pid = int(payload.get("pid", 0))
    if _pid_alive(pid):
        print(
            f"ACTIVE_LOCK: refusing clear while pid={pid} is alive: {path}",
            file=sys.stderr,
        )
        return 73
    path.unlink()
    print(f"CLEARED_STALE_LOCK: {path}")
    return 0


def release_lock(path: Path, pid: int) -> int:
    if not path.is_file():
        return 0
    payload = _read(path)
    owner = int(payload.get("pid", 0))
    if owner != pid:
        print(
            f"LOCK_OWNER_MISMATCH: lock pid={owner}, requested pid={pid}",
            file=sys.stderr,
        )
        return 75
    path.unlink()
    print(f"RELEASED_LOCK: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="e5_scope27_lock")
    parser.add_argument("action", choices=("acquire", "inspect", "clear-stale", "release"))
    parser.add_argument("--path", required=True)
    parser.add_argument("--scope-id", default="e5_batch4_scope27_seed2026")
    parser.add_argument("--pid", type=int, default=os.getpid())
    parser.add_argument("--hostname")
    parser.add_argument("--started-at")
    parser.add_argument("--git-commit")
    parser.add_argument("--python-executable")
    parser.add_argument("--manifest-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.path).resolve()
    try:
        if args.action == "inspect":
            return inspect_lock(path)
        if args.action == "acquire":
            return acquire_lock(path, args)
        if args.action == "clear-stale":
            return clear_stale_lock(path)
        return release_lock(path, args.pid)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
