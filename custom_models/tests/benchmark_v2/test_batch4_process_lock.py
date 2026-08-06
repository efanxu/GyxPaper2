from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "custom_models/src"
for selected in (PROJECT_ROOT, SOURCE_ROOT):
    if str(selected) not in sys.path:
        sys.path.insert(0, str(selected))

from benchmark_v2.process_lock import ProcessLockError
from scripts import e5_batch4_scope27_gate as gate
from scripts import st_mgprompt_a8_batch4_gate as a8_gate


class Batch4ProcessLockTests(unittest.TestCase):
    def test_absent_acquire_active_second_start_and_owner_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scope.lock.json"
            self.assertEqual(gate.lock_status(path)["status"], "ABSENT")
            owner = gate.acquire_lock(path)
            self.assertEqual(gate.lock_status(path)["status"], "ACTIVE")
            with self.assertRaises(ProcessLockError):
                gate.acquire_lock(path)
            non_owner = {**owner, "pid": owner["pid"] + 1}
            self.assertFalse(gate.release_lock(non_owner, path))
            self.assertTrue(path.is_file())
            self.assertTrue(gate.release_lock(owner, path))
            self.assertEqual(gate.lock_status(path)["status"], "ABSENT")

    def test_dead_owner_is_stale_and_exception_finally_releases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scope.lock.json"
            stale = {
                "schema_version": "explicit_process_lock_v1",
                "scope_id": gate.E5_SCOPE_ID,
                "hostname": socket.gethostname(),
                "pid": 99999999,
                "process_start_time": 1.0,
                "created_at": "2026-08-06T00:00:00Z",
            }
            path.write_text(json.dumps(stale), encoding="utf-8")
            self.assertEqual(gate.lock_status(path)["status"], "STALE")
            self.assertEqual(gate.clear_stale_lock(path)["status"], "CLEARED")
            owner = gate.acquire_lock(path)
            try:
                raise RuntimeError("fixture failure")
            except RuntimeError:
                pass
            finally:
                gate.release_lock(owner, path)
            self.assertEqual(gate.lock_status(path)["status"], "ABSENT")

    def test_active_run_directory_cannot_be_archived(self) -> None:
        manifest = gate.load_manifest()
        entry = manifest["entries"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / entry["e5_run_id"]
            run_dir.mkdir()
            import psutil

            (run_dir / "run_status.json").write_text(json.dumps({
                "status": "RUNNING", "pid": os.getpid(),
                "process_start_time": float(psutil.Process(os.getpid()).create_time()),
            }), encoding="utf-8")
            with self.assertRaises(gate.ScopeGateError):
                gate.archive_or_quarantine(manifest, entry, root, apply=True)
            self.assertTrue(run_dir.is_dir())

    def test_a8_runner_exception_releases_owned_lock_in_finally(self) -> None:
        owner = {"scope_id": a8_gate.A8_SCOPE_ID, "hostname": "fixture", "pid": 1, "process_start_time": 1.0}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(a8_gate, "validate_contract"), \
                patch.object(a8_gate, "lock_status", return_value={"status": "ABSENT"}), \
                patch.object(a8_gate, "read_matching_preflight_pass", return_value={"status": "PASS"}), \
                patch.object(a8_gate, "build_plan", return_value={"action": "RUN_MISSING"}), \
                patch.object(a8_gate, "acquire_lock", return_value=owner), \
                patch.object(a8_gate, "release_lock", return_value=True) as release:
            def fail(*_args, **_kwargs):
                raise RuntimeError("fixture runner failure")

            with self.assertRaises(RuntimeError):
                a8_gate.run(log_root=Path(temporary), runner=fail)
            release.assert_called_once_with(owner)


if __name__ == "__main__":
    unittest.main()
