from __future__ import annotations

import json
import tempfile
import unittest
import socket
from pathlib import Path

from scripts import e5_batch4_scope27_gate as gate


class E5ScopeSafetyTests(unittest.TestCase):
    def test_snapshot_is_explicit_and_ordered(self) -> None:
        manifest = gate.load_manifest()
        snapshot = gate.compute_e5_freeze(manifest)
        self.assertEqual(snapshot["batch"], 4)
        self.assertEqual(snapshot["seed"], 2026)
        self.assertEqual([row["ordinal"] for row in snapshot["entries"]], list(range(1, 28)))

    def test_lock_absent_malformed_and_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "e5.lock.json"
            self.assertEqual(gate.lock_status(path)["status"], "ABSENT")
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(gate.lock_status(path)["status"], "MALFORMED")
            path.write_text(json.dumps({"schema_version": "explicit_process_lock_v1", "scope_id": gate.E5_SCOPE_ID, "hostname": socket.gethostname(), "pid": 99999999, "process_start_time": 1.0, "created_at": "now"}), encoding="utf-8")
            self.assertEqual(gate.lock_status(path)["status"], "STALE")


if __name__ == "__main__":
    unittest.main()
