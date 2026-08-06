from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_v2.original_scope26 import load_current_scope_manifest
from scripts.original_batch4_scope26_gate import _lock_status, archive_existing_attempt


class OriginalResumeTests(unittest.TestCase):
    def test_archive_preview_preserves_existing_attempt(self) -> None:
        manifest = load_current_scope_manifest()
        entry = manifest["entries"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / entry["run_id"]
            run_dir.mkdir()
            (run_dir / "run_status.json").write_text("{}", encoding="utf-8")
            receipt = archive_existing_attempt(entry, root, apply=False)
            self.assertTrue(run_dir.is_dir())
            self.assertEqual(receipt["status"], "PREVIEW")
            self.assertEqual(receipt["file_count"], 1)

    def test_lock_states_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scope.lock.json"
            self.assertEqual(_lock_status(path)["status"], "ABSENT")
            path.write_text(json.dumps({"scope_id": "x"}), encoding="utf-8")
            self.assertEqual(_lock_status(path)["status"], "MALFORMED")


if __name__ == "__main__":
    unittest.main()
