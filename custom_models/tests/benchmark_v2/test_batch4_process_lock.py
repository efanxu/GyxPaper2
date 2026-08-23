from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import st_mgprompt_a8_batch4_gate as a8_gate


class Batch4ProcessLockTests(unittest.TestCase):
    def test_a8_runner_exception_releases_owned_lock_in_finally(self) -> None:
        owner = {
            "scope_id": a8_gate.A8_SCOPE_ID,
            "hostname": "fixture",
            "pid": 1,
            "process_start_time": 1.0,
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(a8_gate, "validate_contract"),
            patch.object(a8_gate, "lock_status", return_value={"status": "ABSENT"}),
            patch.object(
                a8_gate,
                "read_matching_preflight_pass",
                return_value={"status": "PASS"},
            ),
            patch.object(a8_gate, "build_plan", return_value={"action": "RUN_MISSING"}),
            patch.object(a8_gate, "acquire_lock", return_value=owner),
            patch.object(a8_gate, "release_lock", return_value=True) as release,
        ):
            def fail(*_args, **_kwargs):
                raise RuntimeError("fixture runner failure")

            with self.assertRaises(RuntimeError):
                a8_gate.run(log_root=Path(temporary), runner=fail)
            release.assert_called_once_with(owner)


if __name__ == "__main__":
    unittest.main()
