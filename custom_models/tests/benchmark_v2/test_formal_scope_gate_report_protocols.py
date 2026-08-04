from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import e5_batch4_scope27_gate as e5_gate
from scripts import original_batch4_scope26_gate as original_gate


class FormalScopeGateReportProtocolTests(unittest.TestCase):
    def test_original_preflight_entrypoint_emits_one_machine_readable_json(self):
        payload = {
            "status": "BLOCK_EXISTING_IDENTITY_MISMATCH",
            "scope_id": original_gate.CURRENT_SCOPE26_ID,
            "gpu_preflight_performed": False,
            "child_launch_count": 0,
        }
        output = io.StringIO()
        with (
            patch.object(original_gate, "run_preflight_suite", return_value=(74, payload)),
            redirect_stdout(output),
        ):
            code = original_gate.main(["preflight"])
        self.assertEqual(code, 74)
        parsed = json.loads(output.getvalue())
        self.assertEqual(parsed, payload)
        self.assertEqual(output.getvalue().count("\"status\""), 1)

    def test_e5_preflight_blocks_before_any_child_when_a8_is_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_path = root / "e5_preflight.json"
            with patch.object(e5_gate.subprocess, "run") as child:
                code, payload = e5_gate.run_preflight_suite(
                    {},
                    preflight_root=root / "preflight",
                    source_revision="fixture-revision",
                    report_path=report_path,
                    child_log_root=root / "child_logs",
                )
            self.assertEqual(code, 74)
            self.assertEqual(payload["status"], "BLOCKED_A8_BATCH4_PREREQUISITE")
            self.assertEqual(payload["child_launch_count"], 0)
            self.assertFalse(child.called)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), payload)


if __name__ == "__main__":
    unittest.main()
