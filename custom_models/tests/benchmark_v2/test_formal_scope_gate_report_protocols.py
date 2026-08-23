from __future__ import annotations

import unittest

from scripts import original_batch4_scope26_gate as original_gate
from scripts import st_mgprompt_a8_batch4_gate as a8_gate


class FormalScopeStaticReportTests(unittest.TestCase):
    def test_static_entrypoints_return_machine_readable_payloads(self) -> None:
        original = original_gate.load_current_scope_manifest()
        self.assertEqual(original_gate.build_plan(original, original_gate.RESULT_ROOT)["scope_id"], original_gate.CURRENT_SCOPE26_ID)
        self.assertEqual(a8_gate.build_plan()["scope_id"], a8_gate.A8_SCOPE_ID)


if __name__ == "__main__":
    unittest.main()
