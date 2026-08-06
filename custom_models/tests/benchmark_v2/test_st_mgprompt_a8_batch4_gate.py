from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import st_mgprompt_a8_batch4_gate as gate


class A8GateTests(unittest.TestCase):
    def test_contract_and_preflight_plan(self) -> None:
        contract = gate.validate_contract()
        self.assertEqual(contract["status"], "PASS")
        self.assertEqual(contract["batch_size"], 4)
        plan = gate.build_preflight_plan()
        self.assertEqual(plan["output_shape"], [4, 134, 10])
        self.assertEqual(plan["precision"], "fp32")

    def test_old_extra_metadata_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest.json"
            payload = {"status": "PASS", "scope_id": gate.A8_SCOPE_ID, "model_id": gate.A8_MODEL_ID, "run_id": gate.A8_RUN_ID, "batch_size": 4, "precision": gate.PRECISION_POLICY, "forward_pass": True, "backward_pass": True, "finite": True, "output_shape": [4, 134, 10], "legacy_" + "ha" + "sh": "ignored"}
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNotNone(gate.read_matching_preflight_pass(path))


if __name__ == "__main__":
    unittest.main()
