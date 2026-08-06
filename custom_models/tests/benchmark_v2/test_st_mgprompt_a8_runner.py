from __future__ import annotations

import unittest

from scripts import st_mgprompt_a8_batch4_gate as gate


class A8ReferenceTests(unittest.TestCase):
    def test_reference_payload_contains_explicit_fields(self) -> None:
        payload = gate.build_reference_payload()
        for key in ("scope_id", "model_id", "run_id", "batch_size", "lookback", "node_count", "feature_count", "horizon", "loss_id", "precision", "formal_training"):
            self.assertIn(key, payload)
        self.assertEqual(payload["batch_size"], 4)
        self.assertTrue(payload["formal_training"])


if __name__ == "__main__":
    unittest.main()
