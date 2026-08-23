from __future__ import annotations

import unittest

from scripts.uniform_batch4_machine_gate import current_freeze_identity, precheck, preflight_inventory


class UniformBatch4GateTests(unittest.TestCase):
    def test_explicit_snapshot(self) -> None:
        snapshot = current_freeze_identity()
        self.assertEqual(snapshot["batch_size"], 4)
        self.assertEqual(snapshot["seed"], 2026)
        self.assertEqual(snapshot["original"]["counts"]["total"], 26)
        self.assertEqual(precheck()["status"], "PASS")

    def test_preflight_inventory(self) -> None:
        self.assertEqual(preflight_inventory("original")["required"], 24)


if __name__ == "__main__":
    unittest.main()
