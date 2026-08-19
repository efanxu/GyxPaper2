from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import e5_batch4_scope27_gate as gate


class E5Scope27Tests(unittest.TestCase):
    def test_manifest_and_preflight_denominator(self) -> None:
        manifest = gate.load_manifest()
        result = gate.validate_manifest(manifest)
        self.assertEqual(result["counts"], {"trainable": 24, "evaluate_only": 2, "a8_reference": 1, "total": 27})
        self.assertEqual(result["loss_id"], "masked_score_aligned_hybrid")
        plan = gate.build_preflight_plan(manifest)
        self.assertEqual(plan["required"], 24)
        self.assertTrue(all(row["batch_size"] == 4 for row in plan["entries"]))
        benchmark = [row for row in manifest["entries"] if row["entry_type"] != "REFERENCE_ONLY_FORMAL_A8"]
        self.assertEqual(len(benchmark), 26)
        self.assertTrue(all(row["e5_run_id"].endswith("_bs4_seed2026") for row in benchmark))
        self.assertTrue(
            all(
                row["precision_identity"]["precision_policy"] == "fp32"
                and row["precision_identity"]["amp_enabled"] is False
                for row in benchmark
            )
        )
        self.assertEqual(manifest["entries"][-1]["e5_run_id"], "STMGPrompt_A8_loss_msa_hybrid_bs4_seed2026_reference")

    def test_readiness_accepts_explicit_27_of_27(self) -> None:
        manifest = gate.load_manifest()
        fixture = {"status": "READY", "ready_entries": 27, "expected_total_entries": 27, "counts": {"trainable": 24, "evaluate_only": 2, "a8_reference": 1, "total": 27}, "entries": []}
        with patch.object(gate, "build_common_readiness", return_value=fixture):
            report = gate.build_readiness(manifest, gate.EXPECTED_OUTPUT_ROOT)
        self.assertEqual(report["status"], "READY")
        self.assertEqual(report["ready_entries"], 27)


if __name__ == "__main__":
    unittest.main()
