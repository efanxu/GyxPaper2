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

    def test_readiness_accepts_explicit_27_of_27(self) -> None:
        manifest = gate.load_manifest()
        def ready(_manifest, entry, _root):
            return {"model_id": entry["model_id"], "run_id": entry.get("e5_run_id"), "run_dir": "fixture", "status": "COMPLETED", "ready": True, "reasons": [], "explicit_config_conflicts": [], "metrics_complete": True, "checkpoint_loadable": True}
        with patch.object(gate, "inspect_run", side_effect=ready):
            report = gate.build_readiness(manifest, gate.EXPECTED_OUTPUT_ROOT)
        self.assertEqual(report["status"], "READY")
        self.assertEqual(report["ready_entries"], 27)


if __name__ == "__main__":
    unittest.main()
