from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import st_mgprompt_a8_batch4_gate as gate


class STMGPromptA8Batch4GateTests(unittest.TestCase):
    def test_contract_is_independent_and_provable(self):
        payload = gate.validate_contract()
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["model_id"], "st_mgprompt_a8")
        self.assertEqual(payload["variant"], "A8")
        self.assertEqual(payload["definition"], "w/o MS-MG-DWU")
        self.assertTrue(payload["source_closure_hash"])
        self.assertTrue(payload["graph_identity_hash"])
        self.assertEqual(payload["source_closure"]["dynamic_imports"], [])

    def test_historical_batch32_a8_cannot_satisfy_current_readiness(self):
        readiness = gate.build_readiness()
        self.assertEqual(readiness["status"], "BLOCKED_A8_BATCH4_PREREQUISITE")
        self.assertIn(
            "HISTORICAL_BATCH32_A8_NOT_CURRENT",
            readiness["artifact"]["reasons"],
        )
        self.assertFalse(readiness["formal_training_executed"])

    def test_smoke_or_noncanonical_a8_candidate_is_not_treated_as_missing(self):
        with tempfile.TemporaryDirectory(dir=gate.PROJECT_ROOT) as temp:
            root = Path(temp)
            candidate = root / "smoke_attempt" / "STMGPrompt_ComponentAblation"
            candidate.mkdir(parents=True)
            (candidate / "effective_config.json").write_text(
                json.dumps(
                    {
                        "model_id": "st_mgprompt_a8",
                        "variant": "A8",
                        "component_ablation": "A8",
                        "smoke": True,
                        "run_mode": "smoke",
                    }
                ),
                encoding="utf-8",
            )
            artifact = gate.inspect_a8_artifact(
                output_root=root,
                project_root=gate.PROJECT_ROOT,
            )
        self.assertEqual(artifact["action"], "BLOCK_EXISTING_IDENTITY_MISMATCH")
        self.assertIn("SMOKE_A8_NOT_FORMAL", artifact["reasons"])

    def test_preflight_plan_is_exact_and_does_not_execute(self):
        plan = gate.build_preflight_plan()
        self.assertEqual(plan["exact_shape"], {"B": 4, "T": 144, "N": 134, "C": 16, "H": 10})
        self.assertFalse(plan["gpu_preflight_performed"])
        self.assertEqual(plan["loss_id"], "masked_score_aligned_hybrid")
        self.assertFalse(plan["execution_performed"])

    def test_formal_run_stops_before_child_without_exact_preflight(self):
        with patch.object(gate, "read_matching_preflight_pass", return_value=None), patch.object(
            gate.subprocess, "run"
        ) as child:
            code, report = gate.run()
        self.assertEqual(code, 74)
        self.assertEqual(report["status"], "PREFLIGHT_MISSING_OR_MISMATCH")
        self.assertFalse(report["gpu_child_started"])
        child.assert_not_called()

    def test_metrics_require_finite_values_and_positive_denominators(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for horizon in gate.REQUIRED_HORIZONS:
                payload = {
                    "horizon": horizon,
                    "MAE": 1.0,
                    "RMSE": 2.0,
                    "R2": 0.5,
                    "Score": 3.0,
                    "valid_target_count": 4,
                }
                (root / f"metrics_eval_h{horizon}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                rows.append(payload)
            with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["horizon", "MAE", "RMSE", "R2", "Score", "valid_target_count"])
                writer.writeheader()
                writer.writerows(rows)
            metrics, reasons, bundle_hash, csv_hash = gate._metric_validation(root)
            self.assertFalse(reasons)
            self.assertEqual(sorted(metrics), [3, 6, 10])
            self.assertTrue(bundle_hash)
            self.assertTrue(csv_hash)


if __name__ == "__main__":
    unittest.main()
