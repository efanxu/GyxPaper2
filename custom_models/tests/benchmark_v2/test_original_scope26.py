from __future__ import annotations

import unittest

from benchmark_v2.original_scope26 import CURRENT_SCOPE26_ID, is_current_scope26_request, load_current_scope_manifest
from scripts.original_batch4_scope26_gate import build_preflight_plan, compute_original_freeze, validate_manifest


class OriginalScopeTests(unittest.TestCase):
    def test_manifest_protocol_and_denominator(self) -> None:
        manifest = load_current_scope_manifest()
        result = validate_manifest(manifest)
        self.assertEqual(result["scope_id"], CURRENT_SCOPE26_ID)
        self.assertEqual(result["counts"], {"trainable": 24, "evaluate_only": 2, "total": 26})
        self.assertEqual(set(result["excluded"]), {"segrnn", "msgnet"})
        snapshot = compute_original_freeze(manifest)
        self.assertEqual(snapshot["batch"], 4)
        self.assertEqual(snapshot["gradient_accumulation_steps"], 1)
        self.assertEqual(snapshot["seed"], 2026)
        self.assertEqual(snapshot["eval_horizons"], [3, 6, 10])

    def test_authorization_and_preflight_plan(self) -> None:
        manifest = load_current_scope_manifest()
        entry = next(row for row in manifest["entries"] if row["entry_type"] == "TRAINABLE")
        common = {"formal_scope_id": CURRENT_SCOPE26_ID, "experiment_profile": None, "training_profile": "uniform_train_batch4_v1", "trainable": True, "manifest": manifest}
        self.assertTrue(is_current_scope26_request(model_id=entry["model_id"], **common))
        self.assertFalse(is_current_scope26_request(model_id="segrnn", **common))
        plan = build_preflight_plan(manifest)
        self.assertEqual(plan["required"], 24)
        self.assertTrue(all(row["batch_size"] == 4 for row in plan["entries"]))


if __name__ == "__main__":
    unittest.main()
