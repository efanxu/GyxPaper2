from __future__ import annotations

import unittest

from scripts import st_mgprompt_a8_batch4_gate as gate


class A8ReferenceTests(unittest.TestCase):
    def test_reference_payload_contains_explicit_fields(self) -> None:
        payload = gate.build_reference_payload()
        for key in ("reference_type", "reference_id", "scope_id", "model_id", "run_id", "training_profile_id", "train_batch_size", "val_batch_size", "test_batch_size", "gradient_accumulation_steps", "seed", "lookback", "max_pred_len", "loss_id", "precision", "formal_training", "source_path", "source_config_path", "checkpoint_path", "source_metrics_paths"):
            self.assertIn(key, payload)
        self.assertEqual(payload["train_batch_size"], 4)
        self.assertTrue(payload["formal_training"])
        for legacy in ("metrics_paths", "run_dir", "batch_size"):
            self.assertNotIn(legacy, payload)


if __name__ == "__main__":
    unittest.main()
