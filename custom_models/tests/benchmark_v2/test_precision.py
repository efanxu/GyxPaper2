from __future__ import annotations

import unittest
from types import SimpleNamespace

from benchmark_v2.precision import (
    FP32_MODEL_IDS,
    apply_model_precision_policy,
    expected_model_precision_identity,
)


class PrecisionPolicyTests(unittest.TestCase):
    def test_uniform_batch4_known_nonfinite_models_use_fp32(self) -> None:
        for model_id in FP32_MODEL_IDS:
            identity = expected_model_precision_identity(
                model_id, "uniform_train_batch4_v1"
            )
            self.assertFalse(identity["amp_enabled"], model_id)
            self.assertEqual(identity["precision_policy"], "fp32")

            runtime = SimpleNamespace(
                model_id=model_id,
                effective_config={
                    "training_batch_profile_id": "uniform_train_batch4_v1",
                    "amp_enabled": True,
                },
            )
            apply_model_precision_policy(runtime)
            self.assertFalse(runtime.effective_config["amp_enabled"], model_id)
            self.assertEqual(runtime.effective_config["precision_policy"], "fp32")

    def test_other_uniform_batch4_models_keep_amp(self) -> None:
        identity = expected_model_precision_identity(
            "timesnet", "uniform_train_batch4_v1"
        )
        self.assertTrue(identity["amp_enabled"])
        self.assertEqual(identity["precision_policy"], "profile_default")


if __name__ == "__main__":
    unittest.main()
