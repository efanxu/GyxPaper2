from __future__ import annotations

import unittest
from types import SimpleNamespace

from benchmark_v2.experiments.e5_common_loss.contracts import (
    NONTRAINABLE_MODELS,
    TRAINABLE_MODELS,
)
from benchmark_v2.experiments.e5_common_loss.config_diff import compare_configs
from benchmark_v2.experiments.e5_common_loss.loss_profile import CLI_PROFILE_ID
from benchmark_v2.precision import (
    apply_model_precision_policy,
    expected_model_precision_identity,
)


class PrecisionPolicyTests(unittest.TestCase):
    def test_e5_models_use_uniform_fp32(self) -> None:
        for model_id in (*TRAINABLE_MODELS, *NONTRAINABLE_MODELS):
            identity = expected_model_precision_identity(
                model_id,
                "uniform_train_batch4_v1",
                experiment_profile_id=CLI_PROFILE_ID,
            )
            self.assertFalse(identity["amp_enabled"], model_id)
            self.assertEqual(identity["precision_policy"], "fp32", model_id)

            runtime = SimpleNamespace(
                model_id=model_id,
                model=SimpleNamespace(
                    encoder=SimpleNamespace(
                        forward=lambda *_args, **_kwargs: None,
                        conv_layers=None,
                        attn_layers=[],
                        norm=None,
                    )
                ),
                effective_config={
                    "training_batch_profile_id": "uniform_train_batch4_v1",
                    "experiment_profile_id": "e5_common_loss_architecture_v1",
                    "amp_enabled": True,
                },
            )
            apply_model_precision_policy(runtime)
            self.assertFalse(runtime.effective_config["amp_enabled"], model_id)
            self.assertEqual(
                runtime.effective_config["precision_policy"], "fp32", model_id
            )
            if model_id == "patchtst":
                self.assertEqual(
                    runtime.effective_config["activation_checkpointing"],
                    "patchtst_encoder_layers",
                )

    def test_non_e5_models_keep_existing_policy(self) -> None:
        gcn = expected_model_precision_identity(
            "gcn", "uniform_train_batch4_v1"
        )
        self.assertTrue(gcn["amp_enabled"])
        self.assertEqual(gcn["precision_policy"], "profile_default")

        transformer = expected_model_precision_identity(
            "transformer", "uniform_train_batch4_v1"
        )
        self.assertFalse(transformer["amp_enabled"])
        self.assertEqual(transformer["precision_policy"], "fp32")

    def test_e5_precision_and_checkpointing_are_explicit_allowed_diffs(self) -> None:
        result = compare_configs(
            {"amp_enabled": True},
            {
                "amp_enabled": False,
                "precision_policy": "fp32",
                "precision_resolution": {"effective_amp_enabled": False},
                "activation_checkpointing": "patchtst_encoder_layers",
            },
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["unexpected_differences"], [])


if __name__ == "__main__":
    unittest.main()
