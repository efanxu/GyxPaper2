from __future__ import annotations

import unittest

import torch

from benchmark_v2.experiments.e5_common_loss.contracts import NONTRAINABLE_MODELS, TRAINABLE_MODELS
from benchmark_v2.experiments.e5_common_loss.loss_profile import CLI_PROFILE_ID, get_profile_metadata, loss_for_profile
from benchmark_v2.experiments.e5_common_loss.variant_manifest import build_variant_manifest


class E5CommonLossTests(unittest.TestCase):
    def test_active_denominator_and_exclusions(self) -> None:
        manifest = build_variant_manifest()
        self.assertEqual(manifest["counts"], {"trainable": 24, "evaluate_only": 2, "a8_reference": 1, "total": 27})
        self.assertEqual(len(TRAINABLE_MODELS), 24)
        self.assertEqual(len(NONTRAINABLE_MODELS), 2)
        ids = {row["model_id"] for row in manifest["entries"]}
        self.assertFalse({"segrnn", "msgnet"} & ids)

    def test_common_loss_is_finite(self) -> None:
        profile = get_profile_metadata(CLI_PROFILE_ID)
        function = loss_for_profile(profile["cli_profile_id"])
        prediction = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
        target = torch.tensor([[[1.5, 1.5, 2.5]]])
        mask = torch.ones_like(target, dtype=torch.bool)
        value = function(prediction, target, mask)
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == "__main__":
    unittest.main()
