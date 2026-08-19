from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark_v2 import hardware_preflight
from benchmark_v2.contracts import BenchmarkBatch
from benchmark_v2.original_scope26 import CURRENT_SCOPE26_ID
from benchmark_v2.experiments.e5_common_loss.loss_profile import CLI_PROFILE_ID
from benchmark_v2.experiments.e5_common_loss.scope27_contract import E5_SCOPE27_ID


class HardwarePreflightTests(unittest.TestCase):
    def test_preflight_step_executes_optimizer_update(self) -> None:
        model = torch.nn.Linear(2, 1)

        def adapter(model, batch, **_):
            prediction = model(batch.x).squeeze(-1).unsqueeze(1)
            return SimpleNamespace(prediction=prediction)

        runtime = SimpleNamespace(
            model=model,
            adapter=adapter,
            effective_config={
                "amp_enabled": False,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
            },
        )
        batch = BenchmarkBatch(
            x=torch.randn(2, 3, 2),
            target=torch.randn(2, 1, 3),
            mask=torch.ones(2, 1, 3, dtype=torch.bool),
            sample_ids=[0, 1],
            window_end_indices=[0, 1],
            node_ids=[1],
        )
        before = model.weight.detach().clone()
        result = hardware_preflight._preflight_training_step(
            runtime,
            batch,
            {
                "amp_enabled": False,
                "horizon": 3,
                "feature_count": 2,
            },
            lambda prediction, target, mask: (
                (prediction - target).square()[mask].mean()
            ),
            torch.device("cpu"),
        )
        self.assertTrue(result["optimizer_step_pass"])
        self.assertTrue(result["finite"])
        self.assertFalse(torch.equal(before, model.weight.detach()))

    def test_formal_shape_and_explicit_identity(self) -> None:
        value = hardware_preflight.preflight_identity("gcn", formal_scope_id=CURRENT_SCOPE26_ID)
        self.assertEqual((value["batch_size"], value["lookback"], value["node_count"], value["feature_count"], value["horizon"]), (4, 144, 134, 16, 10))

    def test_e5_preflight_is_uniform_fp32(self) -> None:
        value = hardware_preflight.preflight_identity(
            "gcn",
            experiment_profile=CLI_PROFILE_ID,
            training_profile="uniform_train_batch4_v1",
            formal_scope_id=E5_SCOPE27_ID,
        )
        self.assertEqual(value["precision"], "fp32")
        self.assertFalse(value["amp_enabled"])

        original = hardware_preflight.preflight_identity(
            "gcn",
            training_profile="uniform_train_batch4_v1",
            formal_scope_id=CURRENT_SCOPE26_ID,
        )
        self.assertEqual(original["precision"], "profile_default")
        self.assertTrue(original["amp_enabled"])

    def test_pass_requires_forward_backward_finite_and_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = hardware_preflight.preflight_identity("gcn", formal_scope_id=CURRENT_SCOPE26_ID)
            path = hardware_preflight.preflight_artifact_path("gcn", root=root)
            path.parent.mkdir(parents=True)
            payload = {**value, "status": "PASS", "forward_pass": True, "backward_pass": True, "optimizer_step_pass": True, "finite": True, "output_shape": [4, 134, 10]}
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNotNone(hardware_preflight.read_matching_pass("gcn", root=root, formal_scope_id=CURRENT_SCOPE26_ID, source_revision="one"))
            payload["finite"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(hardware_preflight.read_matching_pass("gcn", root=root, formal_scope_id=CURRENT_SCOPE26_ID, source_revision="two"))


if __name__ == "__main__":
    unittest.main()
