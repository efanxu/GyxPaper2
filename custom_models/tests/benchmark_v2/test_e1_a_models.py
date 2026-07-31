import unittest

import numpy as np
import torch

from benchmark_v2.adapters import (
    BaselineScalerContext,
    NodeSharedGRUAdapter,
    StatisticalBaselineAdapter,
)
from benchmark_v2.configs import (
    resolve_gru_config,
    resolve_moving_average_config,
    resolve_persistence_config,
)
from benchmark_v2.contracts import BenchmarkBatch
from benchmark_v2.data import StandardScaler
from benchmark_v2.errors import ContractError
from benchmark_v2.losses import masked_mse
from benchmark_v2.model_runtime import build_model_runtime
from benchmark_v2.models.gru import NodeSharedGRU
from benchmark_v2.models.moving_average import MovingAverage, create_model as create_ma
from benchmark_v2.models.persistence import Persistence
from benchmark_v2.protocol import load_protocol
from benchmark_v2.seeds import seed_everything


def scalers():
    input_scaler = StandardScaler(
        mean=np.arange(16, dtype=np.float64) * 10.0,
        std=np.arange(1, 17, dtype=np.float64),
    )
    target_scaler = StandardScaler(
        mean=np.asarray(500.0, dtype=np.float64),
        std=np.asarray(125.0, dtype=np.float64),
    )
    return input_scaler, target_scaler


def batch_from_physical(physical_power):
    protocol = load_protocol()
    input_scaler, target_scaler = scalers()
    physical = torch.as_tensor(physical_power, dtype=torch.float32)
    b, t, n = physical.shape
    x = torch.zeros(b, t, n, 16)
    power_index = protocol["ordered_input_features"].index(
        protocol["input_power_column"]
    )
    x[..., power_index] = (
        physical - float(input_scaler.mean[power_index])
    ) / float(input_scaler.std[power_index])
    target_raw = torch.full((b, n, 10), 700.0)
    target = (target_raw - float(target_scaler.mean)) / float(target_scaler.std)
    batch = BenchmarkBatch(
        x=x,
        target=target,
        mask=torch.ones(b, n, 10, dtype=torch.bool),
        target_raw_or_inverse_transform=target_raw,
        sample_ids=list(range(b)),
        window_end_indices=list(range(b)),
        node_ids=list(range(n)),
        metadata={"contains_future_target": False},
    )
    return batch, input_scaler, target_scaler


class StatisticalBaselineTests(unittest.TestCase):
    def test_persistence_physical_scaler_roundtrip_and_parameters(self):
        physical = torch.zeros(2, 144, 3)
        physical[0, -1] = torch.tensor([800.0, 900.0, 1000.0])
        physical[1, -1] = torch.tensor([400.0, 500.0, 600.0])
        batch, input_scaler, target_scaler = batch_from_physical(physical)
        runtime = build_model_runtime(
            "persistence",
            load_protocol(),
            run_mode="smoke",
            input_scaler=input_scaler,
            target_scaler=target_scaler,
        )
        output = runtime.adapter(runtime.model, batch)
        restored = runtime.inverse_target(output.prediction)
        expected = physical[:, -1].unsqueeze(-1).expand(-1, -1, 10)
        self.assertTrue(torch.allclose(restored, expected, atol=1e-5))
        self.assertTrue(torch.isfinite(output.prediction).all())
        self.assertEqual(runtime.parameter_count, 0)
        self.assertEqual(runtime.trainable_parameter_count, 0)
        self.assertNotEqual(
            runtime.scaler_context.input_power_mean,
            runtime.scaler_context.target_mean,
        )
        self.assertNotEqual(
            runtime.scaler_context.input_power_std,
            runtime.scaler_context.target_std,
        )

    def test_persistence_rejects_nonfinite_last_visible(self):
        batch, input_scaler, target_scaler = batch_from_physical(
            torch.ones(2, 144, 3)
        )
        batch.x[0, -1, 1, 15] = float("nan")
        runtime = build_model_runtime(
            "persistence",
            load_protocol(),
            run_mode="smoke",
            input_scaler=input_scaler,
            target_scaler=target_scaler,
        )
        with self.assertRaisesRegex(ContractError, "last visible"):
            runtime.adapter(runtime.model, batch)

    def test_moving_average_known_values_scaler_and_formal_window(self):
        physical = torch.tensor([100.0, 200.0, 300.0, 400.0]).reshape(1, 4, 1)
        batch, input_scaler, target_scaler = batch_from_physical(physical)
        runtime = build_model_runtime(
            "moving_average",
            load_protocol(),
            run_mode="smoke",
            input_scaler=input_scaler,
            target_scaler=target_scaler,
            ma_window=4,
        )
        output = runtime.adapter(runtime.model, batch)
        restored = runtime.inverse_target(output.prediction)
        self.assertTrue(torch.allclose(restored, torch.full((1, 1, 10), 250.0)))
        formal = create_ma(
            {"run_mode": "formal", "ma_window": 144}, load_protocol()
        )
        self.assertEqual(tuple(formal(torch.ones(3, 144)).shape), (3, 10))

    def test_moving_average_fail_closed_rules_and_protocol_stability(self):
        protocol = load_protocol()
        before = protocol.protocol_hash
        with self.assertRaisesRegex(ContractError, "frozen"):
            resolve_moving_average_config(
                protocol, run_mode="formal", ma_window=4
            )
        smoke = resolve_moving_average_config(
            protocol, run_mode="smoke", ma_window=4
        )
        self.assertEqual(smoke["ma_window"], 4)
        self.assertEqual(load_protocol().protocol_hash, before)
        model = MovingAverage(horizon=10, ma_window=4)
        with self.assertRaisesRegex(ContractError, "T >= ma_window"):
            model(torch.ones(2, 3))
        bad = torch.ones(2, 4)
        bad[0, 2] = float("inf")
        with self.assertRaisesRegex(ContractError, "finite"):
            model(bad)

    def test_scaler_context_resolves_power_index_from_protocol(self):
        protocol = load_protocol()
        input_scaler, target_scaler = scalers()
        context = BaselineScalerContext.from_scalers(
            protocol, input_scaler, target_scaler
        )
        self.assertEqual(
            protocol["ordered_input_features"][context.power_feature_index],
            "Patv_clean_for_input",
        )
        self.assertEqual(context.feature_order_hash, protocol["feature_order_hash"])


class NodeSharedGRUTests(unittest.TestCase):
    def setUp(self):
        seed_everything(2026)
        self.protocol = load_protocol()
        self.runtime = build_model_runtime(
            "gru", self.protocol, run_mode="smoke"
        )

    def make_batch(self, b=2, t=144, n=4):
        x = torch.randn(b, t, n, 16)
        target = torch.randn(b, n, 10)
        mask = torch.ones(b, n, 10, dtype=torch.bool)
        return BenchmarkBatch(
            x=x,
            target=target,
            mask=mask,
            sample_ids=list(range(b)),
            window_end_indices=list(range(b)),
            node_ids=list(range(n)),
            metadata={"contains_future_target": False},
        )

    def test_shape_structure_and_fixed_parameter_count(self):
        batch = self.make_batch()
        output = self.runtime.adapter(self.runtime.model, batch)
        self.assertEqual(tuple(output.prediction.shape), (2, 4, 10))
        self.assertEqual(self.runtime.parameter_count, 16394)
        self.assertEqual(self.runtime.trainable_parameter_count, 16394)
        self.assertEqual(
            sum(isinstance(module, torch.nn.GRU) for module in self.runtime.model.modules()),
            1,
        )
        self.assertEqual(
            sum(
                isinstance(module, torch.nn.Linear)
                for module in self.runtime.model.modules()
            ),
            1,
        )
        self.assertFalse(
            any(
                isinstance(module, (torch.nn.ModuleList, torch.nn.Embedding))
                for module in self.runtime.model.modules()
            )
        )

    def test_parameter_sharing_node_independence_and_ordering(self):
        batch = self.make_batch(b=1, n=3)
        batch.x[:, :, 1] = batch.x[:, :, 0]
        self.runtime.model.eval()
        first = self.runtime.adapter(self.runtime.model, batch).prediction.detach()
        self.assertTrue(torch.equal(first[:, 0], first[:, 1]))
        unchanged_node = first[:, 2].clone()
        batch.x[:, :, 0] += 100.0
        second = self.runtime.adapter(self.runtime.model, batch).prediction.detach()
        self.assertTrue(torch.equal(second[:, 2], unchanged_node))
        flattened = self.runtime.adapter.prepare_model_inputs(batch)
        restored = flattened.reshape(1, 3, 144, 16).permute(0, 2, 1, 3)
        self.assertTrue(torch.equal(restored, batch.x))
        self.assertEqual(self.runtime.parameter_count, 16394)
        larger = self.make_batch(b=1, n=7)
        self.runtime.adapter(self.runtime.model, larger)
        self.assertEqual(self.runtime.parameter_count, 16394)

    def test_masked_backward_gradients_and_invalid_positions(self):
        batch = self.make_batch()
        batch.mask.zero_()
        batch.mask[:, :, :3] = True
        output = self.runtime.adapter(self.runtime.model, batch)
        output.prediction.retain_grad()
        loss = masked_mse(output.prediction, batch.target, batch.mask)
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(parameter.grad is not None for parameter in self.runtime.model.gru.parameters())
        )
        self.assertIsNotNone(self.runtime.model.prediction_head.weight.grad)
        self.assertTrue(
            torch.equal(
                output.prediction.grad[:, :, 3:],
                torch.zeros_like(output.prediction.grad[:, :, 3:]),
            )
        )
        if torch.cuda.is_available():
            cuda_runtime = build_model_runtime(
                "gru", self.protocol, run_mode="smoke"
            )
            cuda_batch = self.make_batch().x.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                raw = cuda_runtime.model.cuda()(
                    cuda_batch.permute(0, 2, 1, 3)
                    .contiguous()
                    .reshape(8, 144, 16)
                )
            self.assertTrue(torch.isfinite(raw).all())

    def test_configs_record_frozen_optimizer_and_semantics(self):
        config = resolve_gru_config(self.protocol, run_mode="formal")
        self.assertEqual(config["optimizer"], "Adam")
        self.assertEqual(config["learning_rate"], 0.001)
        self.assertEqual(config["weight_decay"], 0.0)
        self.assertIsNone(config["scheduler"])
        self.assertIsNone(config["gradient_clip"])
        self.assertFalse(config["uses_graph"])
        self.assertFalse(config["uses_node_embedding"])
        self.assertFalse(config["uses_future_covariates"])
        self.assertEqual(
            resolve_persistence_config(self.protocol)["optimizer"], None
        )


if __name__ == "__main__":
    unittest.main()
