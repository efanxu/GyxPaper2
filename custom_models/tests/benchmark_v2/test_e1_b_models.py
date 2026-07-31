import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark_v2.adapters import (
    DLinearAdapter,
    LightTSAdapter,
    SegRNNAdapter,
    TiDEAdapter,
)
from benchmark_v2.checkpointing import CheckpointManager
from benchmark_v2.configs import (
    resolve_dlinear_config,
    resolve_lightts_config,
    resolve_segrnn_config,
    resolve_tide_config,
)
from benchmark_v2.contracts import BenchmarkBatch
from benchmark_v2.errors import ContractError
from benchmark_v2.losses import masked_mse
from benchmark_v2.model_runtime import build_model_runtime
from benchmark_v2.protocol import load_protocol
from benchmark_v2.seeds import seed_everything
from benchmark_v2.upstream import load_tslib_model_class, resolve_tslib_source


MODEL_IDS = ("dlinear", "lightts", "tide", "segrnn")


def make_batch(b=2, n=4, seed=2026):
    generator = torch.Generator().manual_seed(seed)
    return BenchmarkBatch(
        x=torch.randn(b, 144, n, 16, generator=generator),
        target=torch.randn(b, n, 10, generator=generator),
        mask=torch.ones(b, n, 10, dtype=torch.bool),
        sample_ids=list(range(b)),
        window_end_indices=list(range(b)),
        node_ids=list(range(n)),
        metadata={"contains_future_target": False},
    )


class E1BModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(2026)
        cls.protocol = load_protocol()

    def build(self, model_id):
        return build_model_runtime(model_id, self.protocol, run_mode="smoke")

    def test_all_models_shape_finite_backward_and_provenance(self):
        expected_parameters = {
            "dlinear": 2900,
            "lightts": 104342,
            "tide": 1804777,
            "segrnn": 1583874,
        }
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                runtime = self.build(model_id)
                batch = make_batch()
                output = runtime.adapter(runtime.model, batch)
                self.assertEqual(tuple(output.prediction.shape), (2, 4, 10))
                self.assertTrue(torch.isfinite(output.prediction).all())
                self.assertEqual(
                    output.semantic_trace["selected_feature"],
                    "Patv_clean_for_input",
                )
                self.assertEqual(
                    output.semantic_trace["supervised_target"], "Patv_raw"
                )
                self.assertEqual(
                    output.semantic_trace["output_policy"],
                    "select_power_associated_channel",
                )
                loss = masked_mse(output.prediction, batch.target, batch.mask)
                self.assertIsNotNone(loss)
                loss.backward()
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        for parameter in runtime.model.parameters()
                        if parameter.requires_grad
                    )
                )
                self.assertEqual(
                    runtime.parameter_count, expected_parameters[model_id]
                )
                self.assertEqual(
                    runtime.effective_config["upstream_source_sha256"],
                    resolve_tslib_source(model_id).source_sha256,
                )
                self.assertFalse(runtime.effective_config["source_modified"])
                self.assertFalse(
                    runtime.effective_config["validation_search_performed"]
                )
                self.assertFalse(runtime.effective_config["test_result_used"])

    def test_node_independence_ordering_and_parameter_count_do_not_depend_on_n(self):
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                runtime = self.build(model_id)
                runtime.model.eval()
                batch = make_batch(b=1, n=3)
                batch.x[:, :, 1] = batch.x[:, :, 0]
                first = runtime.adapter(runtime.model, batch).prediction.detach()
                self.assertTrue(torch.equal(first[:, 0], first[:, 1]))
                unaffected = first[:, 2].clone()
                batch.x[:, :, 0] += 100.0
                second = runtime.adapter(runtime.model, batch).prediction.detach()
                self.assertTrue(torch.equal(second[:, 2], unaffected))
                flat = runtime.adapter.prepare_model_inputs(batch)
                if model_id == "tide":
                    flat = flat[0]
                restored = flat.reshape(1, 3, 144, 16).permute(0, 2, 1, 3)
                self.assertTrue(torch.equal(restored, batch.x))
                count = runtime.parameter_count
                runtime.adapter(runtime.model, make_batch(b=1, n=7))
                self.assertEqual(runtime.parameter_count, count)
                names = tuple(name.casefold() for name, _ in runtime.model.named_parameters())
                self.assertFalse(any("node_embedding" in name for name in names))
                self.assertFalse(any("node_head" in name for name in names))

    def test_targets_and_masks_are_not_model_inputs(self):
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                runtime = self.build(model_id)
                runtime.model.eval()
                first_batch = make_batch(b=1, n=2)
                second_batch = copy.copy(first_batch)
                second_batch.target = torch.randn_like(first_batch.target) * 1000
                second_batch.mask = torch.zeros_like(first_batch.mask)
                first = runtime.adapter(
                    runtime.model, first_batch
                ).prediction.detach()
                second = runtime.adapter(
                    runtime.model, second_batch
                ).prediction.detach()
                self.assertTrue(torch.equal(first, second))

    def test_dynamic_power_channel_selection_and_bad_feature_order_rejected(self):
        adapter_classes = (
            DLinearAdapter,
            LightTSAdapter,
            TiDEAdapter,
            SegRNNAdapter,
        )
        batch = make_batch(b=1, n=2)
        raw = torch.zeros(2, 10, 16)
        raw[..., 15] = 7.0
        for adapter_class in adapter_classes:
            adapter = adapter_class(self.protocol)
            output = adapter.normalize_output(raw, batch)
            self.assertTrue(torch.equal(output.prediction, torch.full((1, 2, 10), 7.0)))
            bad = self.protocol.to_dict()
            bad["ordered_input_features"] = list(
                reversed(bad["ordered_input_features"])
            )
            with self.assertRaisesRegex(ContractError, "order hash"):
                adapter_class(bad)

    def test_strict_checkpoint_reload_for_each_model(self):
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id), tempfile.TemporaryDirectory() as td:
                runtime = self.build(model_id)
                config = runtime.effective_config
                manager = CheckpointManager(
                    td,
                    protocol_hash=self.protocol.protocol_hash,
                    model_id=model_id,
                    resolved_config=config,
                    effective_config=config,
                )
                path = manager.save(
                    "best_checkpoint.pt",
                    epoch=1,
                    global_step=1,
                    monitor_value=1.0,
                    model=runtime.model,
                )
                expected = {
                    key: value.detach().clone()
                    for key, value in runtime.model.state_dict().items()
                }
                with torch.no_grad():
                    next(runtime.model.parameters()).add_(1.0)
                manager.load(path, runtime.model)
                for key, value in runtime.model.state_dict().items():
                    self.assertTrue(torch.equal(value, expected[key]))

    def test_dlinear_decomposition_and_lightts_chunks_preserve_144(self):
        dlinear = self.build("dlinear")
        seasonal, trend = dlinear.model.decompsition(torch.randn(2, 144, 16))
        self.assertEqual(tuple(seasonal.shape), (2, 144, 16))
        self.assertEqual(tuple(trend.shape), (2, 144, 16))
        lightts = self.build("lightts")
        self.assertEqual(lightts.model.seq_len, 144)
        self.assertEqual(lightts.model.chunk_size, 8)
        self.assertEqual(lightts.model.num_chunks, 18)
        x = torch.randn(2, 144, 16)
        continuous = x.reshape(2, 18, 8, 16)
        interval = x.reshape(2, 8, 18, 16)
        self.assertEqual(tuple(continuous.shape), (2, 18, 8, 16))
        self.assertEqual(tuple(interval.shape), (2, 8, 18, 16))

    def test_tide_zero_placeholders_and_nonzero_future_covariates_rejected(self):
        runtime = self.build("tide")
        runtime.model.eval()
        batch = make_batch(b=1, n=2)
        output = runtime.adapter(runtime.model, batch)
        trace = output.aux["zero_placeholder_trace"]
        self.assertTrue(trace["x_mark_enc_all_zero"])
        self.assertTrue(trace["y_mark_all_zero"])
        self.assertTrue(trace["x_dec_all_zero"])
        self.assertEqual(trace["x_mark_enc_shape"], (2, 144, 4))
        self.assertEqual(trace["y_mark_shape"], (2, 10, 4))
        with self.assertRaisesRegex(ContractError, "future observed"):
            runtime.adapter(
                runtime.model,
                batch,
                future_observed_covariates=torch.ones(2, 10, 1),
            )

    def test_segrnn_original_failure_and_protocol_compatible_shape(self):
        model_class, _ = load_tslib_model_class("segrnn")
        original = SimpleNamespace(
            seq_len=144,
            pred_len=10,
            enc_in=16,
            d_model=512,
            dropout=0.1,
            task_name="long_term_forecast",
            seg_len=96,
        )
        original_model = model_class(original)
        self.assertEqual(original_model.seg_num_x, 1)
        self.assertEqual(original_model.seg_num_y, 0)
        with self.assertRaisesRegex(RuntimeError, r"size of tensor a \(0\)"):
            original_model(torch.randn(8, 144, 16), None, None, None)
        fixed = self.build("segrnn")
        self.assertEqual(fixed.model.seq_len, 144)
        self.assertEqual(fixed.model.pred_len, 10)
        self.assertEqual(fixed.model.seg_len, 2)
        self.assertEqual(fixed.model.seg_num_x, 72)
        self.assertEqual(fixed.model.seg_num_y, 5)
        self.assertEqual(
            tuple(fixed.model(torch.randn(8, 144, 16), None, None, None).shape),
            (8, 10, 16),
        )

    def test_frozen_configs_are_not_test_selected(self):
        resolvers = (
            resolve_dlinear_config,
            resolve_lightts_config,
            resolve_tide_config,
            resolve_segrnn_config,
        )
        for resolver in resolvers:
            config = resolver(self.protocol, run_mode="formal")
            self.assertEqual(config["seq_len"], 144)
            self.assertEqual(config["pred_len"], 10)
            self.assertEqual(config["enc_in"], 16)
            self.assertFalse(config["validation_search_performed"])
            self.assertFalse(config["test_result_used"])
        self.assertEqual(resolve_lightts_config(self.protocol)["chunk_size"], 8)
        self.assertEqual(resolve_segrnn_config(self.protocol)["seg_len"], 2)


if __name__ == "__main__":
    unittest.main()
