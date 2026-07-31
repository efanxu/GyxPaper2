import copy
import unittest
from types import SimpleNamespace

import torch

from benchmark_v2.contracts import BenchmarkBatch
from benchmark_v2.errors import ContractError
from benchmark_v2.hardware_preflight import PREFLIGHT_SHAPE, preflight_identity
from benchmark_v2.losses import masked_mse
from benchmark_v2.model_runtime import build_model_runtime
from benchmark_v2.protocol import load_protocol
from benchmark_v2.registry import load_registry
from benchmark_v2.upstream import ALLOWED_TSLIB_MODELS, load_tslib_model_class


MODEL_IDS = ("timesnet", "micn", "wpmixer", "multipatchformer")


def make_batch(b=1, n=1, seed=2026):
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


def historical_args(model_id):
    common = dict(
        task_name="long_term_forecast",
        seq_len=144,
        label_len=48,
        pred_len=10,
        enc_in=16,
        c_out=16,
        d_model=16,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        d_ff=32,
        dropout=0.0,
        embed="timeF",
        freq="h",
    )
    return SimpleNamespace(**common)


class E2BModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.protocol = load_protocol()

    def build(self, model_id):
        runtime = build_model_runtime(model_id, self.protocol, run_mode="smoke")
        runtime.model.eval()
        return runtime

    def test_allowlist_is_exactly_eighteen_and_registry_counts(self):
        self.assertEqual(
            ALLOWED_TSLIB_MODELS,
            (
                "dlinear", "lightts", "tide", "segrnn",
                "transformer", "patchtst", "itransformer", "timexer",
                "timesnet", "micn", "wpmixer", "multipatchformer",
                "timemixer", "tsmixer", "frets",
                "crossformer", "msgnet", "timefilter",
            ),
        )
        registry = load_registry()
        stats = registry.statistics()
        self.assertEqual(stats["total_entries"], 28)
        self.assertEqual(
            stats["available_locally_verified"]
            + stats["available_with_hardware_preflight"],
            26,
        )
        self.assertEqual(stats["non_trainable_available"], 2)
        self.assertEqual(stats["true_blocked_or_unavailable"], 0)
        for model_id in MODEL_IDS:
            entry = registry.get(model_id)
            self.assertTrue(entry.supports_train)
            self.assertTrue(entry.supports_evaluate)
            self.assertEqual(entry.node_semantics, "node_shared")
        for model_id in ("timesnet", "micn"):
            entry = registry.get(model_id)
            self.assertEqual(entry.runtime_status, "AVAILABLE_TRAINABLE")
            self.assertFalse(entry.formal_hardware_preflight_required)
            self.assertEqual(entry.local_full_shape_status, "PASS")
        for model_id in ("wpmixer", "multipatchformer"):
            entry = registry.get(model_id)
            self.assertEqual(
                entry.runtime_status,
                "AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED",
            )
            self.assertTrue(entry.formal_hardware_preflight_required)
            self.assertEqual(entry.local_full_shape_status, "FAIL_OOM")
            self.assertEqual(entry.non_oom_engineering_checks, "PASS")

    def test_shapes_backward_target_mask_independence_and_strict_reload(self):
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                runtime = self.build(model_id)
                batch = make_batch()
                output = runtime.adapter(runtime.model, batch)
                self.assertEqual(tuple(output.prediction.shape), (1, 1, 10))
                self.assertTrue(torch.isfinite(output.prediction).all())
                changed = copy.copy(batch)
                changed.target = torch.randn_like(batch.target) * 1000
                changed.mask = torch.zeros_like(batch.mask)
                second = runtime.adapter(runtime.model, changed)
                self.assertTrue(torch.equal(output.prediction, second.prediction))
                loss = masked_mse(output.prediction, batch.target, batch.mask)
                loss.backward()
                self.assertTrue(any(
                    parameter.grad is not None
                    for parameter in runtime.model.parameters()
                    if parameter.requires_grad
                ))
                incompatible = runtime.model.load_state_dict(
                    runtime.model.state_dict(), strict=True
                )
                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])

    def test_node_independence_and_parameter_count_does_not_depend_on_n(self):
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                runtime = self.build(model_id)
                single = make_batch(n=1)
                doubled = make_batch(n=2)
                doubled.x[:, :, 0] = single.x[:, :, 0]
                out_single = runtime.adapter(runtime.model, single).prediction[:, 0]
                out_double = runtime.adapter(runtime.model, doubled).prediction[:, 0]
                self.assertTrue(torch.allclose(out_single, out_double, atol=1e-6))
                count = runtime.parameter_count
                self.assertEqual(count, runtime.parameter_count)
                self.assertFalse(any(
                    "node_embedding" in name or "node_head" in name
                    for name, _ in runtime.model.named_parameters()
                ))

    def test_nonzero_future_observed_covariates_are_rejected(self):
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                runtime = self.build(model_id)
                with self.assertRaisesRegex(ContractError, "future"):
                    runtime.adapter(
                        runtime.model,
                        make_batch(),
                        future_observed_covariates=torch.ones(1, 10, 16),
                    )

    def test_timesnet_history_period_trace_and_top_k(self):
        runtime = self.build("timesnet")
        output = runtime.adapter(runtime.model, make_batch())
        trace = output.aux["period_trace"]
        self.assertEqual(trace["history_length"], 144)
        self.assertEqual(trace["configured_top_k"], 5)
        self.assertEqual(trace["frequency_bins"], 73)
        self.assertTrue(all(value > 0 for value in trace["example_history_periods"]))
        self.assertEqual(trace["period_source"], "BenchmarkBatch.x history only")

    def test_micn_original_decoder_mark_length_mismatch(self):
        model_class, _ = load_tslib_model_class("micn")
        model = model_class(historical_args("micn")).eval()
        x = torch.randn(2, 144, 16)
        with self.assertRaisesRegex(RuntimeError, "154.*58"):
            model(
                x,
                torch.zeros(2, 144, 4),
                torch.zeros(2, 58, 16),
                torch.zeros(2, 58, 4),
            )

    def test_micn_fixed_decoder_and_mark_lengths(self):
        runtime = self.build("micn")
        trace = runtime.adapter(
            runtime.model, make_batch()
        ).aux["decoder_trace"]
        self.assertEqual(trace["x_dec_shape"], (1, 154, 16))
        self.assertEqual(trace["x_mark_dec_shape"], (1, 154, 4))
        self.assertEqual(trace["seasonal_init_dec_length"], 154)
        self.assertTrue(trace["history_matches_observed_x"])
        self.assertTrue(trace["future_decoder_all_zero"])
        self.assertTrue(trace["decoder_marks_all_zero"])
        self.assertEqual(runtime.effective_config["seq_len"], 144)
        self.assertEqual(runtime.effective_config["pred_len"], 10)

    def test_multipatchformer_original_patch_concat_mismatch(self):
        model_class, _ = load_tslib_model_class("multipatchformer")
        model = model_class(historical_args("multipatchformer")).eval()
        with self.assertRaisesRegex(RuntimeError, "Expected size 18.*19"):
            model(torch.randn(2, 144, 16), None, None, None)

    def test_multipatchformer_all_distinct_branches_participate(self):
        runtime = self.build("multipatchformer")
        output = runtime.adapter(runtime.model, make_batch())
        geometry = output.aux["multiscale_geometry"]
        self.assertEqual(geometry["patch_lengths"], [8, 16, 24, 32])
        self.assertEqual(geometry["patch_counts"], [18, 18, 18, 18])
        self.assertFalse(geometry["silent_crop"])
        self.assertEqual(geometry["disabled_branches"], [])
        loss = output.prediction.square().mean()
        loss.backward()
        for index in range(1, 5):
            grad = getattr(runtime.model, f"embedding_patch_{index}").weight.grad
            self.assertIsNotNone(grad)
            self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_wpmixer_wavelet_branches_are_distinct_and_active(self):
        runtime = self.build("wpmixer")
        core = runtime.model.wpmixerCore
        self.assertEqual(core.input_w_dim, [73, 73])
        self.assertEqual(core.pred_w_dim, [6, 6])
        self.assertEqual(len(core.resolutionBranch), 2)
        self.assertFalse(core.no_decomposition)
        output = runtime.adapter(runtime.model, make_batch())
        output.prediction.square().mean().backward()
        for branch in core.resolutionBranch:
            self.assertIsNotNone(branch.patch_embedding_layer.weight.grad)

    def test_hardware_preflight_identity_is_exact_for_all_four(self):
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                identity = preflight_identity(model_id)
                for key, value in PREFLIGHT_SHAPE.items():
                    self.assertEqual(identity[key], value)
                self.assertTrue(identity["amp"])
                self.assertEqual(identity["model_id"], model_id)
                self.assertEqual(len(identity["source_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
