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
MODEL_IDS = ('timemixer', 'tsmixer', 'frets')

def make_batch(b=1, n=2, seed=2026):
    generator = torch.Generator().manual_seed(seed)
    return BenchmarkBatch(x=torch.randn(b, 144, n, 16, generator=generator), target=torch.randn(b, n, 10, generator=generator), mask=torch.ones(b, n, 10, dtype=torch.bool), sample_ids=list(range(b)), window_end_indices=list(range(b)), node_ids=list(range(n)), metadata={'contains_future_target': False})

class E2CModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.protocol = load_protocol()

    def build(self, model_id):
        runtime = build_model_runtime(model_id, self.protocol, run_mode='smoke')
        runtime.model.eval()
        return runtime

    def test_allowlist_is_explicit_and_exactly_eighteen(self):
        self.assertEqual(ALLOWED_TSLIB_MODELS, ('dlinear', 'lightts', 'tide', 'segrnn', 'transformer', 'patchtst', 'itransformer', 'timexer', 'timesnet', 'micn', 'wpmixer', 'multipatchformer', 'timemixer', 'tsmixer', 'frets', 'crossformer', 'msgnet', 'timefilter'))

    def test_shapes_backward_target_mask_test_target_independence_and_reload(self):
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                runtime = self.build(model_id)
                batch = make_batch(n=1)
                first = runtime.adapter(runtime.model, batch)
                self.assertEqual(tuple(first.prediction.shape), (1, 1, 10))
                self.assertEqual(first.raw_output_shape, (1, 10, 16))
                self.assertTrue(torch.isfinite(first.prediction).all())
                changed = copy.copy(batch)
                changed.target = torch.randn_like(batch.target) * 1000
                changed.mask = torch.zeros_like(batch.mask)
                second = runtime.adapter(runtime.model, changed, test_target=torch.randn_like(batch.target) * 1000)
                self.assertTrue(torch.equal(first.prediction, second.prediction))
                loss = masked_mse(first.prediction, batch.target, batch.mask)
                loss.backward()
                self.assertTrue(any((parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in runtime.model.parameters() if parameter.requires_grad)))
                incompatible = runtime.model.load_state_dict(runtime.model.state_dict(), strict=True)
                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])

    def test_node_isolation_parameter_sharing_and_no_node_parameters(self):
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                runtime = self.build(model_id)
                batch = make_batch(n=2)
                original = runtime.adapter(runtime.model, batch).prediction
                changed = copy.copy(batch)
                changed.x = batch.x.clone()
                changed.x[:, :, 0] += 1000
                modified = runtime.adapter(runtime.model, changed).prediction
                self.assertTrue(torch.allclose(original[:, 1], modified[:, 1], atol=1e-06))
                self.assertFalse(torch.equal(original[:, 0], modified[:, 0]))
                count = runtime.parameter_count
                second_runtime = self.build(model_id)
                self.assertEqual(count, second_runtime.parameter_count)
                self.assertFalse(any(('node_embedding' in name or 'node_head' in name for name, _ in runtime.model.named_parameters())))

    def test_all_nonzero_future_inputs_and_marks_fail_closed(self):
        for model_id in MODEL_IDS:
            runtime = self.build(model_id)
            for key in ('future_observed_covariates', 'future_exogenous', 'future_weather', 'future_calendar_marks'):
                with self.subTest(model_id=model_id, key=key):
                    with self.assertRaisesRegex(ContractError, 'rejects non-zero'):
                        runtime.adapter(runtime.model, make_batch(n=1), **{key: torch.ones(1, 10, 16)})
            marked = make_batch(n=1)
            marked.x_mark = torch.ones(1, 144, 1, 4)
            with self.assertRaisesRegex(ContractError, 'calendar/time marks'):
                runtime.adapter(runtime.model, marked)

    def test_timemixer_all_scales_pdm_season_trend_and_predictors_get_gradients(self):
        runtime = self.build('timemixer')
        output = runtime.adapter(runtime.model, make_batch(n=1))
        trace = output.aux['scale_trace']
        self.assertEqual(trace['history_lengths'], [144, 72, 36, 18])
        self.assertEqual(trace['disabled_scales'], [])
        output.prediction.square().mean().backward()
        for predictor in runtime.model.predict_layers:
            self.assertGreater(float(predictor.weight.grad.abs().sum()), 0.0)
        for block in runtime.model.pdm_blocks:
            season = block.mixing_multi_scale_season.down_sampling_layers
            trend = block.mixing_multi_scale_trend.up_sampling_layers
            self.assertTrue(all((layer[0].weight.grad is not None and float(layer[0].weight.grad.abs().sum()) > 0 for layer in season)))
            self.assertTrue(all((layer[0].weight.grad is not None and float(layer[0].weight.grad.abs().sum()) > 0 for layer in trend)))

    def test_tsmixer_temporal_channel_and_projection_gradients(self):
        runtime = self.build('tsmixer')
        output = runtime.adapter(runtime.model, make_batch(n=1))
        self.assertEqual(output.aux['mixer_trace']['layers'], 2)
        output.prediction.square().mean().backward()
        for block in runtime.model.model:
            self.assertGreater(float(block.temporal[0].weight.grad.abs().sum()), 0)
            self.assertGreater(float(block.channel[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(runtime.model.projection.weight.grad.abs().sum()), 0)

    def test_frets_channel_independence_type_matrix(self):
        model_class, _ = load_tslib_model_class('frets')
        base = {'task_name': 'long_term_forecast', 'seq_len': 144, 'pred_len': 10, 'enc_in': 16}
        x = torch.randn(1, 144, 16)
        expected = {0: False, 1: False, '0': True, '1': False}
        for value, channel_active in expected.items():
            with self.subTest(value=repr(value)):
                model = model_class(SimpleNamespace(**base, channel_independence=value))
                output = model(x, None, None, None)
                self.assertEqual(tuple(output.shape), (1, 10, 16))
                output.square().mean().backward()
                self.assertEqual(model.r1.grad is not None, channel_active)
                self.assertIsNotNone(model.r2.grad)
                self.assertGreater(float(model.r2.grad.abs().sum()), 0)
                if channel_active:
                    self.assertGreater(float(model.r1.grad.abs().sum()), 0)

    def test_frets_full_frequency_identity_and_native_amp_policy(self):
        runtime = self.build('frets')
        self.assertEqual(runtime.model.channel_independence, '0')
        self.assertEqual(runtime.model.embed_size, 128)
        self.assertEqual(runtime.model.hidden_size, 256)
        output = runtime.adapter(runtime.model, make_batch(n=1))
        self.assertEqual(output.prediction.dtype, torch.float32)
        trace = output.aux['frequency_trace']
        self.assertTrue(trace['channel_frequency_branch'])
        self.assertTrue(trace['temporal_frequency_branch'])
        output.prediction.square().mean().backward()
        for name in ('embeddings', 'r1', 'i1', 'rb1', 'ib1', 'r2', 'i2', 'rb2', 'ib2'):
            grad = getattr(runtime.model, name).grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.abs().sum()), 0.0, name)
        self.assertGreater(float(runtime.model.fc[0].weight.grad.abs().sum()), 0)

    def test_dynamic_power_channel_selection_is_not_average_or_first(self):
        for model_id in MODEL_IDS:
            runtime = self.build(model_id)
            batch = make_batch(n=1)
            flat = runtime.adapter.prepare_model_inputs(batch)
            raw = runtime.adapter.forward_model(runtime.model, flat)
            output = runtime.adapter.normalize_output(raw, batch)
            self.assertEqual(runtime.adapter.power_feature, 'Patv_clean_for_input')
            self.assertEqual(runtime.adapter.power_feature_index, 15)
            self.assertTrue(torch.equal(output.prediction, raw[..., runtime.adapter.power_feature_index].reshape(1, 1, 10)))
            self.assertFalse(torch.equal(output.prediction, raw[..., 0].reshape(1, 1, 10)))

    def test_final_registry_hardware_routing(self):
        registry = load_registry()
        tsmixer = registry.get('tsmixer')
        self.assertEqual(tsmixer.runtime_status, 'AVAILABLE_TRAINABLE')
        self.assertFalse(tsmixer.formal_hardware_preflight_required)
        self.assertEqual(tsmixer.local_full_shape_status, 'PASS')
        for model_id in ('timemixer', 'frets'):
            entry = registry.get(model_id)
            self.assertEqual(entry.runtime_status, 'AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED')
            self.assertTrue(entry.formal_hardware_preflight_required)
            self.assertEqual(entry.local_full_shape_status, 'FAIL_OOM')
            self.assertEqual(entry.non_oom_engineering_checks, 'PASS')
if __name__ == '__main__':
    unittest.main()
