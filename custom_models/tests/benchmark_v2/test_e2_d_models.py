import copy
import json
import tempfile
import unittest
from types import SimpleNamespace
import torch
from benchmark_v2.contracts import BenchmarkBatch
from benchmark_v2.errors import ContractError
from benchmark_v2.hardware_preflight import PREFLIGHT_SHAPE, launch_formal_train, preflight_artifact_path, preflight_identity
from benchmark_v2.losses import masked_mse
from benchmark_v2.model_runtime import build_model_runtime
from benchmark_v2.protocol import load_protocol
from benchmark_v2.registry import load_registry
from benchmark_v2.upstream import ALLOWED_TSLIB_MODELS, resolve_tslib_source
MODEL_IDS = ('crossformer', 'msgnet', 'timefilter')

def make_batch(b=2, n=2, seed=2026):
    generator = torch.Generator().manual_seed(seed)
    return BenchmarkBatch(x=torch.randn(b, 144, n, 16, generator=generator), target=torch.randn(b, n, 10, generator=generator), mask=torch.ones(b, n, 10, dtype=torch.bool), sample_ids=list(range(b)), window_end_indices=list(range(b)), node_ids=list(range(n)), metadata={'contains_future_target': False})

def subset_batch(batch, indices):
    return BenchmarkBatch(x=batch.x[indices], target=batch.target[indices], mask=batch.mask[indices], sample_ids=[batch.sample_ids[i] for i in indices], window_end_indices=[batch.window_end_indices[i] for i in indices], node_ids=batch.node_ids, metadata=dict(batch.metadata))

class E2DModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.protocol = load_protocol()

    def build(self, model_id, train=False):
        runtime = build_model_runtime(model_id, self.protocol, run_mode='smoke')
        runtime.model.train(train)
        return runtime

    def test_shape_backward_reload_and_target_mask_independence(self):
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                runtime = self.build(model_id)
                batch = make_batch(b=1, n=1)
                first = runtime.adapter(runtime.model, batch)
                self.assertEqual(tuple(first.prediction.shape), (1, 1, 10))
                self.assertEqual(first.raw_output_shape, (1, 10, 16))
                changed = copy.copy(batch)
                changed.target = torch.randn_like(batch.target) * 1000
                changed.mask = torch.zeros_like(batch.mask)
                second = runtime.adapter(runtime.model, changed, test_target=torch.randn_like(batch.target) * 1000)
                self.assertTrue(torch.equal(first.prediction, second.prediction))
                loss = masked_mse(first.prediction, batch.target, batch.mask)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(any((p.grad is not None and torch.isfinite(p.grad).all() and (float(p.grad.abs().sum()) > 0) for p in runtime.model.parameters() if p.requires_grad)))
                incompatible = runtime.model.load_state_dict(runtime.model.state_dict(), strict=True)
                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])

    def test_batch_companion_permutation_and_single_item_invariance(self):
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                runtime = self.build(model_id)
                ab = make_batch(b=2, n=1, seed=1)
                ac = make_batch(b=2, n=1, seed=2)
                ac.x[0] = ab.x[0]
                ac.target[0] = ab.target[0]
                ac.mask[0] = ab.mask[0]
                out_ab = runtime.adapter(runtime.model, ab).prediction
                out_ac = runtime.adapter(runtime.model, ac).prediction
                single = runtime.adapter(runtime.model, subset_batch(ab, [0])).prediction
                permuted = runtime.adapter(runtime.model, subset_batch(ab, [1, 0])).prediction
                self.assertTrue(torch.allclose(out_ab[0], out_ac[0], atol=1e-06))
                self.assertTrue(torch.allclose(out_ab[0], single[0], atol=1e-06))
                self.assertTrue(torch.allclose(out_ab[0], permuted[1], atol=1e-06))

    def test_cross_node_isolation_and_parameter_count_independent_of_n(self):
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                runtime = self.build(model_id)
                batch = make_batch(b=1, n=2)
                original = runtime.adapter(runtime.model, batch).prediction
                changed = copy.copy(batch)
                changed.x = batch.x.clone()
                changed.x[:, :, 0] += 1000
                modified = runtime.adapter(runtime.model, changed).prediction
                self.assertTrue(torch.allclose(original[:, 1], modified[:, 1], atol=1e-06))
                self.assertFalse(torch.equal(original[:, 0], modified[:, 0]))
                self.assertEqual(runtime.parameter_count, self.build(model_id).parameter_count)
                self.assertFalse(any(('node_embedding' in name or 'node_head' in name for name, _ in runtime.model.named_parameters())))

    def test_all_future_inputs_and_marks_fail_closed(self):
        for model_id in MODEL_IDS:
            runtime = self.build(model_id)
            for key in ('future_observed_covariates', 'future_exogenous', 'future_weather', 'future_calendar_marks'):
                with self.subTest(model_id=model_id, key=key):
                    with self.assertRaisesRegex(ContractError, 'rejects non-zero'):
                        runtime.adapter(runtime.model, make_batch(b=1, n=1), **{key: torch.ones(1, 10, 16)})
            marked = make_batch(b=1, n=1)
            marked.x_mark = torch.ones(1, 144, 1, 4)
            with self.assertRaisesRegex(ContractError, 'calendar/time marks'):
                runtime.adapter(runtime.model, marked)

    def test_crossformer_geometry_and_attention_gradients(self):
        runtime = self.build('crossformer')
        output = runtime.adapter(runtime.model, make_batch(b=1, n=1))
        trace = output.aux['shape_trace']
        self.assertEqual(runtime.model.seg_len, 12)
        self.assertEqual(runtime.model.win_size, 2)
        self.assertEqual(runtime.model.in_seg_num, 12)
        self.assertEqual(runtime.model.pad_out_len, 12)
        self.assertFalse(trace['adapter_horizon_crop'])
        output.prediction.square().mean().backward()
        named = dict(runtime.model.named_parameters())
        groups = ('enc_value_embedding.value_embedding.weight', 'router', 'time_attention.query_projection.weight', 'dim_sender.query_projection.weight', 'merge_layer.linear_trans.weight', 'self_attention.router', 'cross_attention.query_projection.weight', 'linear_pred.weight', 'dec_pos_embedding')
        for token in groups:
            matches = [p for name, p in named.items() if token in name]
            self.assertTrue(matches, token)
            self.assertTrue(any((p.grad is not None and torch.isfinite(p.grad).all() and (float(p.grad.abs().sum()) > 0) for p in matches)), token)

    def test_msgnet_period_graph_fft_and_all_scale_gradients(self):
        runtime = self.build('msgnet')
        output = runtime.adapter(runtime.model, make_batch(b=1, n=1))
        trace = output.aux['period_graph_trace']
        self.assertEqual(trace['adaptive_adjacency_shape'], [16, 16])
        self.assertEqual(trace['fft_batch_scope'], 1)
        self.assertEqual(trace['fft_calls'], 2)
        self.assertTrue(all((p > 0 for p in trace['periods'])))
        self.assertTrue(all((row['fft_dtype'] == 'torch.complex64' for row in trace['traces'])))
        output.prediction.square().mean().backward()
        for layer in runtime.model.model:
            for branch in layer.gconv:
                for parameter in (branch.nodevec1, branch.nodevec2):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_timefilter_masks_determinism_reproducibility_and_gradients(self):
        runtime = self.build('timefilter')
        batch = make_batch(b=1, n=1)
        first = runtime.adapter(runtime.model, batch)
        second = runtime.adapter(runtime.model, batch)
        self.assertTrue(torch.equal(first.prediction, second.prediction))
        trace = first.aux['token_mask_trace']
        self.assertEqual(trace['patch_len'], 16)
        self.assertEqual(trace['token_count'], 144)
        self.assertEqual(trace['mask_shape'], [144, 3, 144])
        self.assertFalse(trace['moe_auxiliary_loss_used_for_masked_mse'])
        first.prediction.square().mean().backward()
        for block in runtime.model.backbone.blocks:
            gate = block.gnn.graph_learner.mask_moe.gate.weight
            graph = block.gnn.graph_conv.proj.weight
            for parameter in (gate, graph):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
        seeded = self.build('timefilter', train=True)
        torch.manual_seed(77)
        train_first = seeded.adapter(seeded.model, batch).prediction
        torch.manual_seed(77)
        train_second = seeded.adapter(seeded.model, batch).prediction
        self.assertTrue(torch.equal(train_first, train_second))

    def test_final_hardware_routing(self):
        registry = load_registry()
        crossformer = registry.get('crossformer')
        self.assertEqual(crossformer.runtime_status, 'AVAILABLE_TRAINABLE')
        self.assertFalse(crossformer.formal_hardware_preflight_required)
        self.assertEqual(crossformer.local_full_shape_status, 'PASS')
        for model_id in ('msgnet', 'timefilter'):
            entry = registry.get(model_id)
            self.assertEqual(entry.runtime_status, 'AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED')
            self.assertTrue(entry.formal_hardware_preflight_required)
            self.assertEqual(entry.local_full_shape_status, 'FAIL_OOM')
            self.assertEqual(entry.non_oom_engineering_checks, 'PASS')

    def test_new_model_dummy_preflight_process_routing(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []

            def runner(argv, check=False):
                calls.append(tuple(argv))
                if '_hardware-preflight-worker' in argv:
                    model_id = argv[argv.index('--model') + 1]
                    path = preflight_artifact_path(model_id, root=root)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    identity = preflight_identity(model_id)
                    path.write_text(json.dumps({**identity, 'status': 'PASS', 'forward_pass': True, 'backward_pass': True, 'finite': True, 'output_shape': [identity['batch_size'], identity['node_count'], identity['horizon']], 'preflight_pid': 111}), encoding='utf-8')
                    return SimpleNamespace(returncode=0, pid=111)
                return SimpleNamespace(returncode=0, pid=222)
            for model_id in ('msgnet', 'timefilter'):
                calls.clear()
                self.assertEqual(launch_formal_train(model_id, input_path='input.parquet', target_path='target.parquet', output_root='formal-root', run_id=f'{model_id}-run', device='cuda', preflight_root=root, runner=runner), 0)
                self.assertEqual(len(calls), 2)
                self.assertIn('_hardware-preflight-worker', calls[0])
                self.assertIn('_formal-train-worker', calls[1])
                self.assertNotEqual(111, 222)

    def test_crossformer_skips_mandatory_preflight_and_failures_start_no_formal(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            runner = lambda argv, check=False: calls.append(tuple(argv)) or SimpleNamespace(returncode=0)
            self.assertEqual(launch_formal_train('crossformer', input_path='input.parquet', target_path='target.parquet', output_root='formal-root', run_id='crossformer-run', device='cuda', preflight_root=root, runner=runner), 0)
            self.assertEqual(len(calls), 1)
            self.assertIn('_formal-train-worker', calls[0])
            calls.clear()
            failing = lambda argv, check=False: calls.append(tuple(argv)) or SimpleNamespace(returncode=3)
            self.assertEqual(launch_formal_train('msgnet', input_path='input.parquet', target_path='target.parquet', output_root='formal-root', run_id='msgnet-run', device='cuda', preflight_root=root, runner=failing), 3)
            self.assertEqual(len(calls), 1)
            self.assertNotIn('_formal-train-worker', calls[0])
if __name__ == '__main__':
    unittest.main()
