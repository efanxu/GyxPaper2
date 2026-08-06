from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / 'custom_models' / 'src'
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from benchmark_v2.artifacts import ArtifactError
from benchmark_v2.checkpointing import CheckpointManager
from benchmark_v2.contracts import BenchmarkBatch
from benchmark_v2.errors import ContractError
from benchmark_v2.graph import GraphProtocolError, load_graph_bundle
from benchmark_v2.hardware_preflight import preflight_artifact_path, preflight_identity, read_matching_pass
from benchmark_v2.model_runtime import build_model_runtime
from benchmark_v2.models.graph_models import ChebyshevGraphConv, DCGRUCell, DiffusionLinear, GCNLayer, TemporalGLUConv, chebyshev_basis
from benchmark_v2.protocol import load_protocol
from benchmark_v2.registry import load_registry
from benchmark_v2.upstream.tslib_loader import ALLOWED_TSLIB_MODELS
MODELS = ('gcn', 'stgcn', 'dcrnn')

def graph_batch(batch_size: int=1, *, seed: int=2026) -> BenchmarkBatch:
    generator = torch.Generator().manual_seed(seed)
    return BenchmarkBatch(x=torch.randn(batch_size, 144, 134, 16, generator=generator), target=torch.randn(batch_size, 134, 10, generator=generator), mask=torch.ones(batch_size, 134, 10, dtype=torch.bool), sample_ids=list(range(batch_size)), window_end_indices=list(range(batch_size)), node_ids=list(range(1, 135)), metadata={'contains_future_target': False})

def assert_parameter_gradients(test: unittest.TestCase, model) -> None:
    missing = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or float(parameter.grad.abs().sum()) == 0.0:
            missing.append(name)
    test.assertEqual(missing, [], f'Missing/non-finite/zero gradients: {missing}')

class TestRegistryAndIdentity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.bundle = load_graph_bundle()

    def test_registry_is_28_with_single_graph_canonical_entries(self):
        registry = load_registry()
        self.assertEqual(len(registry.list()), 28)
        for model_id in MODELS:
            matches = [value for value in registry.list() if value.canonical_id == model_id]
            self.assertEqual(len(matches), 1)
            self.assertTrue(matches[0].supports_train)
        self.assertEqual(registry.get('stcn_stgcn_unresolved').canonical_id, 'stgcn')
        self.assertNotIn('STCN', [value.canonical_id for value in registry.list()])
        self.assertEqual(len(ALLOWED_TSLIB_MODELS), 18)
        with self.assertRaises(KeyError):
            registry.get('unknown_graph_model')

    def test_supports_are_float32_nonpersistent_and_not_trainable(self):
        expected_buffers = {'gcn': ('A_gcn',), 'stgcn': ('L_tilde', 'chebyshev_support'), 'dcrnn': ('P_forward', 'P_reverse')}
        for model_id in MODELS:
            runtime = build_model_runtime(model_id, self.protocol, run_mode='smoke')
            state = runtime.model.state_dict()
            parameter_ids = {id(value) for value in runtime.model.parameters()}
            for name in expected_buffers[model_id]:
                support = getattr(runtime.model, name)
                self.assertEqual(support.dtype, torch.float32)
                self.assertFalse(support.requires_grad)
                self.assertNotIn(id(support), parameter_ids)
                self.assertNotIn(name, state)

class TestGCN(unittest.TestCase):

    def test_manual_a_x_w_formula(self):
        layer = GCNLayer(2, 1)
        with torch.no_grad():
            layer.linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
            layer.linear.bias.copy_(torch.tensor([0.5]))
        support = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.25, 0.75]])
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        expected = support @ x[0] @ layer.linear.weight.T + 0.5
        self.assertTrue(torch.allclose(layer(x, support)[0], expected))

    def test_full_path_graph_sensitivity_and_gradients(self):
        runtime = build_model_runtime('gcn', load_protocol(), run_mode='smoke')
        model = runtime.model
        batch = graph_batch()
        prediction = runtime.adapter(model, batch).prediction
        self.assertEqual(tuple(prediction.shape), (1, 134, 10))
        prediction.square().mean().backward()
        assert_parameter_gradients(self, model)
        self.assertEqual(model.last_shape_trace['graph_layer_1'], [1, 144, 134, 64])
        self.assertEqual(model.last_shape_trace['graph_layer_2'], [1, 144, 134, 64])
        model.eval()
        with torch.no_grad():
            real_output = model(batch.x)
            original = model.A_gcn
            model.A_gcn = torch.eye(134)
            identity_output = model(batch.x)
            model.A_gcn = original
        self.assertFalse(torch.allclose(real_output, identity_output))

class TestSTGCN(unittest.TestCase):

    def test_chebyshev_recurrence_and_feature_aggregation(self):
        support = torch.tensor([[0.0, 0.5], [0.5, 0.0]])
        basis = chebyshev_basis(support, 3)
        self.assertTrue(torch.allclose(basis[0], torch.eye(2)))
        self.assertTrue(torch.allclose(basis[1], support))
        self.assertTrue(torch.allclose(basis[2], 2.0 * support @ support - torch.eye(2)))
        layer = ChebyshevGraphConv(1, 1, 3)
        with torch.no_grad():
            layer.weight.fill_(1.0)
            layer.bias.zero_()
        x = torch.tensor([[[[1.0], [2.0]]]])
        expected = sum((term @ x[0, 0] for term in basis))
        self.assertTrue(torch.allclose(layer(x, basis)[0, 0], expected))

    def test_glu_filter_and_gate_have_gradients(self):
        layer = TemporalGLUConv(2, 3, 3)
        x = torch.randn(2, 8, 4, 2, requires_grad=True)
        output = layer(x)
        self.assertEqual(tuple(output.shape), (2, 6, 4, 3))
        output.square().mean().backward()
        gradient = layer.convolution.weight.grad
        self.assertGreater(float(gradient[:3].abs().sum()), 0.0)
        self.assertGreater(float(gradient[3:].abs().sum()), 0.0)

    def test_classic_two_block_trace_and_gradients(self):
        runtime = build_model_runtime('stgcn', load_protocol(), run_mode='smoke')
        batch = graph_batch()
        prediction = runtime.adapter(runtime.model, batch).prediction
        self.assertEqual(tuple(prediction.shape), (1, 134, 10))
        prediction.square().mean().backward()
        assert_parameter_gradients(self, runtime.model)
        trace = runtime.model.last_shape_trace
        self.assertEqual(trace['block1']['temporal1'][1], 142)
        self.assertEqual(trace['block1']['temporal2'][1], 140)
        self.assertEqual(trace['block2']['temporal1'][1], 138)
        self.assertEqual(trace['block2']['temporal2'][1], 136)
        self.assertEqual(tuple(runtime.model.chebyshev_support.shape), (3, 134, 134))

class TestDCRNN(unittest.TestCase):

    def test_directed_diffusion_basis_order_and_identity_once(self):
        forward = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        reverse = forward.T
        x = torch.tensor([[[1.0], [2.0], [4.0]]])
        layer = DiffusionLinear(1, 1, max_diffusion_step=2)
        basis = layer.diffusion_basis(x, (forward, reverse))
        expected = torch.stack([x, torch.einsum('nm,bmf->bnf', forward, x), torch.einsum('nm,bmf->bnf', forward @ forward, x), torch.einsum('nm,bmf->bnf', reverse, x), torch.einsum('nm,bmf->bnf', reverse @ reverse, x)], dim=2)
        self.assertEqual(tuple(basis.shape), (1, 3, 5, 1))
        self.assertTrue(torch.equal(basis, expected))

    def test_dcgru_all_gate_diffusions_have_gradients(self):
        cell = DCGRUCell(2, 4, max_diffusion_step=2)
        supports = (torch.eye(3), torch.eye(3).roll(1, dims=0))
        x = torch.randn(2, 3, 2)
        hidden = torch.randn(2, 3, 4)
        cell(x, hidden, supports).square().mean().backward()
        assert_parameter_gradients(self, cell)

    def test_encoder_decoder_autoregression_and_gradients(self):
        runtime = build_model_runtime('dcrnn', load_protocol(), run_mode='smoke')
        batch = graph_batch()
        prediction = runtime.adapter(runtime.model, batch).prediction
        self.assertEqual(tuple(prediction.shape), (1, 134, 10))
        prediction.square().mean().backward()
        assert_parameter_gradients(self, runtime.model)
        trace = runtime.model.last_shape_trace
        self.assertEqual(trace['encoder_state'], [2, 1, 134, 64])
        self.assertEqual(trace['go_token'], [1, 134, 1])
        self.assertEqual(trace['decoder_steps'], 10)
        self.assertFalse(runtime.model.teacher_forcing)
        self.assertFalse(runtime.model.scheduled_sampling)
        self.assertFalse(runtime.model.curriculum_learning)
        runtime.model.eval()
        with torch.no_grad():
            dual = runtime.model(batch.x)
            forward, reverse = (runtime.model.P_forward, runtime.model.P_reverse)
            runtime.model.P_forward = torch.zeros_like(forward)
            runtime.model.P_reverse = torch.zeros_like(reverse)
            identity_only = runtime.model(batch.x)
            runtime.model.P_forward, runtime.model.P_reverse = (forward, reverse)
        self.assertFalse(torch.allclose(dual, identity_only))

class TestIsolationLeakageAndReload(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_target_mask_invariance_and_future_fail_closed(self):
        for model_id in MODELS:
            runtime = build_model_runtime(model_id, self.protocol, run_mode='smoke')
            runtime.model.eval()
            first = graph_batch()
            changed = graph_batch()
            changed.x = first.x.clone()
            changed.target.fill_(12345.0)
            changed.mask.zero_()
            with torch.no_grad():
                output_a = runtime.adapter(runtime.model, first).prediction
                output_b = runtime.adapter(runtime.model, changed).prediction
            self.assertTrue(torch.equal(output_a, output_b), f'{model_id} target leakage')
            for key in ('future_observed_covariates', 'future_weather', 'future_calendar_marks'):
                with self.subTest(model=model_id, key=key), self.assertRaises(ContractError):
                    runtime.adapter.prepare_model_inputs(first, **{key: torch.ones(1)})

    def test_batch_companion_permutation_and_single_sample_equivalence(self):
        batch = graph_batch(2)
        for model_id in MODELS:
            runtime = build_model_runtime(model_id, self.protocol, run_mode='smoke')
            runtime.model.eval()
            single = BenchmarkBatch(x=batch.x[:1], target=batch.target[:1], mask=batch.mask[:1], sample_ids=[0], window_end_indices=[0], node_ids=batch.node_ids, metadata=dict(batch.metadata))
            permuted = BenchmarkBatch(x=batch.x.flip(0), target=batch.target.flip(0), mask=batch.mask.flip(0), sample_ids=[1, 0], window_end_indices=[1, 0], node_ids=batch.node_ids, metadata=dict(batch.metadata))
            with torch.no_grad():
                together = runtime.adapter(runtime.model, batch).prediction
                alone = runtime.adapter(runtime.model, single).prediction
                swapped = runtime.adapter(runtime.model, permuted).prediction
            self.assertTrue(torch.allclose(together[:1], alone, atol=1e-06))
            self.assertTrue(torch.allclose(together, swapped.flip(0), atol=1e-06))

    def test_synthetic_node_permutation_equivariance(self):
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(permutation)
        support = torch.tensor([[1.0, 0.5, 0.0, 0.0], [0.5, 1.0, 0.25, 0.0], [0.0, 0.25, 1.0, 0.5], [0.0, 0.0, 0.5, 1.0]])
        x = torch.randn(2, 4, 3)
        permuted_support = support[permutation][:, permutation]
        permuted_x = x[:, permutation]
        layer = GCNLayer(3, 2)
        original = layer(x, support)
        permuted = layer(permuted_x, permuted_support)
        self.assertTrue(torch.allclose(original, permuted[:, inverse], atol=1e-06))
        diffusion = DiffusionLinear(3, 2, max_diffusion_step=2)
        original_d = diffusion(x, (support, support.T))
        permuted_d = diffusion(permuted_x, (permuted_support, permuted_support.T))
        self.assertTrue(torch.allclose(original_d, permuted_d[:, inverse], atol=1e-06))
if __name__ == '__main__':
    unittest.main()
