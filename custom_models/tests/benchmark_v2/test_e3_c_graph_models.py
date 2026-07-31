import copy
import json
import tempfile
import unittest

import torch

from benchmark_v2.checkpointing import CheckpointManager
from benchmark_v2.configs import (
    resolve_agcrn_config,
    resolve_graph_wavenet_config,
    resolve_mtgnn_config,
    resolve_stid_config,
)
from benchmark_v2.contracts import BenchmarkBatch
from benchmark_v2.errors import ArtifactError, ContractError
from benchmark_v2.graph import GraphProtocolError
from benchmark_v2.hardware_preflight import (
    preflight_artifact_path,
    preflight_identity,
    read_matching_pass,
)
from benchmark_v2.model_runtime import build_model_runtime
from benchmark_v2.models.graph_models import (
    AdaptiveGraphConv,
    DilatedInception,
    DirectedGraphConstructor,
    canonical_tensor_hash,
)
from benchmark_v2.models.graph_models.adaptive_common import (
    E3_C_POLICIES,
)
from benchmark_v2.protocol import load_protocol
from benchmark_v2.registry import load_registry
from benchmark_v2.upstream.tslib_loader import ALLOWED_TSLIB_MODELS


MODELS = ("graph_wavenet", "mtgnn", "agcrn", "stid")


def node_batch(batch_size=1, seed=2026):
    generator = torch.Generator().manual_seed(seed)
    return BenchmarkBatch(
        x=torch.randn(batch_size, 144, 134, 16, generator=generator),
        target=torch.randn(batch_size, 134, 10, generator=generator),
        mask=torch.ones(batch_size, 134, 10, dtype=torch.bool),
        sample_ids=list(range(batch_size)),
        window_end_indices=list(range(batch_size)),
        node_ids=list(range(1, 135)),
        metadata={
            "contains_future_target": False,
            "history_end_timestamp": [
                "2021-01-04 12:00:00" for _ in range(batch_size)
            ],
            "history_end_time_of_day_id": [72 for _ in range(batch_size)],
            "history_end_day_of_week_id": [0 for _ in range(batch_size)],
            "timezone_policy": "SOURCE_NAIVE_UNCHANGED",
        },
    )


def clone_batch(batch):
    return BenchmarkBatch(
        x=batch.x.clone(),
        target=batch.target.clone(),
        mask=batch.mask.clone(),
        sample_ids=list(batch.sample_ids),
        window_end_indices=list(batch.window_end_indices),
        node_ids=list(batch.node_ids),
        split=batch.split,
        metadata=copy.deepcopy(batch.metadata),
    )


def assert_finite_nonzero(testcase, parameter, name):
    testcase.assertIsNotNone(parameter.grad, name)
    testcase.assertTrue(torch.isfinite(parameter.grad).all(), name)
    testcase.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)


class TestE3CRegistryConfigAndPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_registry_count_unique_ids_and_allowlist(self):
        registry = load_registry()
        self.assertEqual(len(registry.list()), 28)
        self.assertEqual(len(ALLOWED_TSLIB_MODELS), 18)
        for model_id in MODELS:
            matches = [
                entry
                for entry in registry.list()
                if entry.canonical_id == model_id
            ]
            self.assertEqual(len(matches), 1)
        with self.assertRaises(KeyError):
            registry.get("unknown_e3_c_model")

    def test_configs_are_frozen_before_smoke(self):
        gwn = resolve_graph_wavenet_config(self.protocol, run_mode="smoke")
        self.assertEqual(
            (
                gwn["residual_channels"],
                gwn["skip_channels"],
                gwn["blocks"],
                gwn["layers_per_block"],
                gwn["diffusion_order"],
            ),
            (32, 256, 4, 2, 2),
        )
        mtg = resolve_mtgnn_config(self.protocol, run_mode="smoke")
        self.assertEqual(
            (mtg["subgraph_size"], mtg["node_dim"], mtg["layers"]),
            (20, 40, 3),
        )
        self.assertEqual(mtg["num_split"], 1)
        self.assertFalse(mtg["node_sampling"])
        agc = resolve_agcrn_config(self.protocol, run_mode="smoke")
        self.assertEqual(
            (agc["embed_dim"], agc["rnn_units"], agc["num_layers"], agc["cheb_order"]),
            (10, 64, 2, 2),
        )
        stid = resolve_stid_config(self.protocol, run_mode="smoke")
        self.assertEqual(
            (
                stid["embed_dim"],
                stid["node_dim"],
                stid["time_of_day_dim"],
                stid["day_of_week_dim"],
                stid["num_layers"],
            ),
            (32, 32, 32, 32, 3),
        )

    def test_physical_adaptive_and_identity_policies(self):
        expected = {
            "graph_wavenet": (True, ["P_forward", "P_reverse"]),
            "mtgnn": (False, []),
            "agcrn": (False, []),
            "stid": (False, []),
        }
        for model_id, (uses_physical, names) in expected.items():
            runtime = build_model_runtime(
                model_id, self.protocol, run_mode="smoke"
            )
            identity = runtime.model.graph_identity
            self.assertEqual(identity["uses_physical_support"], uses_physical)
            self.assertEqual(identity["physical_support_names"], names)
            self.assertEqual(
                identity["adaptive_graph_policy"],
                E3_C_POLICIES[model_id]["adaptive_graph_policy"],
            )
        stid_model = build_model_runtime(
            "stid", self.protocol, run_mode="smoke"
        ).model
        self.assertFalse(
            any(
                "graph" in type(module).__name__.casefold()
                for module in stid_model.modules()
            )
        )


class TestNativeNodeIdentityAndLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_node_order_and_same_count_wrong_identity_fail_closed(self):
        for model_id in MODELS:
            runtime = build_model_runtime(
                model_id, self.protocol, run_mode="smoke"
            )
            good = node_batch()
            runtime.adapter.prepare_model_inputs(good)
            swapped = clone_batch(good)
            swapped.node_ids[0], swapped.node_ids[1] = (
                swapped.node_ids[1],
                swapped.node_ids[0],
            )
            with self.subTest(model=model_id), self.assertRaises(
                GraphProtocolError
            ):
                runtime.adapter.prepare_model_inputs(swapped)
            wrong_hash = clone_batch(good)
            wrong_hash.metadata["node_order_hash"] = "wrong"
            with self.assertRaises(GraphProtocolError):
                runtime.adapter.prepare_model_inputs(wrong_hash)

    def test_target_mask_invariance_and_future_inputs_rejected(self):
        original = node_batch()
        for model_id in MODELS:
            runtime = build_model_runtime(
                model_id, self.protocol, run_mode="smoke"
            )
            runtime.model.eval()
            changed = clone_batch(original)
            changed.target.fill_(99999.0)
            changed.mask.zero_()
            with torch.no_grad():
                a = runtime.adapter(runtime.model, original).prediction
                b = runtime.adapter(runtime.model, changed).prediction
            self.assertTrue(torch.equal(a, b), model_id)
            for key in (
                "future_observed_covariates",
                "future_weather",
                "future_calendar_marks",
                "future_timestamps",
            ):
                with self.subTest(model=model_id, key=key), self.assertRaises(
                    ContractError
                ):
                    runtime.adapter.prepare_model_inputs(
                        original, **{key: torch.ones(1)}
                    )

    def test_batch_companion_permutation_and_single_equivalence(self):
        batch = node_batch(2)
        for model_id in MODELS:
            runtime = build_model_runtime(
                model_id, self.protocol, run_mode="smoke"
            )
            runtime.model.eval()
            single = clone_batch(batch)
            single.x = single.x[:1]
            single.target = single.target[:1]
            single.mask = single.mask[:1]
            single.sample_ids = [0]
            single.window_end_indices = [0]
            for key in (
                "history_end_timestamp",
                "history_end_time_of_day_id",
                "history_end_day_of_week_id",
            ):
                single.metadata[key] = single.metadata[key][:1]
            permuted = clone_batch(batch)
            permuted.x = permuted.x.flip(0)
            permuted.target = permuted.target.flip(0)
            permuted.mask = permuted.mask.flip(0)
            permuted.sample_ids.reverse()
            permuted.window_end_indices.reverse()
            for key in (
                "history_end_timestamp",
                "history_end_time_of_day_id",
                "history_end_day_of_week_id",
            ):
                permuted.metadata[key] = list(reversed(permuted.metadata[key]))
            with torch.no_grad():
                together = runtime.adapter(runtime.model, batch).prediction
                alone = runtime.adapter(runtime.model, single).prediction
                swapped = runtime.adapter(runtime.model, permuted).prediction
            self.assertTrue(
                torch.allclose(together[:1], alone, atol=1e-6), model_id
            )
            self.assertTrue(
                torch.allclose(together, swapped.flip(0), atol=1e-6), model_id
            )


class TestGraphWaveNet(unittest.TestCase):
    def test_receptive_field_alignment_support_branches_and_gradients(self):
        runtime = build_model_runtime(
            "graph_wavenet", load_protocol(), run_mode="smoke"
        )
        model = runtime.model
        batch = node_batch()
        prediction = runtime.adapter(model, batch).prediction
        self.assertEqual(tuple(prediction.shape), (1, 134, 10))
        self.assertEqual(model.receptive_field, 13)
        self.assertEqual(model.last_shape_trace["final_temporal_width"], 132)
        self.assertEqual(
            model.last_shape_trace["forecast_anchor"],
            "last_valid_causal_temporal_position",
        )
        prediction.square().mean().backward()
        for name, parameter in (
            ("nodevec1", model.nodevec1),
            ("nodevec2", model.nodevec2),
            ("filter", model.filter_convs[0].weight),
            ("gate", model.gate_convs[0].weight),
            ("diffusion", model.graph_convs[0].projection.weight),
            ("skip", model.skip_convs[0].weight),
            ("end", model.end_conv_2.weight),
        ):
            assert_finite_nonzero(self, parameter, name)
        model.eval()
        with torch.no_grad():
            both = model(batch.x, support_mode="physical_adaptive")
            physical = model(batch.x, support_mode="physical_only")
            adaptive = model(batch.x, support_mode="adaptive_only")
        self.assertFalse(torch.allclose(both, physical))
        self.assertFalse(torch.allclose(both, adaptive))
        adjacency = model.adaptive_adjacency()
        self.assertEqual(tuple(adjacency.shape), (134, 134))
        self.assertTrue(
            torch.allclose(adjacency.sum(1), torch.ones(134), atol=1e-6)
        )


class TestMTGNN(unittest.TestCase):
    def test_constructor_inception_mixhop_and_gradients(self):
        constructor = DirectedGraphConstructor(8, 4, 3, 3.0)
        adjacency = constructor()
        self.assertEqual(tuple(adjacency.shape), (8, 8))
        self.assertLessEqual(int((adjacency > 0).sum(1).max()), 3)
        self.assertFalse(torch.allclose(adjacency, adjacency.T))
        inception = DilatedInception(4, 8, 1)
        inc = inception(torch.randn(2, 4, 5, 20))
        self.assertEqual(inception.last_branch_widths, [19, 18, 15, 14])
        self.assertEqual(inc.shape[-1], 14)

        runtime = build_model_runtime("mtgnn", load_protocol(), run_mode="smoke")
        model = runtime.model
        prediction = runtime.adapter(model, node_batch()).prediction
        self.assertEqual(tuple(prediction.shape), (1, 134, 10))
        self.assertEqual(model.receptive_field, 19)
        self.assertEqual(model.last_shape_trace["final_temporal_width"], 1)
        prediction.square().mean().backward()
        for name, parameter in (
            ("emb1", model.graph_constructor.emb1.weight),
            ("emb2", model.graph_constructor.emb2.weight),
            ("inception-k2", model.filter_convs[0].branches[0].weight),
            ("inception-k7", model.filter_convs[0].branches[3].weight),
            ("forward-mixhop", model.forward_props[0].projection.weight),
            ("reverse-mixhop", model.reverse_props[0].projection.weight),
            ("end", model.end_conv_2.weight),
        ):
            assert_finite_nonzero(self, parameter, name)
        learned = model.learned_adjacency()
        self.assertLessEqual(int((learned > 0).sum(1).max()), 20)
        self.assertEqual(model.graph_identity["physical_support_names"], [])


class TestAGCRN(unittest.TestCase):
    def test_dagg_napl_two_basis_recurrent_head_and_gradients(self):
        layer = AdaptiveGraphConv(2, 3, cheb_order=2, embed_dim=4)
        x = torch.randn(1, 5, 2)
        embedding = torch.randn(5, 4)
        adjacency = torch.softmax(torch.relu(embedding @ embedding.T), dim=1)
        self.assertEqual(layer(x, embedding, adjacency).shape, (1, 5, 3))
        self.assertGreater(layer.last_node_weight_variation, 0.0)

        runtime = build_model_runtime("agcrn", load_protocol(), run_mode="smoke")
        model = runtime.model
        batch = node_batch()
        prediction = runtime.adapter(model, batch).prediction
        self.assertEqual(tuple(prediction.shape), (1, 134, 10))
        self.assertEqual(model.last_shape_trace["adaptive_basis_count"], 2)
        self.assertEqual(model.last_shape_trace["encoder_layers"], 2)
        prediction.square().mean().backward()
        for name, parameter in (
            ("node-embedding", model.node_embeddings),
            ("layer0-gate-pool", model.cells[0].gate.weights_pool),
            ("layer0-candidate-pool", model.cells[0].candidate.weights_pool),
            ("layer1-gate-pool", model.cells[1].gate.weights_pool),
            ("head", model.horizon_head.weight),
        ):
            assert_finite_nonzero(self, parameter, name)
        model.eval()
        with torch.no_grad():
            adaptive = model(batch.x)
            identity = model(
                batch.x, adjacency_override=torch.eye(model.num_nodes)
            )
        self.assertFalse(torch.allclose(adaptive, identity))


class TestSTID(unittest.TestCase):
    def test_timestamp_identity_ranges_anchor_change_and_future_independence(self):
        runtime = build_model_runtime("stid", load_protocol(), run_mode="smoke")
        batch = node_batch()
        x, tod, dow = runtime.adapter.prepare_model_inputs(batch)
        self.assertEqual(tod.tolist(), [72])
        self.assertEqual(dow.tolist(), [0])
        future_changed = clone_batch(batch)
        future_changed.target.fill_(123.0)
        self.assertEqual(
            runtime.adapter.prepare_model_inputs(future_changed)[1].tolist(),
            [72],
        )
        anchor_changed = clone_batch(batch)
        anchor_changed.metadata["history_end_timestamp"] = [
            "2021-01-05 01:20:00"
        ]
        anchor_changed.metadata["history_end_time_of_day_id"] = [8]
        anchor_changed.metadata["history_end_day_of_week_id"] = [1]
        _, changed_tod, changed_dow = runtime.adapter.prepare_model_inputs(
            anchor_changed
        )
        self.assertEqual(changed_tod.tolist(), [8])
        self.assertEqual(changed_dow.tolist(), [1])
        missing = clone_batch(batch)
        missing.metadata.pop("history_end_timestamp")
        with self.assertRaises(ContractError):
            runtime.adapter.prepare_model_inputs(missing)

    def test_embeddings_mlp_gradients_and_cross_node_isolation(self):
        runtime = build_model_runtime("stid", load_protocol(), run_mode="smoke")
        model = runtime.model
        batch = node_batch()
        prediction = runtime.adapter(model, batch).prediction
        prediction.square().mean().backward()
        for name, parameter in (
            ("series", model.time_series_embedding.weight),
            ("node", model.node_embedding),
            ("tod", model.time_of_day_embedding.weight),
            ("dow", model.day_of_week_embedding.weight),
            ("mlp0", model.encoder[0].linear1.weight),
            ("head", model.regression_head.weight),
        ):
            assert_finite_nonzero(self, parameter, name)
        model.eval()
        perturbed = clone_batch(batch)
        perturbed.x[:, :, 0] += 10.0
        with torch.no_grad():
            base = runtime.adapter(model, batch).prediction
            changed = runtime.adapter(model, perturbed).prediction
        self.assertFalse(torch.allclose(base[:, 0], changed[:, 0]))
        self.assertTrue(torch.equal(base[:, 1:], changed[:, 1:]))
        self.assertFalse(model.last_shape_trace["graph_propagation"])


class TestCheckpointAndPreflightIdentity(unittest.TestCase):
    def test_strict_reload_regenerates_same_adaptive_identity(self):
        protocol = load_protocol()
        for model_id in MODELS:
            runtime = build_model_runtime(
                model_id, protocol, run_mode="smoke"
            )
            with tempfile.TemporaryDirectory() as directory:
                manager = CheckpointManager(
                    directory,
                    protocol_hash=protocol.protocol_hash,
                    model_id=model_id,
                    resolved_config=dict(runtime.effective_config),
                    effective_config=dict(runtime.effective_config),
                )
                path = manager.save(
                    "best_checkpoint.pt",
                    epoch=1,
                    global_step=1,
                    monitor_value=1.0,
                    model=runtime.model,
                )
                if model_id == "graph_wavenet":
                    before = canonical_tensor_hash(
                        runtime.model.adaptive_adjacency()
                    )
                elif model_id == "mtgnn":
                    before = canonical_tensor_hash(
                        runtime.model.learned_adjacency()
                    )
                elif model_id == "agcrn":
                    before = canonical_tensor_hash(
                        runtime.model.adaptive_adjacency()
                    )
                else:
                    before = canonical_tensor_hash(
                        runtime.model.node_embedding
                    )
                manager.load(path, runtime.model)
                if model_id == "graph_wavenet":
                    after = canonical_tensor_hash(
                        runtime.model.adaptive_adjacency()
                    )
                elif model_id == "mtgnn":
                    after = canonical_tensor_hash(
                        runtime.model.learned_adjacency()
                    )
                elif model_id == "agcrn":
                    after = canonical_tensor_hash(
                        runtime.model.adaptive_adjacency()
                    )
                else:
                    after = canonical_tensor_hash(
                        runtime.model.node_embedding
                    )
                self.assertEqual(before, after)

    def test_preflight_policy_mismatches_are_not_reused(self):
        for model_id in MODELS:
            identity = preflight_identity(model_id)
            for required in (
                "node_order_hash",
                "adaptive_graph_policy",
                "node_identity_policy",
                "temporal_identity_policy",
                "source_closure_hash",
            ):
                self.assertIn(required, identity)
            with tempfile.TemporaryDirectory() as directory:
                path = preflight_artifact_path(model_id, root=directory)
                path.parent.mkdir(parents=True)
                payload = {
                    **identity,
                    "status": "PASS",
                    "forward_completed": True,
                    "backward_completed": True,
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertIsNotNone(
                    read_matching_pass(model_id, root=directory)
                )
                for key in (
                    "node_order_hash",
                    "adaptive_graph_policy",
                    "physical_support_names",
                    "temporal_identity_policy",
                    "source_closure_hash",
                    "model_config_hash",
                ):
                    changed = dict(payload)
                    changed[key] = "wrong"
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    self.assertIsNone(
                        read_matching_pass(model_id, root=directory)
                    )
                    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
