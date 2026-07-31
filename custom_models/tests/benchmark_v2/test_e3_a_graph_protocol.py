from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_v2.checkpointing import config_hash
from benchmark_v2.graph import (
    MATRIX_NAMES,
    GraphProtocolError,
    build_physical_graph,
    graph_protocol_check,
    identity_from_bundle,
    load_graph_bundle,
    node_schema_source_hash,
    validate_graph_identity,
    validate_native_graph_input,
    validate_node_order,
)
from benchmark_v2.graph.builder import _stable_neighbors
from benchmark_v2.graph.hashing import (
    canonical_json,
    file_sha256,
    matrix_hash,
    node_order_hash,
    stable_hash,
)
from benchmark_v2.graph.transforms import (
    gcn_support,
    random_walk,
    scaled_laplacian_fixed_two,
    symmetric_normalized_laplacian,
)
from benchmark_v2.registry import load_registry
from benchmark_v2.runtime import PROJECT_ROOT
from benchmark_v2.upstream.tslib_loader import ALLOWED_TSLIB_MODELS


LOCATION_PATH = (
    PROJECT_ROOT / "dataset" / "sdwpf_turb_location_elevation.csv"
)


class E3AGraphProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_graph_bundle()
        cls.location = pd.read_csv(LOCATION_PATH)

    def test_node_order_is_exact_unique_integer_134(self):
        ids = self.bundle.ordered_node_ids
        self.assertEqual(len(ids), 134)
        self.assertEqual(len(set(ids)), 134)
        self.assertTrue(all(type(value) is int for value in ids))
        self.assertEqual(list(ids), list(range(1, 135)))
        self.assertEqual(
            self.bundle.node_order_hash,
            "1bcfaa0afafb0a6535b9d24129bb2ba260cfd1782b9b485d960e6cfe4af90c35",
        )

    def test_node_order_type_mixing_and_boolean_fail(self):
        with self.assertRaises(TypeError):
            node_order_hash([1, "2"])
        with self.assertRaises(TypeError):
            node_order_hash([True, False])

    def test_reorder_missing_extra_and_duplicate_are_detected(self):
        frozen = list(self.bundle.ordered_node_ids)
        for altered in (
            [frozen[1], frozen[0], *frozen[2:]],
            frozen[:-1],
            [*frozen, 135],
            [*frozen[:-1], frozen[-2]],
        ):
            with self.assertRaises((GraphProtocolError, TypeError)):
                validate_node_order(
                    altered,
                    frozen,
                    expected_hash=self.bundle.node_order_hash,
                )

    def test_input_target_location_audit_is_aligned(self):
        audit_path = (
            PROJECT_ROOT
            / "custom_models/docs/benchmark_v2/E3_A/node_order_audit.json"
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertTrue(audit["input_target_key_order_aligned"])
        self.assertTrue(audit["input_target_location_sets_equal"])
        self.assertEqual(audit["input_duplicate_key_count"], 0)
        self.assertEqual(audit["target_duplicate_key_count"], 0)
        self.assertEqual(audit["location_duplicate_coordinate_count"], 0)
        self.assertEqual(audit["location_coordinate_null_count"], 0)

    def test_selected_k_is_first_connected_candidate(self):
        build = build_physical_graph(
            list(self.bundle.ordered_node_ids), self.location
        )
        records = build.diagnostics["candidate_k_component_counts"]
        self.assertEqual([row["component_count"] for row in records[:4]], [48, 7, 2, 1])
        self.assertEqual(build.diagnostics["selected_k"], 4)
        self.assertTrue(all(row["component_count"] == 1 for row in records[3:]))

    def test_equal_distance_tie_break_is_canonical_index(self):
        distances = np.asarray([np.inf, 1.0, 1.0, 2.0, 1.0])
        self.assertEqual(_stable_neighbors(distances, 3).tolist(), [1, 2, 4])

    def test_duplicate_coordinate_fails_closed(self):
        changed = self.location.copy()
        changed.loc[1, ["x", "y"]] = changed.loc[0, ["x", "y"]].to_numpy()
        with self.assertRaisesRegex(
            GraphProtocolError, "BLOCKED_DUPLICATE_COORDINATE"
        ):
            build_physical_graph(list(self.bundle.ordered_node_ids), changed)

    def test_base_graph_shape_symmetry_self_loop_and_connectivity(self):
        self.assertEqual(self.bundle.A_directed.shape, (134, 134))
        self.assertEqual(self.bundle.A_undirected.shape, (134, 134))
        self.assertFalse(np.any(np.diag(self.bundle.A_directed)))
        self.assertFalse(np.any(np.diag(self.bundle.A_undirected)))
        self.assertTrue(
            np.array_equal(
                self.bundle.A_undirected, self.bundle.A_undirected.T
            )
        )
        check = graph_protocol_check()
        self.assertEqual(check["status"], "PASS")
        self.assertEqual(check["node_count"], 134)

    def test_sigma_and_weight_formula(self):
        build = build_physical_graph(
            list(self.bundle.ordered_node_ids), self.location
        )
        sigma = build.diagnostics["sigma"]
        self.assertGreater(sigma, 0.0)
        support = build.matrices["A_binary_directed"] > 0
        coords = self.location.set_index("TurbID").loc[
            list(self.bundle.ordered_node_ids), ["x", "y"]
        ].to_numpy(np.float64)
        difference = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt(np.sum(difference * difference, axis=-1))
        expected = np.exp(-np.square(distances[support] / sigma))
        self.assertTrue(
            np.allclose(build.matrices["A_directed"][support], expected)
        )
        self.assertTrue(np.all(build.matrices["A_directed"][support] > 0))
        self.assertTrue(np.all(build.matrices["A_directed"][support] <= 1))

    def test_derived_matrix_formulas(self):
        self.assertTrue(
            np.allclose(
                self.bundle.A_gcn, gcn_support(self.bundle.A_undirected)
            )
        )
        self.assertTrue(
            np.allclose(
                self.bundle.P_forward, random_walk(self.bundle.A_directed)
            )
        )
        self.assertTrue(
            np.allclose(
                self.bundle.P_reverse, random_walk(self.bundle.A_directed.T)
            )
        )
        expected_laplacian = symmetric_normalized_laplacian(
            self.bundle.A_undirected
        )
        self.assertTrue(np.allclose(self.bundle.L_sym, expected_laplacian))
        self.assertTrue(
            np.allclose(
                self.bundle.L_tilde,
                scaled_laplacian_fixed_two(expected_laplacian),
            )
        )

    def test_random_walk_rows_are_normalized(self):
        self.assertTrue(
            np.allclose(self.bundle.P_forward.sum(axis=1), 1.0, atol=1e-12)
        )
        self.assertTrue(
            np.allclose(self.bundle.P_reverse.sum(axis=1), 1.0, atol=1e-12)
        )

    def test_all_matrices_have_frozen_hashes_and_are_read_only(self):
        self.assertEqual(set(self.bundle.matrix_hashes), set(MATRIX_NAMES))
        for name in MATRIX_NAMES:
            matrix = getattr(self.bundle, name)
            self.assertEqual(matrix.shape, (134, 134))
            self.assertTrue(np.isfinite(matrix).all())
            self.assertFalse(matrix.flags.writeable)
            self.assertEqual(matrix_hash(matrix), self.bundle.matrix_hashes[name])
        with self.assertRaises(ValueError):
            self.bundle.A_directed[0, 1] = 99.0

    def test_hash_json_key_order_and_matrix_content(self):
        self.assertEqual(
            stable_hash({"a": 1, "b": 2}), stable_hash({"b": 2, "a": 1})
        )
        changed = self.bundle.A_directed.copy()
        changed.setflags(write=True)
        changed[0, 1] += 0.01
        self.assertNotEqual(
            matrix_hash(changed), matrix_hash(self.bundle.A_directed)
        )
        self.assertEqual(
            canonical_json({"b": 2, "a": 1}), '{"a":1,"b":2}'
        )

    def test_file_timestamp_does_not_change_content_hash(self):
        handle = tempfile.NamedTemporaryFile(delete=False)
        path = Path(handle.name)
        try:
            handle.write(b"graph-hash-stable")
            handle.close()
            before = file_sha256(path)
            stat = path.stat()
            os.utime(path, (stat.st_atime + 10, stat.st_mtime + 10))
            self.assertEqual(before, file_sha256(path))
        finally:
            if not handle.closed:
                handle.close()
            path.unlink(missing_ok=True)

    def test_node_reorder_and_coordinate_change_change_identity(self):
        reversed_ids = list(reversed(self.bundle.ordered_node_ids))
        self.assertNotEqual(
            node_order_hash(reversed_ids), self.bundle.node_order_hash
        )
        changed = self.location.copy()
        changed.loc[0, "x"] += 0.25
        changed_build = build_physical_graph(
            list(self.bundle.ordered_node_ids), changed
        )
        self.assertNotEqual(
            changed_build.node_metadata_hash, self.bundle.node_metadata_hash
        )
        self.assertNotEqual(
            changed_build.graph_bundle_hash, self.bundle.graph_bundle_hash
        )

    def test_elevation_change_metadata_only_identity_strategy(self):
        baseline = build_physical_graph(
            list(self.bundle.ordered_node_ids), self.location
        )
        changed = self.location.copy()
        changed.loc[0, "Ele"] += 1.0
        modified = build_physical_graph(
            list(self.bundle.ordered_node_ids), changed
        )
        self.assertNotEqual(
            baseline.node_metadata_hash, modified.node_metadata_hash
        )
        self.assertEqual(baseline.matrix_hashes, modified.matrix_hashes)
        self.assertEqual(
            baseline.graph_bundle_hash, modified.graph_bundle_hash
        )

    def test_node_schema_hash_excludes_feature_target_and_mask_values(self):
        common = {
            "logical_path": "dataset/source.parquet",
            "node_id_column": "TurbID",
            "node_id_type": "integer",
            "ordered_node_ids": list(range(1, 135)),
            "row_count": 1340,
            "timestamp_count": 10,
        }
        before = node_schema_source_hash(**common)
        dummy_feature_values = np.arange(1340)
        dummy_target_values = dummy_feature_values[::-1]
        dummy_mask_values = dummy_feature_values % 2
        self.assertNotEqual(
            dummy_feature_values.tolist(), dummy_target_values.tolist()
        )
        self.assertGreater(dummy_mask_values.sum(), 0)
        self.assertEqual(before, node_schema_source_hash(**common))

    def test_builder_edge_path_has_no_prediction_value_inputs(self):
        source = inspect.getsource(build_physical_graph)
        for forbidden in (
            "Patv_raw",
            "Patv_clean_for_input",
            "valid_target_mask",
            "Wspd",
            "Wdir",
            "train_loss",
            "validation_loss",
            "test_loss",
        ):
            self.assertNotIn(forbidden, source)

    def test_graph_import_does_not_import_torch_or_create_cuda(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(PROJECT_ROOT / "custom_models" / "src")
        command = [
            sys.executable,
            "-c",
            (
                "import sys; import benchmark_v2.graph; "
                "print('torch' in sys.modules)"
            ),
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(completed.stdout.strip(), "False")

    def test_bundle_to_is_explicit_and_cpu_only_when_requested(self):
        tensor_bundle = self.bundle.to("cpu")
        self.assertEqual(tensor_bundle.device, "cpu")
        self.assertEqual(set(tensor_bundle.matrices), set(MATRIX_NAMES))
        self.assertTrue(
            all(str(value.device) == "cpu" for value in tensor_bundle.matrices.values())
        )

    def test_runtime_node_and_graph_identity_fail_closed(self):
        identity = identity_from_bundle(self.bundle)
        self.bundle.validate_runtime_node_order(
            list(self.bundle.ordered_node_ids)
        )
        self.bundle.validate_runtime_identity(
            graph_protocol_hash=identity.graph_protocol_hash,
            node_order_hash=identity.node_order_hash,
            graph_bundle_hash=identity.graph_bundle_hash,
        )
        for key in (
            "graph_protocol_hash",
            "node_order_hash",
            "graph_bundle_hash",
        ):
            values = identity.to_dict()
            values[key] = "mismatch"
            with self.assertRaises(GraphProtocolError):
                self.bundle.validate_runtime_identity(
                    graph_protocol_hash=values["graph_protocol_hash"],
                    node_order_hash=values["node_order_hash"],
                    graph_bundle_hash=values["graph_bundle_hash"],
                )

    def test_checkpoint_evaluation_and_preflight_identity_mismatch(self):
        identity = identity_from_bundle(self.bundle)
        expected = identity.to_dict()
        validate_graph_identity(expected, expected, context="checkpoint reload")
        for context, key in (
            ("checkpoint reload", "graph_bundle_hash"),
            ("evaluation", "node_order_hash"),
            ("hardware preflight", "graph_protocol_hash"),
        ):
            actual = dict(expected)
            actual[key] = "different"
            with self.assertRaisesRegex(GraphProtocolError, context):
                validate_graph_identity(expected, actual, context=context)

    def test_non_graph_config_identity_is_unchanged_when_graph_omitted(self):
        legacy = {"model_id": "gru", "d_model": 32, "protocol_hash": "p"}
        self.assertEqual(config_hash(legacy), config_hash(dict(legacy)))
        self.assertNotIn("graph_protocol_hash", legacy)

    def test_native_dummy_contract_preserves_nodes_and_shape(self):
        import torch

        x = torch.zeros(2, 144, 134, 16)
        validate_native_graph_input(
            x,
            runtime_node_ids=list(self.bundle.ordered_node_ids),
            bundle=self.bundle,
        )

        class Dummy(torch.nn.Module):
            def forward(self, value, graph_bundle):
                self.seen_shape = tuple(value.shape)
                self.seen_bundle = graph_bundle.graph_bundle_hash
                return value[:, -1, :, 15:16].repeat(1, 1, 10)

        model = Dummy()
        output = model(x, self.bundle)
        self.assertEqual(model.seen_shape, (2, 144, 134, 16))
        self.assertEqual(model.seen_bundle, self.bundle.graph_bundle_hash)
        self.assertEqual(tuple(output.shape), (2, 134, 10))
        self.assertEqual(
            list(inspect.signature(model.forward).parameters),
            ["value", "graph_bundle"],
        )

    def test_registry_identity_resolution_and_counts(self):
        registry = load_registry()
        self.assertEqual(len(registry.list()), 28)
        entry = registry.get("stgcn")
        self.assertEqual(entry.display_name, "STGCN")
        self.assertEqual(
            entry.values["identity_status"],
            "VERIFIED_CLASSIC_STGCN_IJCAI_2018",
        )
        self.assertEqual(
            entry.values["implementation_status"], "IMPLEMENTED"
        )
        self.assertEqual(entry.runtime_status, "AVAILABLE_TRAINABLE")
        self.assertTrue(entry.supports_train)
        self.assertTrue(entry.supports_evaluate)
        self.assertEqual(registry.get("stcn_stgcn_unresolved").canonical_id, "stgcn")
        self.assertEqual(sum(e.supports_train for e in registry.list()), 26)
        stats = registry.statistics()
        self.assertEqual(stats["available_locally_verified"], 16)
        self.assertEqual(stats["available_with_hardware_preflight"], 10)
        self.assertEqual(stats["non_trainable_available"], 2)
        self.assertEqual(stats["true_blocked_or_unavailable"], 0)
        self.assertEqual(len(ALLOWED_TSLIB_MODELS), 18)


if __name__ == "__main__":
    unittest.main()
