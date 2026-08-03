from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark_v2.experiments.e5_common_loss.a8_reference import (
    validate_a8_reference,
)
from benchmark_v2.experiments.e5_common_loss.aggregation import (
    aggregate,
)
from benchmark_v2.experiments.e5_common_loss.config_diff import (
    compare_configs,
)
from benchmark_v2.experiments.e5_common_loss.contracts import (
    NONTRAINABLE_MODELS,
    TRAINABLE_MODELS,
    validate_registry_contract,
)
from benchmark_v2.experiments.e5_common_loss.loss_profile import (
    CLI_PROFILE_ID,
    check_loss_profile,
    e5_loss_adapter,
    get_profile_metadata,
)
from benchmark_v2.experiments.e5_common_loss.numerical_parity import (
    run_numerical_parity,
)
from benchmark_v2.experiments.e5_common_loss.readiness import (
    build_readiness,
)
from benchmark_v2.experiments.e5_common_loss.runner import (
    apply_experiment_profile,
    canonical_base_model_config_hash,
)
from benchmark_v2.experiments.e5_common_loss.variant_manifest import (
    _base_config,
    build_run_id_map,
    build_variant_manifest,
)
from benchmark_v2.hardware_preflight import (
    preflight_artifact_path,
    preflight_identity,
    read_matching_pass,
)
from benchmark_v2.registry import load_registry


class E5CommonLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_variant_manifest()

    def test_loss_profile_identity(self):
        check = check_loss_profile(CLI_PROFILE_ID)
        self.assertEqual(check["status"], "PASS")
        self.assertEqual(
            check["profile"]["loss_id"], "masked_score_aligned_hybrid"
        )
        self.assertFalse(check["imports_st_mgprompt_model"])
        self.assertFalse(check["creates_gpu_model"])

    def test_numerical_value_gradient_mask_and_amp_parity(self):
        result = run_numerical_parity()
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["cpu_fp32"], "PASS")
        self.assertIn(result["cuda_amp"], {"PASS", "NOT_AVAILABLE"})
        all_masked = [
            item for item in result["cases"] if item["case"] == "all_masked"
        ]
        self.assertTrue(all(item["adapter_is_none"] for item in all_masked))

    def test_direct_loss_all_masked_behavior(self):
        import torch

        shape = (2, 3, 10)
        result = e5_loss_adapter(
            torch.zeros(shape),
            torch.zeros(shape),
            torch.zeros(shape, dtype=torch.bool),
        )
        self.assertIsNone(result)

    def test_variant_counts_and_registry(self):
        self.assertEqual(
            self.manifest["counts"],
            {
                "TRAIN_COMMON_LOSS": 26,
                "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC": 2,
                "REFERENCE_ONLY_FORMAL_A8": 1,
                "TOTAL": 29,
            },
        )
        self.assertEqual(validate_registry_contract(load_registry())["status"], "PASS")

    def test_run_ids_are_extracted_and_frozen(self):
        mapping = build_run_id_map()
        self.assertEqual(len(mapping["runs"]), 28)
        for item in mapping["runs"].values():
            self.assertTrue(item["base_run_id"].endswith("_seed2026"))
            self.assertTrue(item["e5_run_id"].endswith("_loss_msa_hybrid_seed2026"))

    def test_all_26_config_diffs_are_allowlisted_only(self):
        profile = get_profile_metadata(CLI_PROFILE_ID)
        for model_id in TRAINABLE_MODELS:
            with self.subTest(model_id=model_id):
                base = _base_config(model_id)
                runtime = SimpleNamespace(
                    model_id=model_id, effective_config=copy.deepcopy(base)
                )
                apply_experiment_profile(
                    runtime,
                    CLI_PROFILE_ID,
                    run_id=f"{model_id}_loss_msa_hybrid_seed2026",
                    output_root="formal-root",
                    provenance={"test": True},
                )
                diff = compare_configs(base, runtime.effective_config)
                self.assertEqual(diff["status"], "PASS", diff)
                entry = next(
                    item
                    for item in self.manifest["entries"]
                    if item["model_id"] == model_id
                )
                self.assertEqual(
                    entry["base_model_config_hash"],
                    canonical_base_model_config_hash(model_id),
                )
                self.assertEqual(
                    runtime.effective_config["loss"]["profile_hash"],
                    profile["loss_profile_hash"],
                )

    def test_config_diff_rejects_model_optimizer_batch_and_graph_changes(self):
        base = {
            "architecture": "same",
            "optimizer": "Adam",
            "train_batch_size": 32,
            "graph_id": "frozen",
        }
        for key, value in (
            ("architecture", "changed"),
            ("optimizer", "SGD"),
            ("train_batch_size", 1),
            ("graph_id", "changed"),
        ):
            with self.subTest(key=key):
                variant = dict(base)
                variant[key] = value
                self.assertEqual(
                    compare_configs(base, variant)["status"],
                    "BLOCKED_CONFIG_DIFF",
                )

    def test_nontrainable_contract_has_no_optimizer_or_checkpoint(self):
        for model_id in NONTRAINABLE_MODELS:
            entry = next(
                item
                for item in self.manifest["entries"]
                if item["model_id"] == model_id
            )
            self.assertEqual(entry["training_mode"], "EVALUATE_ONLY")
            self.assertEqual(entry["training_loss"], "NOT_APPLICABLE")
            self.assertFalse(entry["trained_with_common_loss"])
            self.assertTrue(entry["common_loss_evaluation_applied"])
            base = _base_config(model_id)
            self.assertIsNone(base["optimizer"])
            self.assertIsNone(base["scheduler"])

    def test_a8_is_valid_read_only_reference(self):
        result = validate_a8_reference()
        self.assertEqual(result["status"], "VALID", result)
        self.assertTrue(result["formal_complete"])
        self.assertTrue(result["metrics_complete"])
        self.assertTrue(result["protocol_match"])
        self.assertFalse(result["checkpoint_copied"])
        self.assertFalse(result["metrics_copied"])
        self.assertFalse(result["retrained_in_e5"])

    def test_preflight_identity_is_loss_specific_and_base_is_unchanged(self):
        base = preflight_identity("gru")
        explicit_base = preflight_identity(
            "gru", experiment_profile="default_benchmark_v1"
        )
        e5 = preflight_identity("gru", experiment_profile=CLI_PROFILE_ID)
        self.assertEqual(base, explicit_base)
        self.assertNotIn("loss_profile_hash", base)
        self.assertEqual(e5["loss_id"], "masked_score_aligned_hybrid")
        self.assertNotEqual(
            preflight_artifact_path("gru"),
            preflight_artifact_path("gru", experiment_profile=CLI_PROFILE_ID),
        )
        for key in (
            "experiment_profile_id",
            "e5_common_loss_protocol_hash",
            "loss_id",
            "loss_source_hash",
            "loss_profile_hash",
            "seed",
        ):
            self.assertIn(key, e5)

    def test_e5_preflight_pass_reuse_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity = preflight_identity(
                "gcn", experiment_profile=CLI_PROFILE_ID
            )
            path = preflight_artifact_path(
                "gcn",
                root=root,
                experiment_profile=CLI_PROFILE_ID,
            )
            path.parent.mkdir(parents=True)
            valid = {
                **identity,
                "status": "PASS",
                "forward_completed": True,
                "backward_completed": True,
            }
            path.write_text(json.dumps(valid), encoding="utf-8")
            self.assertIsNotNone(
                read_matching_pass(
                    "gcn",
                    root=root,
                    experiment_profile=CLI_PROFILE_ID,
                )
            )
            for key in (
                "loss_source_hash",
                "loss_profile_hash",
                "base_model_config_hash",
                "model_source_closure_hash",
                "graph_protocol_hash",
                "B",
            ):
                with self.subTest(key=key):
                    bad = dict(valid)
                    bad[key] = "mismatch"
                    path.write_text(json.dumps(bad), encoding="utf-8")
                    self.assertIsNone(
                        read_matching_pass(
                            "gcn",
                            root=root,
                            experiment_profile=CLI_PROFILE_ID,
                        )
                    )
            path.write_text(json.dumps(valid), encoding="utf-8")
            self.assertIsNone(read_matching_pass("gcn", root=root))

    def test_readiness_lists_29_and_aggregation_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = build_readiness(output_root=root, legacy_scope29=True)
            self.assertEqual(len(report["entries"]), 29)
            self.assertEqual(report["status"], "NOT_READY")
            self.assertEqual(report["ready_entries"], 1)
            with self.assertRaisesRegex(RuntimeError, "E5_RESULT_NOT_READY"):
                aggregate(
                    output_root=root,
                    require_complete=True,
                    legacy_scope29=True,
                )
            for name in (
                "COMMON_LOSS_ARCHITECTURE_SEED2026.xlsx",
                "COMMON_LOSS_ARCHITECTURE_SEED2026.md",
                "common_loss_architecture_seed2026.csv",
                "common_loss_architecture_audit.json",
            ):
                self.assertFalse((root / name).exists())

    def test_aggregation_requires_explicit_complete_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "require"):
                aggregate(
                    output_root=temp,
                    require_complete=False,
                    legacy_scope29=True,
                )


if __name__ == "__main__":
    unittest.main()

