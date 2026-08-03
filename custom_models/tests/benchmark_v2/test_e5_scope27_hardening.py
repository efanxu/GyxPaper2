from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark_v2.experiments.e5_common_loss.active_scope import (
    ActiveScopeError,
    active_scope_path,
    load_active_scope_pointer,
)
from benchmark_v2.experiments.e5_common_loss.runner import apply_experiment_profile
from benchmark_v2.model_source_identity import (
    ModelSourceIdentityError,
    _resolve_repo_file,
    _sha256_file,
    canonical_hash_for_records,
    canonical_model_source_identity,
)
from benchmark_v2.precision import apply_model_precision_policy
from benchmark_v2.registry import load_registry


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "custom_models/docs/benchmark_v2/E5/E5_SCOPE27_VARIANT_MANIFEST.json"
)


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _entry(manifest: dict, model_id: str) -> dict:
    return next(row for row in manifest["entries"] if row["model_id"] == model_id)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _make_fixture(root: Path, manifest: dict, model_id: str) -> Path:
    entry = _entry(manifest, model_id)
    evaluate_only = entry["entry_type"] == "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC"
    run_dir = root / entry["e5_run_id"]
    run_dir.mkdir(parents=True)
    identity = entry["source_identity"]
    precision = entry["precision_identity"]
    effective = {
        "model_id": model_id,
        "run_mode": "formal",
        "artifact_profile": "NON_TRAINABLE" if evaluate_only else "TRAIN",
        "formal_registry_entry": True,
        "formal_training": not evaluate_only,
        "experiment_profile_id": manifest["experiment_profile_id"],
        "training_batch_profile_id": manifest["training_profile_id"],
        "training_batch_profile_hash": manifest["training_profile_hash"],
        "base_benchmark_protocol_hash": "0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b",
        "train_batch_size": 4,
        "val_batch_size": 4,
        "test_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "effective_train_batch_size": 4,
        **precision,
        "loss": {
            "id": manifest["loss_identity"]["loss_id"],
            "source_hash": manifest["loss_identity"]["loss_source_hash"],
            "profile_hash": manifest["loss_identity"]["loss_profile_hash"],
        },
        "provenance": {
            "active_scope_id": manifest["scope_id"],
            "base_model_config_hash": entry["base_model_config_hash"],
            "base_model_source_closure_hash": entry["base_model_source_hash"],
            "base_model_source_identity_schema_version": manifest[
                "source_identity_schema_version"
            ],
            "base_model_source_closure_files": identity[
                "source_closure_files"
            ],
            "e5_common_loss_protocol_hash": manifest["loss_identity"][
                "e5_common_loss_protocol_hash"
            ],
        },
    }
    _write_json(run_dir / "resolved_config.json", effective)
    _write_json(run_dir / "effective_config.json", effective)
    _write_json(
        run_dir / "protocol_check.json",
        {"protocol_hash": manifest["benchmark_protocol_hash"]},
    )
    _write_json(run_dir / "environment.json", {})
    _write_json(
        run_dir / "data_signature.json",
        {
            "dataset_id": "SDWPF",
            "node_count": 134,
            "feature_order_hash": manifest["dataset_identity"][
                "feature_order_hash"
            ],
        },
    )
    _write_json(run_dir / "model_summary.json", {"model_id": model_id})
    _write_json(
        run_dir / "artifact_manifest.json",
        {
            "schema_version": "artifact_schema_v1",
            "artifact_profile": effective["artifact_profile"],
            "protocol_hash": manifest["benchmark_protocol_hash"],
            "run_mode": "formal",
            "formal_training": not evaluate_only,
            "model_id": model_id,
            "training_batch_profile_id": manifest["training_profile_id"],
            "training_batch_profile_hash": manifest["training_profile_hash"],
        },
    )
    for horizon in (3, 6, 10):
        _write_json(
            run_dir / f"metrics_eval_h{horizon}.json",
            {
                "horizon": horizon,
                "Score": 1.0,
                "MAE": 1.0,
                "RMSE": 1.0,
                "R2": 1.0,
            },
        )
    (run_dir / "metrics.csv").write_text("horizon,Score,MAE,RMSE,R2\n", encoding="utf-8")
    _write_json(
        run_dir / "prediction_metadata.json",
        {"source_checkpoint": None if evaluate_only else "best_checkpoint.pt"},
    )
    _write_json(
        run_dir / "run_status.json",
        {
            "status": "COMPLETED",
            "run_mode": "formal",
            "artifact_profile": effective["artifact_profile"],
            "formal_training": not evaluate_only,
        },
    )
    if evaluate_only:
        _write_json(
            run_dir / "baseline_state.json",
            {
                "schema_version": "e5_non_trainable_baseline_v1",
                "model_id": model_id,
                "training_mode": "EVALUATE_ONLY",
                "training_loss": "NOT_APPLICABLE",
                "trained_with_common_loss": False,
                "common_loss_evaluation_applied": True,
                "best_checkpoint": None,
                "best_epoch": None,
            },
        )
        _write_json(
            run_dir / "common_loss_diagnostic.json",
            {
                "schema_version": "e5_common_loss_diagnostic_v1",
                "model_id": model_id,
                "loss_id": "masked_score_aligned_hybrid",
                "loss_space": "normalized Patv_raw",
                "training_loss": "NOT_APPLICABLE",
                "trained_with_common_loss": False,
                "common_loss_evaluation_applied": True,
                "value": 1.0,
            },
        )
    else:
        (run_dir / "best_checkpoint.pt").write_bytes(b"fixture checkpoint")
        (run_dir / "last_checkpoint.pt").write_bytes(b"fixture checkpoint")
        (run_dir / "train_log.csv").write_text("epoch,loss\n", encoding="utf-8")
    return run_dir


class E5Scope27HardeningTests(unittest.TestCase):
    def test_active_pointer_is_current_and_manifest_is_27(self):
        pointer = load_active_scope_pointer()
        self.assertEqual(pointer["scope_id"], "e5_batch4_scope27_seed2026")
        manifest = _load_manifest()
        self.assertEqual(manifest["counts"]["total_evidence"], 27)
        self.assertEqual(active_scope_path(pointer, "manifest"), MANIFEST_PATH)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ActiveScopeError):
                load_active_scope_pointer(Path(temp) / "invalid_pointer.json")
            with self.assertRaises(ActiveScopeError):
                active_scope_path({"manifest": "../outside.json"}, "manifest")

    def test_legacy_readiness_builder_is_not_a_default_current_entrypoint(self):
        from benchmark_v2.experiments.e5_common_loss.readiness import build_readiness

        with self.assertRaisesRegex(RuntimeError, "SUPERSEDED_E5_SCOPE29_COMMAND"):
            build_readiness()

    def test_evaluate_only_real_filesystem_artifacts_need_no_checkpoint(self):
        manifest = _load_manifest()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for model_id in ("persistence", "moving_average"):
                run_dir = _make_fixture(root, manifest, model_id)
                from scripts import e5_batch4_scope27_gate as gate

                result = gate.inspect_run(manifest, _entry(manifest, model_id), root)
                self.assertTrue(result["ready"], result)
                self.assertIsNone(result["checkpoint_sha256"])
                self.assertIn("baseline_state_sha256", result)
                self.assertIn("common_loss_diagnostic_sha256", result)
                (run_dir / "best_checkpoint.pt").write_bytes(b"wrong semantic checkpoint")
                again = gate.inspect_run(manifest, _entry(manifest, model_id), root)
                self.assertTrue(again["ready"], again)
                self.assertIsNone(again["checkpoint_sha256"])

    def test_evaluate_only_missing_evidence_and_nonfinite_metrics_fail_closed(self):
        manifest = _load_manifest()
        from scripts import e5_batch4_scope27_gate as gate

        for missing_name in ("baseline_state.json", "common_loss_diagnostic.json"):
            with self.subTest(missing_name=missing_name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                run_dir = _make_fixture(root, manifest, "persistence")
                (run_dir / missing_name).unlink()
                result = gate.inspect_run(manifest, _entry(manifest, "persistence"), root)
                self.assertFalse(result["ready"])
        for bad_value in (None, float("nan"), float("inf")):
            with self.subTest(bad_value=bad_value), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                run_dir = _make_fixture(root, manifest, "moving_average")
                path = run_dir / "metrics_eval_h3.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["Score"] = bad_value
                if bad_value is None:
                    _write_json(path, payload)
                else:
                    path.write_text(
                        json.dumps(payload, allow_nan=True), encoding="utf-8"
                    )
                result = gate.inspect_run(
                    manifest, _entry(manifest, "moving_average"), root
                )
                self.assertFalse(result["ready"])

    def test_source_identity_is_shared_nonempty_and_platform_stable(self):
        manifest = _load_manifest()
        for model_id in ("gru", "persistence", "moving_average"):
            identity = canonical_model_source_identity(model_id)
            entry = _entry(manifest, model_id)
            self.assertTrue(identity["source_closure_files"])
            self.assertEqual(
                identity["canonical_combined_hash"], entry["base_model_source_hash"]
            )
        records_posix = [
            {"path": "custom_models/src/x.py", "sha256": "a" * 64},
            {"path": "custom_models/src/y.py", "sha256": "b" * 64},
        ]
        records_windows = [
            {"path": "custom_models\\src\\x.py", "sha256": "a" * 64},
            {"path": "custom_models\\src\\y.py", "sha256": "b" * 64},
        ]
        self.assertEqual(
            canonical_hash_for_records(records_posix),
            canonical_hash_for_records(records_windows),
        )
        self.assertNotEqual(
            canonical_hash_for_records(records_posix),
            canonical_hash_for_records(
                [dict(records_posix[0]), {"path": "custom_models/src/y.py", "sha256": "c" * 64}]
            ),
        )
        with self.assertRaises(ModelSourceIdentityError):
            canonical_hash_for_records(records_posix + [records_posix[0]])
        with self.assertRaises(ModelSourceIdentityError):
            canonical_hash_for_records(
                [{"path": "C:/outside.py", "sha256": "a" * 64}]
            )

    def test_source_closure_file_changes_and_missing_files_fail_closed(self):
        from benchmark_v2.model_source_identity import canonical_hash_for_records

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            closure_path = root / "closure.py"
            unrelated_path = root / "unrelated.txt"
            closure_path.write_text("value = 1\n", encoding="utf-8")
            unrelated_path.write_text("unrelated = 1\n", encoding="utf-8")

            def closure_hash() -> str:
                resolved = _resolve_repo_file("closure.py", root)
                return canonical_hash_for_records(
                    [
                        {
                            "path": "custom_models/src/closure.py",
                            "sha256": _sha256_file(resolved),
                        }
                    ]
                )

            first = closure_hash()
            closure_path.write_text("value = 2\n", encoding="utf-8")
            second = closure_hash()
            self.assertNotEqual(first, second)
            unrelated_path.write_text("unrelated = 2\n", encoding="utf-8")
            self.assertEqual(second, closure_hash())
            with self.assertRaises(ModelSourceIdentityError):
                _resolve_repo_file("missing.py", root)

    def test_runtime_provenance_and_manifest_identity_match(self):
        manifest = _load_manifest()
        runtime = SimpleNamespace(model_id="gru", effective_config={"model_id": "gru"})
        apply_experiment_profile(runtime, "e5_common_loss_v1")
        provenance = runtime.effective_config["provenance"]
        entry = _entry(manifest, "gru")
        self.assertEqual(
            provenance["base_model_source_closure_hash"],
            entry["base_model_source_hash"],
        )
        self.assertEqual(
            provenance["base_model_source_closure_files"],
            entry["source_identity"]["source_closure_files"],
        )

    def test_precision_policy_is_profile_and_model_bound(self):
        transformer = SimpleNamespace(
            model_id="transformer",
            effective_config={
                "amp_enabled": True,
                "training_batch_profile_id": "uniform_train_batch4_v1",
            },
        )
        apply_model_precision_policy(transformer)
        self.assertFalse(transformer.effective_config["amp_enabled"])
        self.assertEqual(transformer.effective_config["precision_policy"], "fp32")
        self.assertTrue(
            transformer.effective_config["precision_resolution"][
                "requested_amp_enabled"
            ]
        )
        no_profile = SimpleNamespace(
            model_id="transformer", effective_config={"amp_enabled": True}
        )
        apply_model_precision_policy(no_profile)
        self.assertTrue(no_profile.effective_config["amp_enabled"])
        gru = SimpleNamespace(
            model_id="gru",
            effective_config={
                "amp_enabled": True,
                "training_batch_profile_id": "uniform_train_batch4_v1",
            },
        )
        apply_model_precision_policy(gru)
        self.assertTrue(gru.effective_config["amp_enabled"])
        self.assertNotEqual(gru.effective_config["precision_policy"], "fp32")

    def test_gate_accepts_and_rejects_transformer_precision_fixture(self):
        manifest = _load_manifest()
        from scripts import e5_batch4_scope27_gate as gate

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = _make_fixture(root, manifest, "transformer")
            entry = _entry(manifest, "transformer")
            accepted = gate.inspect_run(manifest, entry, root)
            self.assertTrue(accepted["ready"], accepted)
            bad = json.loads((run_dir / "effective_config.json").read_text(encoding="utf-8"))
            bad["amp_enabled"] = True
            bad.pop("precision_policy", None)
            bad.pop("precision_resolution", None)
            _write_json(run_dir / "effective_config.json", bad)
            rejected = gate.inspect_run(manifest, entry, root)
            self.assertFalse(rejected["ready"])
            self.assertIn("PRECISION_IDENTITY_MISMATCH", rejected["reasons"])

    def test_original_scope26_and_e5_scope27_sets_are_consistent(self):
        original = json.loads(
            (
                PROJECT_ROOT
                / "custom_models/docs/benchmark_v2/BATCH4/CURRENT_BATCH4_SCOPE26_MANIFEST.json"
            ).read_text(encoding="utf-8")
        )
        exclusions = json.loads(
            (
                PROJECT_ROOT
                / "custom_models/docs/benchmark_v2/BATCH4/CURRENT_SCOPE_EXCLUSIONS.json"
            ).read_text(encoding="utf-8")
        )
        manifest = _load_manifest()
        registry = load_registry()
        def canonical_model_name(value: str) -> str:
            for item in registry.list():
                if item.canonical_id.casefold() == value.casefold():
                    return item.canonical_id
                if item.display_name.casefold() == value.casefold():
                    return item.canonical_id
            raise KeyError(value)
        original_models = {
            canonical_model_name(model)
            for group in original["groups"]
            for model in group["models"]
        }
        original_eval = {"persistence", "moving_average"}
        e5_train = {
            row["model_id"]
            for row in manifest["entries"]
            if row["entry_type"] == "TRAIN_COMMON_LOSS"
        }
        e5_eval = {
            row["model_id"]
            for row in manifest["entries"]
            if row["entry_type"] == "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC"
        }
        self.assertEqual(e5_train, original_models - original_eval)
        self.assertEqual(e5_eval, original_eval)
        self.assertEqual(
            {row["model_id"] for row in manifest["exclusions"]},
            {item["model_id"] for item in exclusions["exclusions"]},
        )
        self.assertEqual(len(manifest["entries"]), len(original_models) + 1)

    def test_lock_active_and_stale_states_are_explicit(self):
        from scripts import e5_scope27_lock as lock

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scope27_lock.json"
            args = SimpleNamespace(
                scope_id="e5_batch4_scope27_seed2026",
                pid=os.getpid(),
                hostname="test-host",
                started_at="2026-08-03T00:00:00+00:00",
                git_commit="head",
                python_executable="python",
                manifest_sha256="a" * 64,
            )
            self.assertEqual(lock.acquire_lock(path, args), 0)
            self.assertEqual(lock.inspect_lock(path), 0)
            self.assertEqual(lock.release_lock(path, os.getpid()), 0)
            path.write_text(
                json.dumps({"schema_version": lock.LOCK_SCHEMA_VERSION, "pid": 999999}),
                encoding="utf-8",
            )
            self.assertEqual(lock.inspect_lock(path), 74)
            self.assertEqual(lock.clear_stale_lock(path), 0)
            self.assertFalse(path.exists())

    def test_current_launcher_and_cli_do_not_default_to_scope29(self):
        launcher = (
            PROJECT_ROOT
            / "custom_models/docs/benchmark_v2/E5/E5_RUN_ALL_27_BATCH4_LINUX.sh"
        ).read_text(encoding="utf-8")
        autoshutdown = (
            PROJECT_ROOT
            / "custom_models/docs/benchmark_v2/E5/E5_RUN_ALL_27_BATCH4_LINUX_AUTOSHUTDOWN.sh"
        ).read_text(encoding="utf-8")
        cli = (
            PROJECT_ROOT / "custom_models/src/benchmark_v2/cli.py"
        ).read_text(encoding="utf-8")
        for text in (launcher, autoshutdown):
            self.assertNotIn("E5_RUN_ALL_29", text)
            self.assertNotIn("SegRNN", text)
            self.assertNotIn("MSGNet", text)
        self.assertIn("--require-complete", launcher)
        self.assertIn("E5_SCOPE27_VARIANT_MANIFEST.json", launcher)
        self.assertIn("e5_scope27_lock.json", launcher)
        self.assertIn("--legacy-scope29", cli)
        self.assertIn("_load_active_scope27_gate", cli)


if __name__ == "__main__":
    unittest.main()
