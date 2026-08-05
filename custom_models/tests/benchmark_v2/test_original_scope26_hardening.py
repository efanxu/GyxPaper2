from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_v2.model_source_identity import canonical_model_source_identity
from benchmark_v2.hardware_preflight import (
    preflight_artifact_path,
    preflight_identity,
    read_matching_pass,
)
from benchmark_v2.original_scope26 import load_current_scope_manifest, resolve_source_revision
from scripts import original_batch4_scope26_gate as gate
from scripts.original_batch4_scope26_gate import (
    CURRENT_RUN_MAP,
    CURRENT_SCOPE26_ID,
    QUARANTINE_DIRECTORY,
    _canonical_json_hash,
    _failure_diagnostics,
    _lock_status,
    _plan_action,
    archive_existing_attempt,
    build_readiness,
    compute_original_freeze,
    inspect_run,
    quarantine_existing,
)

from .test_original_scope26 import _fixture_run, _write_json


class OriginalScope26HardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_current_scope_manifest()
        cls.run_map = json.loads(CURRENT_RUN_MAP.read_text(encoding="utf-8"))
        cls.revision = resolve_source_revision(manifest=cls.manifest)
        cls.freeze = compute_original_freeze(cls.manifest, run_map=cls.run_map)

    def _entry(self, model_id: str = "transformer"):
        return next(item for item in self.manifest["entries"] if item["model_id"] == model_id)

    def _minimal_canonical(self, root: Path, model_id: str = "transformer", *, status: str = "FAILED") -> Path:
        entry = self._entry(model_id)
        run_dir = root / entry["run_id"]
        run_dir.mkdir(parents=True)
        source_hash = entry["source_identity"]["canonical_combined_hash"]
        config_hash = entry["model_config_identity"]["config_hash"]
        effective = {
            "model_id": model_id,
            "run_id": entry["run_id"],
            "output_root": entry["output_root"],
            "run_mode": "formal",
            "artifact_profile": entry["artifact_profile"],
            "formal_training": True,
            "seed": 2026,
            "seq_len": 144,
            "pred_len": 10,
            "active_scope_id": CURRENT_SCOPE26_ID,
            "current_scope_manifest_hash": _canonical_json_hash(self.manifest),
            "current_scope_entry_id": entry["entry_id"],
            "source_revision": self.revision,
            "source_closure_hash": source_hash,
            "model_config_hash": config_hash,
            "training_batch_profile_id": self.manifest["training_profile_id"],
            "training_batch_profile_hash": self.manifest["training_profile_hash"],
            "train_batch_size": 4,
            "val_batch_size": 4,
            "test_batch_size": 4,
            "effective_train_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "amp_enabled": entry["precision_identity"]["amp_enabled"],
            "precision_policy": entry["precision_identity"]["precision_policy"],
            "precision_resolution": entry["precision_identity"].get("precision_resolution"),
            "provenance": {
                "active_scope_id": CURRENT_SCOPE26_ID,
                "base_model_source_closure_hash": source_hash,
                "base_model_config_hash": config_hash,
            },
        }
        _write_json(run_dir / "effective_config.json", effective)
        _write_json(run_dir / "run_status.json", {"status": status, "exit_code": 19})
        _write_json(run_dir / "artifact_manifest.json", {"run_id": entry["run_id"], "output_root": entry["output_root"]})
        _write_json(run_dir / "prediction_metadata.json", {"run_id": entry["run_id"], "model_id": model_id})
        _write_json(run_dir / "protocol_check.json", {"protocol_hash": self.manifest["benchmark_protocol_hash"]})
        _write_json(
            run_dir / "data_signature.json",
            {
                "dataset_id": self.manifest["dataset_identity"]["dataset_id"],
                "feature_order_hash": self.manifest["dataset_identity"]["feature_order_hash"],
                "node_count": self.manifest["dataset_identity"]["expected_node_count"],
            },
        )
        return run_dir

    def test_allowlisted_retry2_does_not_block_missing_canonical(self):
        entry = self._entry()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            retry = root / "Transformer_node_shared_d512_fp32_bs4_seed2026_retry2"
            _write_json(retry / "effective_config.json", {"model_id": "transformer"})
            action, inspected = _plan_action(
                self.manifest,
                entry,
                root,
                current_freeze=self.freeze,
                current_revision=self.revision,
                current_run_map=self.run_map,
            )
            self.assertEqual(action, "RUN_MISSING")
            self.assertFalse(inspected.get("found"))
            self.assertTrue(retry.is_dir())

    def test_allowlisted_legacy_fp32_transformer_does_not_enter_denominator_or_block(self):
        entry = self._entry()
        self.assertEqual(self.manifest["counts"]["total"], 26)
        self.assertEqual(self.manifest["counts"]["trainable"], 24)
        self.assertEqual(self.manifest["counts"]["evaluate_only"], 2)
        allowlist = gate._historical_allowlist(self.manifest)
        old_run_id = "Transformer_node_shared_d512_fp32_bs4_seed2026"
        retry_run_id = "Transformer_node_shared_d512_fp32_bs4_seed2026_retry2"
        self.assertIn(("transformer", old_run_id), allowlist)
        self.assertIn(("transformer", retry_run_id), allowlist)
        self.assertEqual(
            [item["run_id"] for item in self.manifest["entries"] if item["model_id"] == "transformer"],
            ["Transformer_node_shared_d512_bs4_seed2026"],
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = root / old_run_id
            retry = root / retry_run_id
            _write_json(old / "effective_config.json", {"model_id": "transformer", "legacy": True})
            _write_json(retry / "effective_config.json", {"model_id": "transformer", "retry2": True})
            old_before = (old / "effective_config.json").read_text(encoding="utf-8")
            retry_before = (retry / "effective_config.json").read_text(encoding="utf-8")
            action, inspected = _plan_action(
                self.manifest,
                entry,
                root,
                current_freeze=self.freeze,
                current_revision=self.revision,
                current_run_map=self.run_map,
            )
            self.assertEqual(action, "RUN_MISSING")
            self.assertFalse(inspected.get("found"))
            self.assertEqual((old / "effective_config.json").read_text(encoding="utf-8"), old_before)
            self.assertEqual((retry / "effective_config.json").read_text(encoding="utf-8"), retry_before)

    def test_source_revision_supports_explicit_no_git_and_rejects_wrong_value(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = {"source_revision_policy": {"declared_source_revision": "manual-revision"}}
            resolved = resolve_source_revision(
                project_root=root,
                explicit_source_revision="manual-revision",
                manifest=manifest,
            )
            self.assertEqual(resolved["source_revision_type"], "explicit_no_git")
            with self.assertRaises(ValueError):
                resolve_source_revision(
                    project_root=root,
                    explicit_source_revision="wrong-revision",
                    manifest=manifest,
                )

    def test_undeclared_renamed_copy_blocks_and_retry2_is_not_modified(self):
        entry = self._entry()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            renamed = root / "transformer_renamed_copy"
            _write_json(renamed / "effective_config.json", {"model_id": "transformer"})
            action, inspected = _plan_action(
                self.manifest,
                entry,
                root,
                current_freeze=self.freeze,
                current_revision=self.revision,
                current_run_map=self.run_map,
            )
            self.assertEqual(action, "BLOCK_EXISTING_IDENTITY_MISMATCH")
            self.assertIn("RUN_ID_RENAMED_OR_NONCANONICAL", inspected["reasons"])
            self.assertEqual(json.loads((renamed / "effective_config.json").read_text()), {"model_id": "transformer"})

    def test_failed_canonical_is_archived_atomically(self):
        entry = self._entry()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._minimal_canonical(root)
            result = archive_existing_attempt(
                self.manifest,
                entry,
                root,
                current_freeze=self.freeze,
                current_revision=self.revision,
                current_run_map=self.run_map,
            )
            target = Path(result["target"])
            self.assertEqual(result["status"], "ARCHIVED")
            self.assertFalse(source.exists())
            self.assertTrue((target / "archive_receipt.json").is_file())
            receipt = json.loads((target / "archive_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["scope_id"], CURRENT_SCOPE26_ID)
            self.assertEqual(receipt["run_id"], entry["run_id"])
            self.assertIn("recursive_file_manifest", receipt)

    def test_failed_persistence_attempt_is_archived_before_fresh_retry(self):
        entry = self._entry("persistence")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            failed = _fixture_run(root, self.manifest, "persistence")
            _write_json(
                failed / "run_status.json",
                {
                    "status": "FAILED",
                    "exit_code": 2,
                    "run_mode": "formal",
                    "artifact_profile": "NON_TRAINABLE",
                    "formal_training": False,
                },
            )
            action, _ = _plan_action(
                self.manifest,
                entry,
                root,
                current_freeze=self.freeze,
                current_revision=self.revision,
                current_run_map=self.run_map,
            )
            self.assertEqual(action, "ARCHIVE_INCOMPLETE_THEN_RUN")
            archived = archive_existing_attempt(
                self.manifest,
                entry,
                root,
                current_freeze=self.freeze,
                current_revision=self.revision,
                current_run_map=self.run_map,
            )
            archived_dir = Path(archived["target"])
            self.assertFalse(failed.exists())
            self.assertEqual(
                json.loads(
                    (archived_dir / "run_status.json").read_text(encoding="utf-8")
                )["status"],
                "FAILED",
            )
            retry = _fixture_run(root, self.manifest, "persistence")
            self.assertTrue(retry.is_dir())
            self.assertNotEqual(retry.resolve(), archived_dir.resolve())

    def test_archive_failure_keeps_source_and_identity_mismatch_is_not_auto_archived(self):
        entry = self._entry()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._minimal_canonical(root)
            with patch.object(gate.os, "replace", side_effect=OSError("move blocked")):
                with self.assertRaises(OSError):
                    archive_existing_attempt(
                        self.manifest,
                        entry,
                        root,
                        current_freeze=self.freeze,
                        current_revision=self.revision,
                        current_run_map=self.run_map,
                    )
            self.assertTrue(source.is_dir())
            effective_path = source / "effective_config.json"
            effective = json.loads(effective_path.read_text(encoding="utf-8"))
            effective["source_closure_hash"] = "0" * 64
            _write_json(effective_path, effective)
            action, _ = _plan_action(
                self.manifest,
                entry,
                root,
                current_freeze=self.freeze,
                current_revision=self.revision,
                current_run_map=self.run_map,
            )
            self.assertEqual(action, "BLOCK_EXISTING_IDENTITY_MISMATCH")
            self.assertTrue(source.is_dir())

    def test_archive_retry_after_move_failure_keeps_source_and_records_intent(self):
        entry = self._entry()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._minimal_canonical(root)
            real_replace = os.replace

            def fail_only_for_canonical_move(source_path, target_path):
                if Path(source_path).resolve() == source.resolve():
                    raise OSError("move blocked once")
                return real_replace(source_path, target_path)

            with patch.object(gate.os, "replace", side_effect=fail_only_for_canonical_move):
                with self.assertRaises(OSError):
                    archive_existing_attempt(
                        self.manifest,
                        entry,
                        root,
                        current_freeze=self.freeze,
                        current_revision=self.revision,
                        current_run_map=self.run_map,
                    )
            self.assertTrue(source.is_dir())
            intents = list(
                (root / gate.ARCHIVE_DIRECTORY / gate.INTENT_DIRECTORY).glob("*.json")
            )
            self.assertEqual(len(intents), 1)
            self.assertEqual(json.loads(intents[0].read_text(encoding="utf-8"))["status"], "FAILED")

            result = archive_existing_attempt(
                self.manifest,
                entry,
                root,
                current_freeze=self.freeze,
                current_revision=self.revision,
                current_run_map=self.run_map,
            )
            self.assertEqual(result["status"], "ARCHIVED")
            self.assertFalse(source.exists())
            self.assertTrue(Path(result["receipt"]).is_file())
            statuses = {
                json.loads(path.read_text(encoding="utf-8"))["status"]
                for path in intents
            }
            statuses.update(
                json.loads(path.read_text(encoding="utf-8"))["status"]
                for path in (root / gate.ARCHIVE_DIRECTORY / gate.INTENT_DIRECTORY).glob("*.json")
            )
            self.assertEqual(statuses, {"FAILED", "COMPLETED"})

    def test_quarantine_requires_apply(self):
        entry = self._entry()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._minimal_canonical(root)
            effective_path = source / "effective_config.json"
            effective = json.loads(effective_path.read_text(encoding="utf-8"))
            effective["source_closure_hash"] = "0" * 64
            _write_json(effective_path, effective)
            preview = quarantine_existing(
                self.manifest,
                entry,
                root,
                apply=False,
                current_freeze=self.freeze,
                current_revision=self.revision,
                current_run_map=self.run_map,
            )
            self.assertEqual(preview["status"], "READY_TO_APPLY")
            self.assertTrue(source.exists())
            self.assertFalse((root / QUARANTINE_DIRECTORY).exists())
            applied = quarantine_existing(
                self.manifest,
                entry,
                root,
                apply=True,
                current_freeze=self.freeze,
                current_revision=self.revision,
                current_run_map=self.run_map,
            )
            self.assertEqual(applied["status"], "QUARANTINED")
            self.assertFalse(source.exists())

    def test_receipt_and_artifact_bindings_fail_closed(self):
        entry = self._entry("gru")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = _fixture_run(root, self.manifest, "gru")
            valid = inspect_run(
                self.manifest,
                entry,
                root,
                current_freeze=self.freeze,
                current_revision=self.revision,
                current_run_map=self.run_map,
            )
            self.assertTrue(valid["ready"], valid)
            receipt_path = run_dir / "execution_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            for field, bad in (
                ("manifest_hash", "0" * 64),
                ("freeze_hash", "1" * 64),
                ("git_commit", "wrong"),
                ("output_root", "wrong-root"),
                ("metrics_bundle_hash", "2" * 64),
            ):
                mutated = copy.deepcopy(receipt)
                mutated[field] = bad
                _write_json(receipt_path, mutated)
                rejected = inspect_run(
                    self.manifest,
                    entry,
                    root,
                    current_freeze=self.freeze,
                    current_revision=self.revision,
                    current_run_map=self.run_map,
                )
                self.assertFalse(rejected["ready"], field)
                _write_json(receipt_path, receipt)
            checkpoint = run_dir / "best_checkpoint.pt"
            checkpoint.write_bytes(b"replaced checkpoint")
            rejected = inspect_run(
                self.manifest,
                entry,
                root,
                current_freeze=self.freeze,
                current_revision=self.revision,
                current_run_map=self.run_map,
            )
            self.assertFalse(rejected["ready"])

    def test_failure_diagnostics_extract_real_exception_categories(self):
        cases = {
            "cuda": "RuntimeError: CUDA out of memory.\n",
            "nonfinite": "ContractError: prediction contains NaN/Inf\n",
            "collision": "FileExistsError: canonical directory already exists\n",
            "identity": "Scope26GateError: source/config identity mismatch\n",
            "unknown": "ValueError: unexpected failure\n",
        }
        with tempfile.TemporaryDirectory() as temp:
            for name, text in cases.items():
                path = Path(temp) / f"{name}.log"
                path.write_text(text, encoding="utf-8")
                diagnostic = _failure_diagnostics(path, exit_code=2)
                self.assertTrue(diagnostic["error_type"])
                if name == "cuda":
                    self.assertTrue(diagnostic["oom"])
                if name == "nonfinite":
                    self.assertTrue(diagnostic["nonfinite"])
                if name == "collision":
                    self.assertEqual(diagnostic["error_type"], "FileExistsError")
                if name == "identity":
                    self.assertTrue(diagnostic["identity_failure"])

    def test_lock_payload_statuses_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scope26.lock"
            payload = {
                "schema_version": "original_scope26_lock_v1",
                "scope_id": CURRENT_SCOPE26_ID,
                "hostname": "remote-host",
                "pid": 999999,
                "process_start_time": 1.0,
                "git_commit": "head",
                "manifest_hash": "a" * 64,
                "freeze_hash": "b" * 64,
                "created_at": "2026-08-03T00:00:00+00:00",
            }
            _write_json(path, payload)
            with patch.object(gate.socket, "gethostname", return_value="local-host"):
                self.assertEqual(_lock_status(path)["status"], "UNKNOWN_REMOTE")
            payload["hostname"] = "local-host"
            _write_json(path, payload)
            with patch.object(gate.socket, "gethostname", return_value="local-host"), patch.object(gate, "_pid_alive", return_value=False):
                self.assertEqual(_lock_status(path)["status"], "STALE")
            path.write_text("not json\n", encoding="utf-8")
            self.assertEqual(_lock_status(path)["status"], "MALFORMED")

    def test_transformer_transitive_closure_contains_upstream_layers(self):
        identity = canonical_model_source_identity("transformer")
        paths = {row["path"] for row in identity["source_closure_files"]}
        self.assertTrue(
            {
                "Time-Series-Library/models/Transformer.py",
                "Time-Series-Library/layers/Transformer_EncDec.py",
                "Time-Series-Library/layers/SelfAttention_Family.py",
                "Time-Series-Library/layers/Embed.py",
            }.issubset(paths)
        )

    def test_windows_formal_command_file_is_explicit_and_non_destructive(self):
        path = (
            gate.PROJECT_ROOT
            / "custom_models/docs/benchmark_v2/BATCH4/"
            "ORIGINAL_SCOPE26_WINDOWS_FORMAL_COMMANDS.ps1"
        )
        text = path.read_text(encoding="utf-8")
        self.assertIn("StaticAudit", text)
        self.assertIn("Exact Original GPU preflight PASS: 24/24", text)
        self.assertIn("COMPLETED_READY_26_OF_26", text)
        self.assertIn("origin/main", text)
        self.assertIn("$Launcher", text)
        self.assertIn("$Head", text)
        self.assertIn("$env:PYTHONUTF8", text)
        self.assertIn("Invoke-GyxPythonGate", text)
        self.assertIn("-AllowedExitCodes @(0, 1, 4)", text)
        self.assertIn("final_status.json", text)
        self.assertIn("FORMAL FAILURE", text)
        self.assertIn("Formal Original suite completed with preserved per-model failures", text)
        self.assertIn("Formal Original suite completed but is NOT_READY", text)
        self.assertIn("GraphSourceStatus", text)
        self.assertIn("GraphSourceRepair", text)
        self.assertIn("graph-source-status", text)
        self.assertIn("graph-source-repair", text)
        self.assertIn("-Action QuarantineExisting -Model '$model' -Apply", text)
        self.assertIn("RequireComplete", text)
        self.assertNotIn("Remove-Item -Recurse", text)
        self.assertNotIn("quarantine-existing --model $model --apply", text)
        self.assertNotIn("shutdown.exe", text.casefold())
        self.assertNotIn("stop-computer", text.casefold())

    def test_single_preflight_marks_no_cuda_not_run_without_child_launch(self):
        with tempfile.TemporaryDirectory() as temp, patch(
            "torch.cuda.is_available", return_value=False
        ):
            code, payload = gate.run_single_preflight(
                "gcn",
                self.manifest,
                preflight_root=Path(temp),
                child_log_root=Path(temp) / "child_logs",
            )
            self.assertEqual(code, 4)
            self.assertEqual(payload["status"], "NOT_RUN")
            self.assertFalse(payload["gpu_preflight_performed"])
            self.assertFalse(payload["denominator_counted"])
            self.assertFalse(list(Path(temp).glob("**/attempts/*/result.json")))

    def test_old_revision_preflight_pass_is_not_reusable(self):
        with tempfile.TemporaryDirectory() as temp:
            identity = preflight_identity(
                "gru",
                training_profile="uniform_train_batch4_v1",
                formal_scope_id=CURRENT_SCOPE26_ID,
                source_revision=self.revision["git_commit"],
                manifest=self.manifest,
                run_map=self.run_map,
                freeze=self.freeze,
            )
            payload = {
                **identity,
                "status": "PASS",
                "forward_completed": True,
                "backward_completed": True,
                "finite_prediction": True,
                "finite_loss": True,
                "finite_gradients": True,
                "prediction_shape": [identity["B"], identity["N"], identity["H"]],
                "peak_allocated_memory": 0,
                "peak_reserved_memory": 0,
            }
            payload["git_commit"] = "0" * 40
            path = preflight_artifact_path("gru", root=temp)
            _write_json(path, payload)
            self.assertIsNone(
                read_matching_pass(
                    "gru",
                    root=temp,
                    training_profile="uniform_train_batch4_v1",
                    formal_scope_id=CURRENT_SCOPE26_ID,
                    source_revision=self.revision["git_commit"],
                    manifest=self.manifest,
                    run_map=self.run_map,
                    freeze=self.freeze,
                )
            )


if __name__ == "__main__":
    unittest.main()
