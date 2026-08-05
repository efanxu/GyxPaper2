from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_v2.original_scope26 import (
    CURRENT_SCOPE26_ID,
    current_model_config_hash,
    load_current_scope_manifest,
    resolve_source_revision,
)
from benchmark_v2.hardware_preflight import launch_model_suite
from scripts.original_batch4_scope26_gate import (
    CURRENT_RUN_MAP,
    EXCLUDED_IDS,
    HORIZONS,
    REQUIRED_METRICS,
    TRAINABLE_TYPE,
    _plan_action,
    _canonical_json_hash,
    _metrics_bundle_hash,
    build_result_inventory,
    compute_original_freeze,
    inspect_run,
    validate_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: object, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
    )


def _fixture_run(root: Path, manifest: dict, model_id: str) -> Path:
    entry = next(item for item in manifest["entries"] if item["model_id"] == model_id)
    run_dir = root / entry["run_id"]
    run_dir.mkdir(parents=True)
    trainable = entry["entry_type"] == TRAINABLE_TYPE
    precision = entry["precision_identity"]
    effective = {
        "model_id": model_id,
        "run_id": entry["run_id"],
        "run_mode": "formal",
        "artifact_profile": entry["artifact_profile"],
        "formal_training": trainable,
        "output_root": entry["output_root"],
        "seed": 2026,
        "seq_len": 144,
        "pred_len": 10,
        "protocol_hash": manifest["benchmark_protocol_hash"],
        "base_benchmark_protocol_hash": manifest["benchmark_protocol_hash"],
        "training_batch_profile_id": manifest["training_profile_id"],
        "training_batch_profile_hash": manifest["training_profile_hash"],
        "train_batch_size": 4,
        "val_batch_size": 4,
        "test_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "effective_train_batch_size": 4,
        "amp_enabled": precision["amp_enabled"],
        "precision_policy": precision["precision_policy"],
        "precision_resolution": precision["precision_resolution"],
        "active_scope_id": CURRENT_SCOPE26_ID,
        "source_closure_hash": entry["source_identity"]["canonical_combined_hash"],
        "model_config_hash": entry["model_config_identity"]["config_hash"],
        "provenance": {
            "active_scope_id": CURRENT_SCOPE26_ID,
            "base_model_source_closure_hash": entry["source_identity"]["canonical_combined_hash"],
            "base_model_config_hash": entry["model_config_identity"]["config_hash"],
        },
    }
    _write_json(run_dir / "effective_config.json", effective)
    _write_json(run_dir / "resolved_config.json", effective)
    _write_json(
        run_dir / "protocol_check.json",
        {
            "status": "PASS",
            "protocol_hash": manifest["benchmark_protocol_hash"],
            "base_benchmark_protocol_hash": manifest["benchmark_protocol_hash"],
        },
    )
    _write_json(
        run_dir / "data_signature.json",
        {
            "dataset_id": manifest["dataset_identity"]["dataset_id"],
            "feature_order_hash": manifest["dataset_identity"]["feature_order_hash"],
            "node_count": manifest["dataset_identity"]["expected_node_count"],
        },
    )
    _write_json(run_dir / "model_summary.json", {"model_id": model_id})
    _write_json(
        run_dir / "artifact_manifest.json",
        {
            "artifact_profile": entry["artifact_profile"],
            "protocol_hash": manifest["benchmark_protocol_hash"],
            "run_mode": "formal",
            "formal_training": trainable,
            "model_id": model_id,
            "run_id": entry["run_id"],
            "output_root": entry["output_root"],
            **precision,
        },
    )
    _write_json(
        run_dir / "prediction_metadata.json",
        {
            "run_id": entry["run_id"],
            "model_id": model_id,
            "source_checkpoint": "best_checkpoint.pt" if trainable else None,
        },
    )
    for horizon in HORIZONS:
        _write_json(
            run_dir / f"metrics_eval_h{horizon}.json",
            {
                "horizon": horizon,
                "MAE": 1.0,
                "RMSE": 2.0,
                "R2": 0.5,
                "Score": 3.0,
                "valid_target_count": 10,
            },
        )
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["horizon", "MAE", "RMSE", "R2", "Score", "valid_target_count"],
        )
        writer.writeheader()
        for horizon in HORIZONS:
            writer.writerow(
                {
                    "horizon": horizon,
                    "MAE": 1.0,
                    "RMSE": 2.0,
                    "R2": 0.5,
                    "Score": 3.0,
                    "valid_target_count": 10,
                }
            )
    _write_json(
        run_dir / "run_status.json",
        {
            "status": "COMPLETED",
            "run_mode": "formal",
            "artifact_profile": entry["artifact_profile"],
            "formal_training": trainable,
            "exit_code": 0,
        },
    )
    if trainable:
        (run_dir / "best_checkpoint.pt").write_bytes(b"fixture checkpoint")
        (run_dir / "last_checkpoint.pt").write_bytes(b"fixture checkpoint")
        (run_dir / "train_log.csv").write_text("epoch,loss\n1,1\n", encoding="utf-8")
    else:
        _write_json(
            run_dir / "baseline_state.json",
            {"model_id": model_id, "training_mode": "EVALUATE_ONLY", "best_checkpoint": None},
        )
    revision = resolve_source_revision(project_root=PROJECT_ROOT, manifest=manifest)
    manifest_hash = _canonical_json_hash(manifest)
    run_map = json.loads(CURRENT_RUN_MAP.read_text(encoding="utf-8"))
    run_map_hash = _canonical_json_hash(run_map)
    freeze = compute_original_freeze(manifest, run_map=run_map)
    effective_path = run_dir / "effective_config.json"
    effective = json.loads(effective_path.read_text(encoding="utf-8"))
    effective.update(
        {
            "current_scope_manifest_hash": manifest_hash,
            "current_scope_entry_id": entry["entry_id"],
            "source_revision": revision,
        }
    )
    _write_json(effective_path, effective)
    _write_json(run_dir / "resolved_config.json", effective)
    checkpoint = run_dir / "best_checkpoint.pt"
    _write_json(
        run_dir / "execution_receipt.json",
        {
            "schema_version": "original_scope26_execution_receipt_v2",
            "status": "SUCCESS",
            "scope_id": CURRENT_SCOPE26_ID,
            "entry_id": entry["entry_id"],
            "model_id": model_id,
            "run_id": entry["run_id"],
            "output_root": entry["output_root"],
            "entry_type": entry["entry_type"],
            "exit_code": 0,
            "git_commit": revision["git_commit"],
            "source_revision_type": revision["source_revision_type"],
            "source_closure_hash": entry["source_identity"]["canonical_combined_hash"],
            "model_config_hash": entry["model_config_identity"]["config_hash"],
            "precision_identity_hash": _canonical_json_hash(precision),
            "protocol_hash": manifest["benchmark_protocol_hash"],
            "training_profile_id": manifest["training_profile_id"],
            "training_profile_hash": manifest["training_profile_hash"],
            "dataset_identity_hash": _canonical_json_hash(manifest["dataset_identity"]),
            "graph_identity_hash": _canonical_json_hash(manifest["graph_identity"]),
            "manifest_hash": manifest_hash,
            "run_map_hash": run_map_hash,
            "freeze_hash": freeze["freeze_hash"],
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else None,
            "metrics_bundle_hash": _metrics_bundle_hash(run_dir),
            "command": ["fixture", model_id],
            "started_at": "2026-08-03T00:00:00+00:00",
            "finished_at": "2026-08-03T00:00:01+00:00",
        },
    )
    return run_dir


class OriginalScope26Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_current_scope_manifest()

    def test_active_scope_has_exact_26_entries_and_exclusions(self):
        validation = validate_manifest(self.manifest)
        self.assertEqual(validation["errors"], [])
        self.assertEqual(validation["counts"], {"total": 26, "trainable": 24, "evaluate_only": 2, "excluded": 2})
        active = {entry["model_id"] for entry in self.manifest["entries"]}
        self.assertEqual(active & EXCLUDED_IDS, set())
        self.assertEqual(
            {item["model_id"] for item in self.manifest["exclusions"]},
            EXCLUDED_IDS,
        )

    def test_run_map_is_exact_and_transformer_precision_is_frozen(self):
        run_map = json.loads(CURRENT_RUN_MAP.read_text(encoding="utf-8"))
        self.assertEqual(len(run_map["entries"]), 26)
        self.assertEqual(len({entry["run_id"] for entry in run_map["entries"]}), 26)
        transformer = next(entry for entry in self.manifest["entries"] if entry["model_id"] == "transformer")
        self.assertEqual(transformer["run_id"], "Transformer_node_shared_d512_bs4_seed2026")
        self.assertFalse(transformer["precision_identity"]["amp_enabled"])
        self.assertEqual(transformer["precision_identity"]["precision_policy"], "fp32")
        self.assertEqual(
            current_model_config_hash("transformer"),
            transformer["model_config_identity"]["config_hash"],
        )

    def test_valid_trainable_fixture_is_accepted(self):
        entry = next(item for item in self.manifest["entries"] if item["model_id"] == "gru")
        with tempfile.TemporaryDirectory() as temp:
            run_dir = _fixture_run(Path(temp), self.manifest, "gru")
            result = inspect_run(self.manifest, entry, run_dir.parent)
            self.assertTrue(result["ready"], result)
            self.assertEqual(result["checkpoint_sha256"], __import__("hashlib").sha256(b"fixture checkpoint").hexdigest())

    def test_completed_metrics_null_nan_inf_and_missing_checkpoint_fail_closed(self):
        entry = next(item for item in self.manifest["entries"] if item["model_id"] == "gru")
        for bad_value in (None, float("nan"), float("inf")):
            with self.subTest(bad_value=bad_value), tempfile.TemporaryDirectory() as temp:
                run_dir = _fixture_run(Path(temp), self.manifest, "gru")
                path = run_dir / "metrics_eval_h3.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["Score"] = bad_value
                _write_json(path, payload, allow_nan=True)
                result = inspect_run(self.manifest, entry, run_dir.parent)
                self.assertFalse(result["ready"])
                self.assertTrue(any("METRICS_H3" in reason for reason in result["reasons"]))
        with tempfile.TemporaryDirectory() as temp:
            run_dir = _fixture_run(Path(temp), self.manifest, "gru")
            (run_dir / "best_checkpoint.pt").unlink()
            result = inspect_run(self.manifest, entry, run_dir.parent)
            self.assertFalse(result["ready"])
            self.assertIn("BEST_CHECKPOINT_MISSING_OR_EMPTY", result["reasons"])

    def test_evaluate_only_fixture_needs_baseline_but_no_checkpoint(self):
        entry = next(item for item in self.manifest["entries"] if item["model_id"] == "persistence")
        with tempfile.TemporaryDirectory() as temp:
            run_dir = _fixture_run(Path(temp), self.manifest, "persistence")
            self.assertFalse((run_dir / "best_checkpoint.pt").exists())
            result = inspect_run(self.manifest, entry, run_dir.parent)
            self.assertTrue(result["ready"], result)
            (run_dir / "baseline_state.json").unlink()
            result = inspect_run(self.manifest, entry, run_dir.parent)
            self.assertFalse(result["ready"])

    def test_wrong_source_precision_batch_and_renamed_run_are_rejected(self):
        entry = next(item for item in self.manifest["entries"] if item["model_id"] == "transformer")
        with tempfile.TemporaryDirectory() as temp:
            run_dir = _fixture_run(Path(temp), self.manifest, "transformer")
            effective_path = run_dir / "effective_config.json"
            effective = json.loads(effective_path.read_text(encoding="utf-8"))
            effective["source_closure_hash"] = "0" * 64
            effective["provenance"]["base_model_source_closure_hash"] = "0" * 64
            effective["precision_policy"] = "profile_default"
            effective["amp_enabled"] = True
            effective_path.write_text(json.dumps(effective), encoding="utf-8")
            result = inspect_run(self.manifest, entry, run_dir.parent)
            self.assertFalse(result["ready"])
            self.assertIn("SOURCE_CLOSURE_IDENTITY_MISMATCH", result["reasons"])
            self.assertIn("PRECISION_IDENTITY_MISMATCH:amp_enabled", result["reasons"])

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            renamed = _fixture_run(root, self.manifest, "gru")
            renamed_target = root / "renamed_copy"
            renamed.rename(renamed_target)
            action, inspected = _plan_action(self.manifest, next(item for item in self.manifest["entries"] if item["model_id"] == "gru"), root)
            self.assertEqual(action, "BLOCK_EXISTING_IDENTITY_MISMATCH")
            self.assertIn("RUN_ID_RENAMED_OR_NONCANONICAL", inspected["reasons"])

    def test_freeze_is_current_original_only(self):
        freeze = compute_original_freeze(self.manifest)
        self.assertEqual(freeze["scope_id"], CURRENT_SCOPE26_ID)
        revision = resolve_source_revision(project_root=PROJECT_ROOT, manifest=self.manifest)
        self.assertEqual(freeze["git_commit"], revision["git_commit"])
        self.assertEqual(freeze["source_revision_type"], revision["source_revision_type"])
        material_text = json.dumps(freeze["freeze_material"], ensure_ascii=False)
        self.assertNotIn("E5_SCOPE27_VARIANT_MANIFEST", material_text)
        self.assertNotIn("segrnn", material_text.casefold())
        self.assertNotIn("msgnet", material_text.casefold())
        self.assertIn("active_gate_revision", freeze["freeze_material"])
        self.assertIn("active_launcher_revision", freeze["freeze_material"])
        self.assertIn("checkout_identity", freeze["freeze_material"])

        changed_map = copy.deepcopy(json.loads(CURRENT_RUN_MAP.read_text(encoding="utf-8")))
        changed_map["entries"][0]["run_id"] += "_changed"
        changed = compute_original_freeze(self.manifest, run_map=changed_map)
        self.assertNotEqual(freeze["freeze_hash"], changed["freeze_hash"])
        changed_dataset = copy.deepcopy(self.manifest)
        changed_dataset["dataset_identity"]["feature_order_hash"] = "f" * 64
        self.assertNotEqual(
            freeze["freeze_hash"], compute_original_freeze(changed_dataset)["freeze_hash"]
        )

    def test_inventory_excludes_segrnn_and_records_orphans(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _fixture_run(root, self.manifest, "gru")
            excluded = root / "SegRNN_node_shared_seg2_d512_bs4_seed2026"
            excluded.mkdir()
            _write_json(excluded / "effective_config.json", {"model_id": "segrnn"})
            inventory = build_result_inventory(self.manifest, root=root)
            self.assertEqual(inventory["counts"]["CURRENT_SCOPE26_REUSABLE"], 1)
            self.assertEqual(inventory["counts"]["EXCLUDED_MODEL_HISTORICAL"], 1)

    def test_failure_continuation_calls_all_26_and_preserves_failure_code(self):
        calls: list[str] = []

        def launcher(**request):
            calls.append(request["model_id"])
            return 19 if len(calls) == 3 else 0

        requests = [{"model_id": f"model_{index}"} for index in range(26)]
        results = launch_model_suite(requests, launcher=launcher)
        self.assertEqual(len(calls), 26)
        self.assertEqual(len(results), 26)
        self.assertEqual(results[2]["exit_code"], 19)
        self.assertEqual(results[-1]["model_id"], "model_25")


if __name__ == "__main__":
    unittest.main()
