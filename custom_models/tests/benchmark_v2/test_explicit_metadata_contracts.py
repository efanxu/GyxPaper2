from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_v2.data.signatures import data_signature
from benchmark_v2.hardware_preflight import preflight_identity, read_matching_pass
from scripts import e5_batch4_scope27_gate as e5_gate
from scripts import original_batch4_scope26_gate as original_gate
from scripts import st_mgprompt_a8_batch4_gate as a8_gate


def assert_clean_keys(test: unittest.TestCase, value: object) -> None:
    forbidden = ("ha" + "sh", "sha" + "256", "dig" + "est", "finger" + "print")
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            test.assertFalse(any(term in lowered for term in forbidden), key)
            assert_clean_keys(test, child)
    elif isinstance(value, list):
        for child in value:
            assert_clean_keys(test, child)


class ExplicitMetadataContractTests(unittest.TestCase):
    def test_data_metadata_is_explicit(self) -> None:
        value = data_signature(
            dataset_id="sdwpf",
            input_path="input.parquet",
            target_path="target.parquet",
            feature_names=["wind", "power"],
            node_ids=[1, 2],
            timestamps=["t0", "t1", "t2"],
            shape=(3, 2, 2),
            scalers_fit_split="train",
            split={"train": [0, 2]},
            stride=6,
        )
        self.assertEqual(value["feature_names"], ["wind", "power"])
        self.assertEqual(value["node_count"], 2)
        self.assertEqual(value["shape"], [3, 2, 2])
        self.assertEqual(value["split"], {"train": [0, 2]})
        assert_clean_keys(self, value)

    def test_original_completed_missing_resume_and_conflict(self) -> None:
        manifest = original_gate.load_current_scope_manifest(original_gate.CURRENT_MANIFEST)
        entry = next(row for row in manifest["entries"] if row["entry_type"] == "EVALUATE_ONLY")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = original_gate.inspect_run(manifest, entry, root)
            self.assertEqual(original_gate._plan_action(entry, missing), "RUN_MISSING")
            run_dir = root / entry["run_id"]
            run_dir.mkdir()
            legacy_key = "legacy_" + "ha" + "sh"
            (run_dir / "run_status.json").write_text(json.dumps({"status": "COMPLETED", "exit_code": 0, legacy_key: "ignored"}), encoding="utf-8")
            (run_dir / "effective_config.json").write_text(json.dumps({"model_id": entry["model_id"], legacy_key: "ignored"}), encoding="utf-8")
            for horizon in (3, 6, 10):
                payload = {"horizon": horizon, "MAE": 1.0, "RMSE": 1.0, "R2": 0.0, "Score": 1.0, "valid_target_count": 1}
                (run_dir / f"metrics_eval_h{horizon}.json").write_text(json.dumps(payload), encoding="utf-8")
            completed = original_gate.inspect_run(manifest, entry, root)
            self.assertEqual(original_gate._plan_action(entry, completed), "SKIP_COMPLETED")
            (run_dir / "effective_config.json").write_text(json.dumps({"model_id": "different"}), encoding="utf-8")
            conflict = original_gate.inspect_run(manifest, entry, root)
            self.assertEqual(original_gate._plan_action(entry, conflict), "BLOCK_EXPLICIT_CONFIG_CONFLICT")
            stale = {**conflict, "explicit_config_conflicts": [], "pid": 99999999, "status": "RUNNING"}
            self.assertEqual(original_gate._plan_action(entry, stale), "ARCHIVE_AND_RUN")

    def test_preflight_reuse_uses_explicit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = preflight_identity("gcn", formal_scope_id=original_gate.CURRENT_SCOPE26_ID)
            path = root / "gcn" / "latest.json"
            path.parent.mkdir(parents=True)
            payload = {
                **expected,
                "status": "PASS",
                "device": "cuda",
                "forward_pass": True,
                "backward_pass": True,
                "finite": True,
                "output_shape": [4, 134, 10],
                "created_at": "2026-08-06T00:00:00Z",
                "legacy_" + "ha" + "sh": "ignored",
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            first = read_matching_pass("gcn", root=root, formal_scope_id=original_gate.CURRENT_SCOPE26_ID, source_revision="old")
            second = read_matching_pass("gcn", root=root, formal_scope_id=original_gate.CURRENT_SCOPE26_ID, source_revision="new")
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)

    def test_scope_manifests_and_snapshots(self) -> None:
        original = original_gate.load_current_scope_manifest(original_gate.CURRENT_MANIFEST)
        original_result = original_gate.validate_manifest(original)
        self.assertEqual(original_result["counts"], {"trainable": 24, "evaluate_only": 2, "total": 26})
        self.assertEqual(set(original_result["excluded"]), {"segrnn", "msgnet"})
        e5 = e5_gate.load_manifest(e5_gate.DEFAULT_MANIFEST)
        e5_result = e5_gate.validate_manifest(e5)
        self.assertEqual(e5_result["counts"], {"trainable": 24, "evaluate_only": 2, "a8_reference": 1, "total": 27})
        self.assertEqual(e5_result["loss_id"], "masked_score_aligned_hybrid")
        a8 = a8_gate.validate_contract()
        self.assertEqual(a8["batch_size"], 4)
        for payload in (original_gate.compute_original_freeze(original), e5_gate.compute_e5_freeze(e5), a8_gate.build_freeze_plan()):
            assert_clean_keys(self, payload)


if __name__ == "__main__":
    unittest.main()
