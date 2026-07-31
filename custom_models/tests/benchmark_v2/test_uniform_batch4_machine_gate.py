import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_v2.hardware_preflight import preflight_artifact_path
from scripts.uniform_batch4_machine_gate import (
    ORIGINAL_SKIP_MSGNET_EXPECTED,
    ORIGINAL_SKIP_MSGNET_MODEL,
    ORIGINAL_SKIP_MSGNET_REASON,
    ORIGINAL_SKIP_MSGNET_SUITE_STATUS,
    TRAINABLE_MODELS,
    classify_formal_resume_evidence,
    e5_core_allowed,
    e5_finalize_blockers,
    freeze_identities_match,
    machine_identities_match,
    msgnet_has_fail_oom_evidence,
    original_skip_msgnet_preflight_blockers,
    original_skip_msgnet_run_rows,
    parallel_execution_allowed,
    partial_original_execution_metadata,
    preflight_kind_matches,
    status_from_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROFILE_ID = "uniform_train_batch4_v1"


def _will_run_resume_plan() -> list[dict]:
    return [
        {
            "model_id": row["model_id"],
            "run_id": row["new_run_id"],
            "run_root": f"missing/{row['new_run_id']}",
            "decision": "WILL_RUN",
            "reason": "test plan",
        }
        for row in original_skip_msgnet_run_rows()
    ]


def _first_six_skip_resume_plan() -> list[dict]:
    completed = {
        "persistence",
        "moving_average",
        "gru",
        "dlinear",
        "lightts",
        "tide",
    }
    plan = _will_run_resume_plan()
    for item in plan:
        if item["model_id"] in completed:
            item["decision"] = "SKIP_COMPLETED"
            item["reason"] = "completed identity matched"
    return plan


def _completed_resume_evidence() -> dict:
    return {
        "run_root_exists": True,
        "run_status": "COMPLETED",
        "identity_mismatches": [],
        "active_run": False,
        "required_files_missing": [],
        "artifact_validation": {"status": "PASS"},
    }


def _freeze() -> dict:
    return {
        "git_commit": "same-commit",
        "base_benchmark_protocol_hash": "protocol",
        "uniform_batch4_protocol_hash": "batch4-protocol",
        "training_batch_profile_hash": (
            "f58bbc161dfba0f00774879fdf2f78ec7c59a7faff9ef28a1733a2ed6835fe66"
        ),
        "graph_protocol_hash": "graph",
        "graph_bundle_hash": "bundle",
        "node_order_hash": "nodes",
        "dataset_hashes": {"data": {"sha256": "dataset"}},
        "run_id_map_hash": "run-map",
        "original_model_identities_hash": "original-models",
        "e5_model_identities_hash": "e5-models",
        "protected_files_hash": "protected",
        "seed": 2026,
        "amp": True,
        "B": 4,
        "T": 144,
        "N": 134,
        "C": 16,
        "H": 10,
    }


class UniformBatch4MachineGateTests(unittest.TestCase):
    def test_gtx1060_fail_does_not_block_new_machine_preflight(self):
        new_machine = {
            "machine_id": "NEW_TARGET",
            "hostname": "target",
            "operating_system": "Linux",
            "python_version": "3.11",
            "pytorch_version": "2.x",
            "cuda_version": "12.x",
            "gpu_name": "Target GPU",
            "gpu_uuid": "GPU-new",
            "gpu_total_memory": 48 * 1024**3,
            "driver_version": "driver",
        }
        with patch(
            "benchmark_v2.hardware_preflight.machine_identity",
            return_value=new_machine,
        ):
            path = preflight_artifact_path(
                "segrnn",
                root=PROJECT_ROOT
                / "custom_models/results_smoke/benchmark_v2_uniform_bs4"
                / "hardware_preflight",
                training_profile=PROFILE_ID,
            )
        self.assertNotIn("b9b981cd1166b620a1bd", str(path))

    def test_new_machine_all_pass_releases_formal_suite(self):
        status, counts = status_from_rows(
            [{"status": "PASS"} for _ in range(28)]
        )
        self.assertEqual(status, "READY_FOR_FORMAL_RUN")
        self.assertEqual(counts["PASS"], 28)

    def test_current_machine_or_freeze_mismatch_is_rejected(self):
        expected_machine = {
            key: f"expected-{key}"
            for key in (
                "machine_id",
                "hostname",
                "operating_system",
                "python_version",
                "pytorch_version",
                "cuda_version",
                "gpu_name",
                "gpu_uuid",
                "gpu_total_memory",
                "driver_version",
            )
        }
        current_machine = dict(expected_machine)
        current_machine["gpu_uuid"] = "different-gpu"
        machine_matched, machine_mismatches = machine_identities_match(
            expected_machine, current_machine
        )
        self.assertFalse(machine_matched)
        self.assertIn("gpu_uuid", machine_mismatches)
        left = _freeze()
        right = _freeze()
        right["training_batch_profile_hash"] = "mismatch"
        matched, mismatches = freeze_identities_match(left, right)
        self.assertFalse(matched)
        self.assertIn("training_batch_profile_hash", mismatches)

    def test_original_preflight_cannot_release_e5(self):
        self.assertFalse(
            preflight_kind_matches("e5", "ORIGINAL_LOSS_ALL26")
        )

    def test_e5_preflight_cannot_release_original(self):
        self.assertFalse(
            preflight_kind_matches("original", "E5_COMMON_LOSS_ALL26")
        )

    def test_e5_core_allowed_before_base28_freeze(self):
        self.assertTrue(
            e5_core_allowed(
                e5_preflight_pass=True,
                base28_complete=False,
                a8_complete=True,
            )
        )

    def test_e5_finalize_fails_before_base28_freeze(self):
        self.assertIn(
            "BASE28_NOT_FROZEN",
            e5_finalize_blockers(
                base28_complete=False, a8_complete=True
            ),
        )

    def test_e5_core_allowed_before_a8_completion(self):
        self.assertTrue(
            e5_core_allowed(
                e5_preflight_pass=True,
                base28_complete=True,
                a8_complete=False,
            )
        )

    def test_e5_finalize_fails_before_a8_completion(self):
        self.assertIn(
            "BATCH4_A8_NOT_COMPLETE",
            e5_finalize_blockers(
                base28_complete=True, a8_complete=False
            ),
        )

    def test_two_machines_with_same_freeze_can_run_disjoint_roots(self):
        allowed, reasons = parallel_execution_allowed(
            _freeze(),
            _freeze(),
            ["custom_models/results/benchmark_v2_uniform_bs4/e2_a_seed2026"],
            [
                "custom_models/results/benchmark_v2_uniform_bs4/"
                "common_loss_architecture_seed2026"
            ],
        )
        self.assertTrue(allowed)
        self.assertEqual(reasons, [])

    def test_source_config_profile_or_data_mismatch_rejects_import(self):
        mismatch_keys = (
            "original_model_identities_hash",
            "e5_model_identities_hash",
            "training_batch_profile_hash",
            "dataset_hashes",
        )
        for key in mismatch_keys:
            with self.subTest(key=key):
                left = _freeze()
                right = _freeze()
                right[key] = "mismatch"
                matched, mismatches = freeze_identities_match(left, right)
                self.assertFalse(matched)
                self.assertIn(key, mismatches)

    def test_historical_gtx1060_oom_artifacts_are_unchanged(self):
        expected = {
            (
                "custom_models/results_smoke/benchmark_v2_uniform_bs4/"
                "hardware_preflight/msgnet/210e887f26c4440b8307/"
                "hardware_preflight.json"
            ): "042c10bd796f5d7c17219a71577fd81bdc9a3fdb3401e47a706d3bd8f516b6e7",
            (
                "custom_models/results_smoke/benchmark_v2_uniform_bs4/"
                "hardware_preflight/segrnn/b9b981cd1166b620a1bd/"
                "hardware_preflight.json"
            ): "aa4909c587a1d6e5b7d9ee3f284dcdc1222e4cd36d3619c2b591bb34429db316",
        }
        for relative, expected_hash in expected.items():
            path = PROJECT_ROOT / relative
            self.assertTrue(path.is_file())
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash
            )

    def test_partial_original_suite_skips_only_msgnet_in_original_order(self):
        original_rows = [
            {
                "model_id": model,
                "mode": "EVALUATE_ONLY" if index < 2 else "TRAIN",
            }
            for index, model in enumerate(
                [
                    "persistence",
                    "moving_average",
                    *[
                        model
                        for model in (
                            "gru",
                            "dlinear",
                            "lightts",
                            "tide",
                            "segrnn",
                            "transformer",
                            "patchtst",
                            "itransformer",
                            "timexer",
                            "timesnet",
                            "micn",
                            "wpmixer",
                            "multipatchformer",
                            "timemixer",
                            "tsmixer",
                            "frets",
                            "crossformer",
                            "msgnet",
                            "timefilter",
                            "gcn",
                            "stgcn",
                            "dcrnn",
                            "graph_wavenet",
                            "mtgnn",
                            "agcrn",
                            "stid",
                        )
                    ],
                ]
            )
        ]
        selected = original_skip_msgnet_run_rows(original_rows)
        self.assertEqual(len(selected), ORIGINAL_SKIP_MSGNET_EXPECTED)
        self.assertNotIn(
            ORIGINAL_SKIP_MSGNET_MODEL,
            [row["model_id"] for row in selected],
        )
        self.assertEqual(
            [row["model_id"] for row in selected],
            [
                row["model_id"]
                for row in original_rows
                if row["model_id"] != ORIGINAL_SKIP_MSGNET_MODEL
            ],
        )

    def test_partial_original_preflight_accepts_only_msgnet_fail_oom(self):
        rows = [
            {
                "model_id": model,
                "status": (
                    "FAIL_OOM" if model == ORIGINAL_SKIP_MSGNET_MODEL else "PASS"
                ),
                "identity_mismatches": [],
            }
            for model in TRAINABLE_MODELS
        ]
        self.assertEqual(original_skip_msgnet_preflight_blockers(rows), [])
        msgnet = next(
            row
            for row in rows
            if row["model_id"] == ORIGINAL_SKIP_MSGNET_MODEL
        )
        msgnet["status"] = "FAIL_NON_OOM"
        blockers = original_skip_msgnet_preflight_blockers(rows)
        self.assertTrue(any("msgnet" in blocker for blocker in blockers))

    def test_any_non_msgnet_preflight_failure_blocks_partial_suite(self):
        for failed_model in (
            model
            for model in TRAINABLE_MODELS
            if model != ORIGINAL_SKIP_MSGNET_MODEL
        ):
            with self.subTest(model=failed_model):
                rows = [
                    {
                        "model_id": model,
                        "status": (
                            "FAIL_OOM"
                            if model
                            in {ORIGINAL_SKIP_MSGNET_MODEL, failed_model}
                            else "PASS"
                        ),
                        "identity_mismatches": [],
                    }
                    for model in TRAINABLE_MODELS
                ]
                blockers = original_skip_msgnet_preflight_blockers(rows)
                self.assertTrue(
                    any(failed_model in blocker for blocker in blockers)
                )

    def test_msgnet_skip_requires_actual_oom_evidence(self):
        evidence = {
            "model_id": "msgnet",
            "status": "FAIL_OOM",
            "error_type": "OutOfMemoryError",
            "error_message": "CUDA out of memory",
        }
        self.assertTrue(msgnet_has_fail_oom_evidence(evidence))
        evidence["error_type"] = "ValueError"
        evidence["error_message"] = "shape mismatch"
        self.assertFalse(msgnet_has_fail_oom_evidence(evidence))
        evidence["status"] = "FAIL_NON_OOM"
        evidence["error_type"] = "OutOfMemoryError"
        evidence["error_message"] = "CUDA out of memory"
        self.assertFalse(msgnet_has_fail_oom_evidence(evidence))

    def test_partial_original_execution_manifest_metadata_is_explicit(self):
        metadata = partial_original_execution_metadata()
        self.assertEqual(
            metadata["suite_status"], ORIGINAL_SKIP_MSGNET_SUITE_STATUS
        )
        self.assertEqual(
            metadata["completed_expected"], ORIGINAL_SKIP_MSGNET_EXPECTED
        )
        self.assertEqual(metadata["skipped_models"], ["msgnet"])
        self.assertEqual(metadata["skip_reason"], ORIGINAL_SKIP_MSGNET_REASON)
        self.assertFalse(metadata["full_original_28_complete"])
        self.assertFalse(metadata["e4_aggregation_allowed"])
        self.assertFalse(metadata["original_28_readiness_allowed"])

    def test_completed_first_six_classify_as_skip_not_retrain(self):
        completed = {
            "persistence",
            "moving_average",
            "gru",
            "dlinear",
            "lightts",
            "tide",
        }
        for row in original_skip_msgnet_run_rows():
            if row["model_id"] not in completed:
                continue
            with self.subTest(model=row["model_id"]):
                decision = classify_formal_resume_evidence(
                    row, _completed_resume_evidence()
                )
                self.assertEqual(decision["decision"], "SKIP_COMPLETED")

    def test_incomplete_segrnn_classifies_as_will_run(self):
        segrnn = next(
            row
            for row in original_skip_msgnet_run_rows()
            if row["model_id"] == "segrnn"
        )
        evidence = _completed_resume_evidence()
        evidence["run_status"] = "FAILED"
        evidence["artifact_validation"] = None
        decision = classify_formal_resume_evidence(segrnn, evidence)
        self.assertEqual(decision["decision"], "WILL_RUN")

    def test_completed_artifact_missing_required_file_never_skips(self):
        row = original_skip_msgnet_run_rows()[0]
        evidence = _completed_resume_evidence()
        evidence["required_files_missing"] = ["metrics_eval_h10.json"]
        decision = classify_formal_resume_evidence(row, evidence)
        self.assertEqual(
            decision["decision"], "BLOCKED_IDENTITY_MISMATCH"
        )

    def test_identity_hash_mismatch_never_skips(self):
        row = original_skip_msgnet_run_rows()[2]
        evidence = _completed_resume_evidence()
        evidence["identity_mismatches"] = [
            "freeze_identity.original_model_identities_hash"
        ]
        decision = classify_formal_resume_evidence(row, evidence)
        self.assertEqual(
            decision["decision"], "BLOCKED_IDENTITY_MISMATCH"
        )

    def test_runner_starts_with_segrnn_after_skipping_first_six(self):
        manifest = {"executions": []}
        with (
            patch(
                "scripts.uniform_batch4_machine_gate.verify_original_skip_msgnet_suite",
                return_value=[],
            ),
            patch(
                "scripts.uniform_batch4_machine_gate.formal_resume_plan",
                return_value=_first_six_skip_resume_plan(),
            ),
            patch(
                "scripts.uniform_batch4_machine_gate._load_execution_manifest",
                return_value=manifest,
            ),
            patch("scripts.uniform_batch4_machine_gate._execution_path"),
            patch("scripts.uniform_batch4_machine_gate.write_json"),
            patch(
                "scripts.uniform_batch4_machine_gate.machine_manifest_paths",
                return_value=(Path("preflight.json"), Path("preflight.md")),
            ),
            patch(
                "scripts.uniform_batch4_machine_gate._formal_log_path",
                side_effect=lambda suite, model: Path(f"{suite}_{model}.log"),
            ),
            patch(
                "scripts.uniform_batch4_machine_gate._tee_process",
                return_value=0,
            ) as tee_process,
        ):
            from scripts.uniform_batch4_machine_gate import run_formal_suite

            code = run_formal_suite("original_skip_msgnet", "python.exe")

        self.assertEqual(code, 0)
        self.assertEqual(tee_process.call_count, 21)
        commands = [call.args[0] for call in tee_process.call_args_list]
        self.assertEqual(commands[0][commands[0].index("--model") + 1], "segrnn")
        self.assertEqual(
            manifest["executions"][0]["skipped_completed"],
            ["persistence", "moving_average", "gru", "dlinear", "lightts", "tide"],
        )

    @patch(
        "scripts.uniform_batch4_machine_gate.run_id_collision_blockers",
        return_value=[],
    )
    @patch(
        "scripts.uniform_batch4_machine_gate.verify_original_skip_msgnet_suite",
        return_value=[],
    )
    @patch(
        "scripts.uniform_batch4_machine_gate._formal_log_path",
        side_effect=lambda suite, model: Path(f"{suite}_{model}.log"),
    )
    @patch(
        "scripts.uniform_batch4_machine_gate.machine_manifest_paths",
        return_value=(Path("preflight.json"), Path("preflight.md")),
    )
    @patch("scripts.uniform_batch4_machine_gate._execution_path")
    @patch("scripts.uniform_batch4_machine_gate.write_json")
    @patch(
        "scripts.uniform_batch4_machine_gate._tee_process",
        return_value=0,
    )
    def test_partial_formal_suite_uses_27_isolated_python_processes(
        self,
        tee_process,
        _write_json,
        _execution_path,
        _machine_manifest_paths,
        _formal_log_path,
        _verify,
        _collisions,
    ):
        manifest = {"executions": []}
        with (
            patch(
                "scripts.uniform_batch4_machine_gate._load_execution_manifest",
                return_value=manifest,
            ),
            patch(
                "scripts.uniform_batch4_machine_gate.formal_resume_plan",
                return_value=_will_run_resume_plan(),
            ),
        ):
            from scripts.uniform_batch4_machine_gate import run_formal_suite

            code = run_formal_suite("original_skip_msgnet", "python.exe")
        self.assertEqual(code, 0)
        self.assertEqual(tee_process.call_count, 27)
        commands = [call.args[0] for call in tee_process.call_args_list]
        self.assertTrue(all(command[0] == "python.exe" for command in commands))
        self.assertTrue(
            all("msgnet" not in command for command in commands)
        )
        execution = manifest["executions"][0]
        self.assertEqual(execution["completed_count"], 27)
        self.assertNotIn("msgnet", execution["formal_models"])
        self.assertEqual(execution["status"], "COMPLETED_EXPECTED_27")

    @patch(
        "scripts.uniform_batch4_machine_gate.run_id_collision_blockers",
        return_value=[],
    )
    @patch(
        "scripts.uniform_batch4_machine_gate.verify_original_skip_msgnet_suite",
        return_value=[],
    )
    @patch(
        "scripts.uniform_batch4_machine_gate._formal_log_path",
        side_effect=lambda suite, model: Path(f"{suite}_{model}.log"),
    )
    @patch(
        "scripts.uniform_batch4_machine_gate.machine_manifest_paths",
        return_value=(Path("preflight.json"), Path("preflight.md")),
    )
    @patch("scripts.uniform_batch4_machine_gate._execution_path")
    @patch("scripts.uniform_batch4_machine_gate.write_json")
    @patch(
        "scripts.uniform_batch4_machine_gate._tee_process",
        side_effect=[0, 0, 19],
    )
    def test_partial_formal_suite_stops_at_first_non_msgnet_failure(
        self,
        tee_process,
        _write_json,
        _execution_path,
        _machine_manifest_paths,
        _formal_log_path,
        _verify,
        _collisions,
    ):
        manifest = {"executions": []}
        with (
            patch(
                "scripts.uniform_batch4_machine_gate._load_execution_manifest",
                return_value=manifest,
            ),
            patch(
                "scripts.uniform_batch4_machine_gate.formal_resume_plan",
                return_value=_will_run_resume_plan(),
            ),
        ):
            from scripts.uniform_batch4_machine_gate import run_formal_suite

            code = run_formal_suite("original_skip_msgnet", "python.exe")
        self.assertEqual(code, 19)
        self.assertEqual(tee_process.call_count, 3)
        execution = manifest["executions"][0]
        self.assertEqual(execution["status"], "FAILED_STOPPED")
        self.assertEqual(execution["completed_count"], 2)


if __name__ == "__main__":
    unittest.main()
