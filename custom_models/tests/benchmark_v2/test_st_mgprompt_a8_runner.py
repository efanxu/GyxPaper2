from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import st_mgprompt_a8_batch4_gate as gate
from st_mgprompt import run_st_mgprompt as runner
from st_mgprompt.a8_batch4_contract import (
    A8_OUTPUT_ROOT,
    A8_RUN_ID,
    A8_SCOPE_ID,
    FEATURE_ORDER_HASH,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _formal_argv():
    return [
        "run_st_mgprompt.py",
        "--component-ablation",
        "A8",
        "--training-profile",
        "uniform_train_batch4_v1",
        "--run-id",
        A8_RUN_ID,
        "--output-root",
        A8_OUTPUT_ROOT,
        "--model-name",
        "STMGPrompt_ComponentAblation",
        "--device",
        "cuda",
        "--no-amp",
        "--windows-safe-mode",
        "--source-revision",
        gate._git_commit(),
    ]


def _formal_config_and_args():
    with patch.object(
        sys,
        "argv",
        _formal_argv(),
    ):
        args = runner.parse_args()
    return runner.build_config(args), args


def _make_ready_fixture(root: Path, *, simulate_runner_main: bool = False) -> tuple[Path, object, object]:
    run_dir = root / A8_RUN_ID / "STMGPrompt_ComponentAblation"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg, args = _formal_config_and_args()
    runner._write_training_batch_identity(run_dir, cfg)
    _write_json(run_dir / "effective_config_diff.json", {"passed": True})
    for name, payload in (
        ("train_complete.json", {"status": "completed"}),
        ("evaluation_complete.json", {"status": "completed"}),
        ("protocol_check.json", {"passed": True, "protocol_hash": cfg.base_benchmark_protocol_hash}),
        ("prediction_metadata.json", {"run_id": A8_RUN_ID}),
    ):
        _write_json(run_dir / name, payload)
    (run_dir / "best_checkpoint.pt").write_bytes(b"fixture-best-checkpoint")
    (run_dir / "last_checkpoint.pt").write_bytes(b"fixture-last-checkpoint")
    (run_dir / "train_log.csv").write_text("epoch\n1\n", encoding="utf-8")
    metrics = []
    for horizon in (3, 6, 10):
        payload = {
            "horizon": horizon,
            "MAE": 1.0,
            "RMSE": 2.0,
            "R2": 0.5,
            "Score": 3.0,
            "valid_target_count": 4,
        }
        metrics.append(payload)
        _write_json(run_dir / f"metrics_eval_h{horizon}.json", payload)
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["horizon", "MAE", "RMSE", "R2", "Score", "valid_target_count"],
        )
        writer.writeheader()
        writer.writerows(metrics)
    data = SimpleNamespace(feature_cols=list(cfg.feature_cols), num_nodes=134)
    runner._write_a8_data_signature(cfg, data, run_dir)
    if simulate_runner_main:
        with (
            patch.object(sys, "argv", _formal_argv()),
            patch.object(runner, "_run_dir", return_value=run_dir),
            patch.object(
                runner,
                "_run_once",
                return_value={"run_dir": str(run_dir), "metrics": {}, "protocol_passed": True},
            ),
            patch.object(runner, "_print_environment_summary"),
        ):
            runner.main()
    else:
        runner._finalize_a8_formal_success(
            cfg,
            run_dir,
            args,
            started_at="2026-08-05T00:00:00Z",
            finished_at="2026-08-05T00:01:00Z",
        )
    return run_dir, cfg, args


class STMGPromptA8RunnerIntegrationTests(unittest.TestCase):
    def test_real_success_finalizer_writes_signature_receipt_and_ready_artifact(self):
        with tempfile.TemporaryDirectory(dir=gate.PROJECT_ROOT) as temp:
            root = Path(temp)
            run_dir, cfg, _ = _make_ready_fixture(root, simulate_runner_main=True)
            self.assertTrue(runner._is_formal_a8_batch4(cfg))
            status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
            signature = json.loads((run_dir / "data_signature.json").read_text(encoding="utf-8"))
            receipt = json.loads(
                (run_dir / "a8_batch4_execution_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                {
                    status["current_stage"],
                    status["status"],
                    status["exit_code"],
                    status["run_mode"],
                    status["formal_training"],
                    status["artifact_profile"],
                    status["scope_id"],
                    status["run_id"],
                },
                {"PROCESS_FINISHED", "COMPLETED", 0, "formal", True, "TRAIN", A8_SCOPE_ID, A8_RUN_ID},
            )
            self.assertEqual(signature["dataset_id"], "SDWPF")
            self.assertEqual(signature["node_count"], 134)
            self.assertEqual(signature["feature_order_hash"], FEATURE_ORDER_HASH)
            self.assertEqual(signature["input_relative_path"], "dataset/sdwpf_model_input_base.parquet")
            self.assertEqual(signature["target_relative_path"], "dataset/sdwpf_eval_target.parquet")
            self.assertEqual(
                signature["input_sha256"],
                runner._a8_sha256(gate.PROJECT_ROOT / "dataset/sdwpf_model_input_base.parquet"),
            )
            self.assertEqual(
                signature["target_sha256"],
                runner._a8_sha256(gate.PROJECT_ROOT / "dataset/sdwpf_eval_target.parquet"),
            )
            self.assertEqual(signature["split_ratios"], [0.8, 0.1, 0.1])
            self.assertEqual(signature["stride"], {"train": 6, "val": 3, "test": 1})
            self.assertNotIn("D:\\", json.dumps(signature))
            expected_hash = gate.canonical_hash(signature)
            self.assertEqual(receipt["dataset_identity_hash"], expected_hash)
            self.assertEqual(receipt["data_signature_hash"], expected_hash)
            inspected = gate.inspect_a8_artifact(
                output_root=root,
                project_root=gate.PROJECT_ROOT,
            )
            self.assertTrue(inspected["ready"], inspected)

    def test_missing_signature_status_or_exit_code_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=gate.PROJECT_ROOT) as temp:
            root = Path(temp)
            run_dir, _, _ = _make_ready_fixture(root)
            (run_dir / "data_signature.json").unlink()
            missing = gate.inspect_a8_artifact(output_root=root, project_root=gate.PROJECT_ROOT)
            self.assertFalse(missing["ready"])
            self.assertTrue(any("data_signature" in reason.lower() for reason in missing["reasons"]))

        for missing_key in ("status", "exit_code"):
            with self.subTest(missing_key=missing_key), tempfile.TemporaryDirectory(dir=gate.PROJECT_ROOT) as temp:
                root = Path(temp)
                run_dir, _, _ = _make_ready_fixture(root)
                status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
                status.pop(missing_key)
                _write_json(run_dir / "run_status.json", status)
                inspected = gate.inspect_a8_artifact(output_root=root, project_root=gate.PROJECT_ROOT)
                self.assertFalse(inspected["ready"])
                self.assertIn("A8_RUN_STATUS_NOT_COMPLETED", inspected["reasons"])

    def test_smoke_and_historical_output_are_not_formal_a8(self):
        cfg, _ = _formal_config_and_args()
        cfg.smoke = True
        self.assertFalse(runner._is_formal_a8_batch4(cfg))
        cfg.smoke = False
        cfg.output_root = "custom_models/results/st_mgprompt_component_ablation"
        self.assertFalse(runner._is_formal_a8_batch4(cfg))

    def test_failure_finalizer_contains_nonzero_failure_identity(self):
        with tempfile.TemporaryDirectory(dir=gate.PROJECT_ROOT) as temp:
            root = Path(temp)
            run_dir = root / A8_RUN_ID / "STMGPrompt_ComponentAblation"
            cfg, _ = _formal_config_and_args()
            failure = runner._failure_payload(
                cfg,
                run_dir,
                "TRAIN_STARTED",
                RuntimeError("fixture failure"),
            )
            runner.update_run_status(run_dir, "PROCESS_EXCEPTION", failure)
            status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["current_stage"], "PROCESS_EXCEPTION")
            self.assertEqual(status["status"], "FAILED")
            self.assertNotEqual(status["exit_code"], 0)
            self.assertEqual(status["failure_stage"], "TRAIN_STARTED")
            self.assertEqual(status["error_type"], "RuntimeError")
            self.assertEqual(status["error"], "fixture failure")

    def test_child_zero_but_readiness_failure_returns_gate_74_and_no_reference(self):
        with tempfile.TemporaryDirectory(dir=gate.PROJECT_ROOT) as temp:
            log_root = Path(temp) / "logs"
            completed = subprocess.CompletedProcess([], 0)
            revision = gate._git_commit()
            with (
                patch.object(gate, "read_matching_preflight_pass", return_value={"status": "PASS"}),
                patch.object(gate, "build_plan", return_value={"entries": [{"action": "RUN_MISSING"}]}),
                patch.object(gate, "_acquire_lock", return_value={"lock_owner": "fixture"}),
                patch.object(gate, "_release_lock"),
                patch.object(gate, "build_readiness", return_value={"status": "NOT_READY_A8_BATCH4"}),
                patch.object(gate, "write_reference") as write_reference,
                patch.object(gate.subprocess, "run", return_value=completed),
            ):
                code, report = gate.run(log_root=log_root, source_revision=revision)
            self.assertEqual(code, 74)
            self.assertEqual(report["status"], "COMPLETED_BUT_NOT_READY")
            self.assertEqual(report["child_exit_code"], 0)
            self.assertEqual(report["gate_exit_code"], 74)
            write_reference.assert_not_called()

    def test_child_nonzero_is_propagated_and_reference_write_failure_is_nonzero(self):
        revision = gate._git_commit()
        with tempfile.TemporaryDirectory(dir=gate.PROJECT_ROOT) as temp:
            with (
                patch.object(gate, "read_matching_preflight_pass", return_value={"status": "PASS"}),
                patch.object(gate, "build_plan", return_value={"entries": [{"action": "RUN_MISSING"}]}),
                patch.object(gate, "_acquire_lock", return_value={"lock_owner": "fixture"}),
                patch.object(gate, "_release_lock"),
                patch.object(gate, "build_readiness", return_value={"status": "READY_A8_BATCH4_PREREQUISITE"}),
                patch.object(gate, "write_reference") as write_reference,
                patch.object(gate.subprocess, "run", return_value=subprocess.CompletedProcess([], 19)),
            ):
                code, report = gate.run(log_root=Path(temp) / "child-failed", source_revision=revision)
            self.assertEqual(code, 19)
            self.assertEqual(report["child_exit_code"], 19)
            self.assertEqual(report["gate_exit_code"], 19)
            write_reference.assert_not_called()

        with tempfile.TemporaryDirectory(dir=gate.PROJECT_ROOT) as temp:
            with (
                patch.object(gate, "read_matching_preflight_pass", return_value={"status": "PASS"}),
                patch.object(gate, "build_plan", return_value={"entries": [{"action": "RUN_MISSING"}]}),
                patch.object(gate, "_acquire_lock", return_value={"lock_owner": "fixture"}),
                patch.object(gate, "_release_lock"),
                patch.object(gate, "build_readiness", return_value={"status": "READY_A8_BATCH4_PREREQUISITE"}),
                patch.object(gate, "write_reference", side_effect=OSError("fixture reference failure")),
                patch.object(gate.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)),
            ):
                code, report = gate.run(log_root=Path(temp) / "reference-failed", source_revision=revision)
            self.assertEqual(code, 74)
            self.assertEqual(report["status"], "REFERENCE_WRITE_FAILED")
            self.assertEqual(report["child_exit_code"], 0)
            self.assertEqual(report["gate_exit_code"], 74)

    def test_reference_is_written_only_after_ready_and_archive_preserves_old_receipt(self):
        with tempfile.TemporaryDirectory(dir=gate.PROJECT_ROOT) as temp:
            root = Path(temp)
            log_root = root / "logs"
            payload = {"status": "VALID", "reference_id": "fixture"}
            revision = gate._git_commit()

            def write_fixture_reference(path):
                path.parent.mkdir(parents=True, exist_ok=True)
                _write_json(path, payload)
                return payload

            with (
                patch.object(gate, "read_matching_preflight_pass", return_value={"status": "PASS"}),
                patch.object(gate, "build_plan", return_value={"entries": [{"action": "RUN_MISSING"}]}),
                patch.object(gate, "_acquire_lock", return_value={"lock_owner": "fixture"}),
                patch.object(gate, "_release_lock"),
                patch.object(gate, "build_readiness", return_value={"status": "READY_A8_BATCH4_PREREQUISITE"}),
                patch.object(gate, "write_reference", side_effect=write_fixture_reference) as write_reference,
                patch.object(gate.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)),
            ):
                code, report = gate.run(log_root=log_root, source_revision=revision)
            self.assertEqual(code, 0)
            self.assertEqual(report["child_exit_code"], 0)
            self.assertEqual(report["gate_exit_code"], 0)
            self.assertTrue(report["reference_written"])
            write_reference.assert_called_once()

        with tempfile.TemporaryDirectory(dir=gate.PROJECT_ROOT) as temp:
            root = Path(temp)
            source = root / A8_RUN_ID / "STMGPrompt_ComponentAblation"
            source.mkdir(parents=True)
            _write_json(source / "a8_batch4_execution_receipt.json", {"receipt": "old"})
            with (
                patch.object(gate, "RESULT_ROOT", root),
                patch.object(gate, "RUN_DIR", source),
                patch.object(gate, "_worker_evidence", return_value=(False, "INACTIVE")),
                patch.object(gate, "inspect_a8_artifact", return_value={"reasons": ["A8_RECEIPT_INCOMPLETE"]}),
            ):
                archived = gate.archive_or_quarantine(kind=gate.ARCHIVE_DIRECTORY, apply=True)
            target = Path(archived["target"])
            self.assertFalse(source.exists())
            self.assertEqual(
                json.loads((target / "a8_batch4_execution_receipt.json").read_text(encoding="utf-8")),
                {"receipt": "old"},
            )
            self.assertTrue((target / "archive_receipt.json").is_file())

            # A later failed attempt may be archived again; each attempt gets
            # a unique target and the first receipt remains immutable.
            source.mkdir(parents=True)
            _write_json(source / "a8_batch4_execution_receipt.json", {"receipt": "new"})
            with (
                patch.object(gate, "RESULT_ROOT", root),
                patch.object(gate, "RUN_DIR", source),
                patch.object(gate, "_worker_evidence", return_value=(False, "INACTIVE")),
                patch.object(gate, "inspect_a8_artifact", return_value={"reasons": ["A8_RECEIPT_INCOMPLETE"]}),
            ):
                retried = gate.archive_or_quarantine(kind=gate.ARCHIVE_DIRECTORY, apply=True)
            retry_target = Path(retried["target"])
            self.assertNotEqual(retry_target, target)
            self.assertEqual(
                json.loads((target / "a8_batch4_execution_receipt.json").read_text(encoding="utf-8")),
                {"receipt": "old"},
            )
            self.assertEqual(
                json.loads((retry_target / "a8_batch4_execution_receipt.json").read_text(encoding="utf-8")),
                {"receipt": "new"},
            )


if __name__ == "__main__":
    unittest.main()
