from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from st_mgprompt.artifact_status import _expected_config_hash, inspect_variant_artifacts
from st_mgprompt.experiment_protocol import apply_variant
from st_mgprompt.formal_runner import run_family


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _complete_artifacts(run_dir: Path, *, missing_h10: bool = False) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    expected = apply_variant(__import__("st_mgprompt.config", fromlist=["STMGPromptConfig"]).STMGPromptConfig(), "P4", "precision")
    expected.output_root = str(run_dir.parents[2].resolve())
    expected.run_id = f"{run_dir.parents[1].name}/P4"
    expected_dict = expected.to_dict()
    expected_hash = _expected_config_hash(expected_dict)
    _write_json(run_dir / "active_config.json", expected_dict)
    _write_json(run_dir / "config.json", expected_dict)
    (run_dir / "best_checkpoint.pt").write_bytes(b"test checkpoint")
    (run_dir / "train_log.csv").write_text("epoch,train_loss\n1,0.1\n", encoding="utf-8")
    _write_json(
        run_dir / "train_complete.json",
        {"status": "completed", "best_epoch": 1, "best_val_score_h10": 1.0, "config_hash": expected_hash},
    )
    _write_json(run_dir / "evaluation_complete.json", {"status": "completed", "checkpoint": "best_checkpoint.pt"})
    _write_json(run_dir / "prediction_metadata.json", {"artifact_source_checkpoint": "best_checkpoint.pt"})
    for horizon in (3, 6) if missing_h10 else (3, 6, 10):
        _write_json(run_dir / f"metrics_eval_h{horizon}.json", {"MAE": 1.0, "RMSE": 2.0, "R2": 0.5, "Score": 3.0})
    return expected


class ArtifactDrivenStatusTests(unittest.TestCase):
    def test_nonzero_exit_with_complete_artifacts_is_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "P4" / "STMGPrompt_ComponentAblation"
            expected = _complete_artifacts(run_dir)
            with patch(
                "st_mgprompt.artifact_status._cpu_load_checkpoint",
                return_value={"valid": True, "config_hash": _expected_config_hash(expected.to_dict())},
            ):
                audit = inspect_variant_artifacts(run_dir, "P4", expected, returncode=0xC0000005)
        self.assertEqual(audit["final_status"], "completed_with_exit_warning")
        self.assertEqual(audit["returncode_hex"], "0xC0000005")

    def test_zero_exit_with_missing_metric_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "P4" / "STMGPrompt_ComponentAblation"
            expected = _complete_artifacts(run_dir, missing_h10=True)
            with patch(
                "st_mgprompt.artifact_status._cpu_load_checkpoint",
                return_value={"valid": True, "config_hash": _expected_config_hash(expected.to_dict())},
            ):
                audit = inspect_variant_artifacts(run_dir, "P4", expected, returncode=0)
        self.assertEqual(audit["final_status"], "training_completed_evaluation_incomplete")

    def test_resume_skip_completed_does_not_launch_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            run_id = "precision_ablation_fixed_dual_seed2026"
            run_dir = output_root / run_id / "P4" / "STMGPrompt_ComponentAblation"
            expected = _complete_artifacts(run_dir)
            with patch(
                "st_mgprompt.artifact_status._cpu_load_checkpoint",
                return_value={"valid": True, "config_hash": _expected_config_hash(expected.to_dict())},
            ), patch("st_mgprompt.formal_runner.subprocess.Popen") as popen:
                result = run_family(
                    "precision",
                    [
                        "--variants", "P4", "--run-full", "--resume", "--skip-completed",
                        "--output-root", str(output_root), "--run-id", run_id,
                    ],
                )
        self.assertEqual(result["failed_variants"], [])
        self.assertEqual(result["statuses"][0]["decision"], "SKIP_COMPLETED")
        popen.assert_not_called()

    def test_failed_history_is_corrected_from_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            run_id = "precision_ablation_fixed_dual_seed2026"
            root = output_root / run_id
            run_dir = root / "P4" / "STMGPrompt_ComponentAblation"
            expected = _complete_artifacts(run_dir)
            _write_json(root / "experiment_status.json", [{"variant": "P4", "status": "FAILED", "returncode": 0xC0000005}])
            with patch(
                "st_mgprompt.artifact_status._cpu_load_checkpoint",
                return_value={"valid": True, "config_hash": _expected_config_hash(expected.to_dict())},
            ):
                result = run_family(
                    "precision",
                    [
                        "--variants", "P4", "--run-full", "--resume", "--skip-completed",
                        "--output-root", str(output_root), "--run-id", run_id,
                    ],
                )
            status = json.loads((root / "precision_ablation_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["failed_variants"], [])
        self.assertEqual(result["statuses"][0]["final_status"], "completed_with_exit_warning")

    def test_p5_dry_run_is_independent_of_p4_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_family(
                "precision",
                ["--variants", "P5", "--dry-run", "--output-root", tmp, "--run-id", "precision_ablation_fixed_dual_seed2026"],
            )
        self.assertEqual(result["failed_variants"], [])
        self.assertEqual(result["statuses"][0]["final_status"], "pending")


if __name__ == "__main__":
    unittest.main()

