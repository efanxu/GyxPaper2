import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from benchmark_v2.experiments.e5_common_loss.scope27_contract import (
    E5_SCOPE27_ID,
    E5_SCOPE27_TRAINABLE_MODELS,
    TRAINING_PROFILE_ID,
)
from benchmark_v2.hardware_preflight import launch_formal_train


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "custom_models/docs/benchmark_v2/E5/E5_SCOPE27_VARIANT_MANIFEST.json"
)


class E5Scope27Tests(unittest.TestCase):
    def test_manifest_counts_order_exclusions_and_unique_run_ids(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        entries = manifest["entries"]
        trainable = [
            row for row in entries if row["entry_type"] == "TRAIN_COMMON_LOSS"
        ]
        evaluate_only = [
            row
            for row in entries
            if row["entry_type"]
            == "EVALUATE_ONLY_COMMON_LOSS_DIAGNOSTIC"
        ]
        reference = [
            row
            for row in entries
            if row["entry_type"] == "REFERENCE_ONLY_FORMAL_A8"
        ]
        self.assertEqual(manifest["counts"], {
            "trainable": 24,
            "evaluate_only": 2,
            "a8_reference": 1,
            "total_evidence": 27,
        })
        self.assertEqual(
            tuple(row["model_id"] for row in trainable),
            E5_SCOPE27_TRAINABLE_MODELS,
        )
        self.assertEqual(len(evaluate_only), 2)
        self.assertEqual(len(reference), 1)
        self.assertEqual(len(entries), 27)
        self.assertEqual(len({row["e5_run_id"] for row in entries}), 27)
        self.assertFalse({"segrnn", "msgnet"} & {
            row["model_id"] for row in entries
        })

    def test_scope27_launcher_does_not_start_or_read_hardware_artifact(self):
        calls = []

        def runner(argv, check=False):
            calls.append(tuple(argv))
            return SimpleNamespace(returncode=0)

        code = launch_formal_train(
            "gru",
            input_path="input.parquet",
            target_path="target.parquet",
            output_root="formal-root",
            run_id="gru-scope27",
            device="cuda",
            experiment_profile="e5_common_loss_v1",
            training_profile=TRAINING_PROFILE_ID,
            formal_scope_id=E5_SCOPE27_ID,
            runner=runner,
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("_formal-train-worker", calls[0])
        self.assertNotIn("_hardware-preflight-worker", calls[0])
        self.assertIn("--formal-scope-id", calls[0])
        self.assertIn(E5_SCOPE27_ID, calls[0])

    def test_excluded_model_cannot_use_scope27_authorization(self):
        with self.assertRaises(ValueError):
            launch_formal_train(
                "segrnn",
                input_path="input.parquet",
                target_path="target.parquet",
                output_root="formal-root",
                run_id="segrnn-scope27",
                device="cuda",
                experiment_profile="e5_common_loss_v1",
                training_profile=TRAINING_PROFILE_ID,
                formal_scope_id=E5_SCOPE27_ID,
                runner=lambda argv, check=False: SimpleNamespace(returncode=0),
            )

    def test_readiness_denominator_is_27_without_excluded_models(self):
        from scripts import e5_batch4_scope27_gate as gate

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        def fake_run(_manifest, entry, _root):
            return {
                "entry_id": entry["entry_id"],
                "model_id": entry["model_id"],
                "entry_type": entry["entry_type"],
                "ready": True,
            }

        def fake_a8(_manifest, entry):
            return {
                "entry_id": entry["entry_id"],
                "model_id": entry["model_id"],
                "entry_type": entry["entry_type"],
                "ready": True,
            }

        with (
            patch.object(gate, "EXPECTED_OUTPUT_ROOT", PROJECT_ROOT / "fixture"),
            patch.object(gate, "inspect_run", side_effect=fake_run),
            patch.object(gate, "inspect_a8", side_effect=fake_a8),
        ):
            report = gate.build_readiness(manifest, PROJECT_ROOT / "fixture")
        self.assertEqual(report["status"], "READY")
        self.assertEqual(
            report["counts"]["total_evidence"],
            {"ready": 27, "expected": 27},
        )
        self.assertEqual(
            report["counts"]["trainable"],
            {"ready": 24, "expected": 24},
        )

    def test_linux_launcher_has_no_excluded_or_hardware_artifact_gate_text(self):
        text = (
            PROJECT_ROOT
            / "custom_models/docs/benchmark_v2/E5/"
            "E5_RUN_ALL_27_BATCH4_LINUX.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("SegRNN", text)
        self.assertNotIn("MSGNet", text)
        self.assertNotIn("preflight", text.casefold())
        self.assertIn("--require-complete", text)
        self.assertIn("e5_scope27_readiness.json", text)

    def test_null_metric_fails_closed_and_is_not_replaced_with_zero(self):
        from scripts import e5_batch4_scope27_gate as gate

        paths = []
        for horizon in gate.HORIZONS:
            path = MagicMock()
            path.name = f"metrics_eval_h{horizon}.json"
            path.is_file.return_value = True
            paths.append(path)
        payloads = [
            {
                "horizon": horizon,
                "Score": None if horizon == 3 else 1.0,
                "MAE": 1.0,
                "RMSE": 1.0,
                "R2": 1.0,
            }
            for horizon in gate.HORIZONS
        ]
        with patch.object(gate, "load_json", side_effect=payloads):
            metrics, reasons = gate._metric_payloads(paths)
        self.assertIsNone(metrics[3]["Score"])
        self.assertIn(
            "NON_FINITE_OR_MISSING:metrics_eval_h3.json:Score",
            reasons,
        )


if __name__ == "__main__":
    unittest.main()
