import json
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_v2.hardware_preflight as hardware_preflight
import benchmark_v2.model_cli as model_cli
from benchmark_v2.hardware_preflight import (
    PREFLIGHT_SHAPE,
    launch_formal_train,
    launch_model_suite,
    preflight_artifact_path,
    preflight_identity,
)


class HardwarePreflightLauncherTests(unittest.TestCase):
    def _write_pass(self, model_id, root, pid=101):
        path = preflight_artifact_path(model_id, root=root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            **preflight_identity(model_id),
            "status": "PASS",
            "forward_completed": True,
            "backward_completed": True,
            "preflight_pid": pid,
        }), encoding="utf-8")
        return path

    def _launch(self, model_id, root, runner):
        return launch_formal_train(
            model_id,
            input_path="input.parquet",
            target_path="target.parquet",
            output_root="formal-root",
            run_id=f"{model_id}-run",
            device="cuda",
            preflight_root=root,
            runner=runner,
        )

    def test_exact_shape_is_frozen(self):
        self.assertEqual(
            PREFLIGHT_SHAPE,
            {"B": 32, "T": 144, "N": 134, "C": 16, "H": 10},
        )

    def test_missing_pass_runs_preflight_then_new_formal_process(self):
        with tempfile.TemporaryDirectory() as td:
            calls = []
            def runner(argv, check=False):
                calls.append(tuple(argv))
                if "_hardware-preflight-worker" in argv:
                    self._write_pass("tide", td, pid=111)
                    return SimpleNamespace(returncode=0, pid=111)
                return SimpleNamespace(returncode=0, pid=222)
            self.assertEqual(self._launch("tide", td, runner), 0)
            self.assertEqual(len(calls), 2)
            self.assertIn("_hardware-preflight-worker", calls[0])
            self.assertIn("_formal-train-worker", calls[1])
            self.assertNotEqual(111, 222)

    def test_matching_pass_is_reused_and_only_train_is_spawned(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_pass("segrnn", td)
            calls = []
            runner = lambda argv, check=False: (
                calls.append(tuple(argv)) or SimpleNamespace(returncode=0)
            )
            self.assertEqual(self._launch("segrnn", td, runner), 0)
            self.assertEqual(len(calls), 1)
            self.assertIn("_formal-train-worker", calls[0])

    def test_fail_oom_does_not_start_formal_train(self):
        with tempfile.TemporaryDirectory() as td:
            calls = []
            runner = lambda argv, check=False: (
                calls.append(tuple(argv)) or SimpleNamespace(returncode=3)
            )
            self.assertEqual(self._launch("tide", td, runner), 3)
            self.assertEqual(len(calls), 1)
            self.assertIn("_hardware-preflight-worker", calls[0])

    def test_fail_non_oom_does_not_start_formal_train(self):
        with tempfile.TemporaryDirectory() as td:
            calls = []
            runner = lambda argv, check=False: (
                calls.append(tuple(argv)) or SimpleNamespace(returncode=4)
            )
            self.assertEqual(self._launch("segrnn", td, runner), 4)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("_formal-train-worker", calls[0])

    def test_shape_mismatch_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_pass("tide", td)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["B"] = 31
            path.write_text(json.dumps(payload), encoding="utf-8")
            calls = []
            def runner(argv, check=False):
                calls.append(tuple(argv))
                return SimpleNamespace(returncode=3)
            self.assertEqual(self._launch("tide", td, runner), 3)
            self.assertIn("_hardware-preflight-worker", calls[0])

    def test_config_mismatch_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_pass("timemixer", td)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["model_config_hash"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            calls = []

            def runner(argv, check=False):
                calls.append(tuple(argv))
                return SimpleNamespace(returncode=3)

            self.assertEqual(self._launch("timemixer", td, runner), 3)
            self.assertIn("_hardware-preflight-worker", calls[0])

    def test_source_hash_mismatch_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_pass("frets", td)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["source_hash"] = "f" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            calls = []

            def runner(argv, check=False):
                calls.append(tuple(argv))
                return SimpleNamespace(returncode=3)

            self.assertEqual(self._launch("frets", td, runner), 3)
            self.assertIn("_hardware-preflight-worker", calls[0])

    def test_suite_waits_for_model_a_exit_before_model_b_and_stays_cpu_only(self):
        events = []
        def launcher(**request):
            model_id = request["model_id"]
            events.append(("start", model_id))
            events.append(("exit", model_id))
            return 0
        requests = [
            {"model_id": "itransformer"},
            {"model_id": "timexer"},
        ]
        results = launch_model_suite(requests, launcher=launcher)
        self.assertEqual(
            events,
            [
                ("start", "itransformer"),
                ("exit", "itransformer"),
                ("start", "timexer"),
                ("exit", "timexer"),
            ],
        )
        self.assertEqual([row["exit_code"] for row in results], [0, 0])

    def test_default_child_environment_contains_repository_root(self):
        calls = []

        def fake_run(argv, **kwargs):
            calls.append((argv, kwargs))
            return SimpleNamespace(returncode=0)

        self.assertEqual(
            hardware_preflight._run_child(
                ["--help"],
                runner=fake_run,
            ),
            0,
        )
        self.assertEqual(calls[0][1], {"check": False})

        with patch.object(hardware_preflight.subprocess, "run", side_effect=fake_run):
            self.assertEqual(
                hardware_preflight._run_child(["--help"], runner=hardware_preflight.subprocess.run),
                0,
            )
        kwargs = calls[-1][1]
        self.assertEqual(kwargs["cwd"], str(hardware_preflight.PROJECT_ROOT))
        self.assertIn(str(hardware_preflight.PROJECT_ROOT), kwargs["env"]["PYTHONPATH"])

    def test_failed_attempt_isolated_and_does_not_replace_matching_pass(self):
        identity = {"model_id": "tide", "B": 4, "N": 134, "H": 10}
        with tempfile.TemporaryDirectory() as td, patch.object(
            hardware_preflight,
            "preflight_identity",
            return_value=identity,
        ), patch("torch.cuda.is_available", return_value=False):
            first = hardware_preflight.run_preflight_worker("tide", root=td)
            self.assertEqual(first["status"], "FAIL_NON_OOM")
            self.assertTrue(Path(first["artifact_path"]).is_file())
            self.assertEqual(
                len(list((Path(td) / "tide" / "attempts").iterdir())),
                1,
            )

            second_attempt = hardware_preflight.new_preflight_attempt_id()
            second_path = hardware_preflight._attempt_result_path("tide", td, second_attempt)
            pass_payload = {
                **identity,
                "attempt_id": second_attempt,
                "identity_hash": hardware_preflight.stable_hash(identity),
                "status": "PASS",
                "forward_completed": True,
                "backward_completed": True,
                "finite_prediction": True,
                "finite_loss": True,
                "finite_gradients": True,
                "prediction_shape": [4, 134, 10],
            }
            second_path.parent.mkdir(parents=True, exist_ok=True)
            second_path.write_text(json.dumps(pass_payload), encoding="utf-8")
            hardware_preflight._publish_latest(
                "tide",
                root=td,
                attempt_id=second_attempt,
                result=pass_payload,
                result_path=second_path,
            )
            third = hardware_preflight.run_preflight_worker("tide", root=td)
            latest = json.loads((Path(td) / "tide" / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(third["status"], "FAIL_NON_OOM")
            self.assertEqual(latest["attempt_id"], second_attempt)

    def test_child_contract_keeps_exception_when_artifact_is_missing(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "child.log"
            log.write_text(
                "Traceback (most recent call last):\n"
                "ModuleNotFoundError: No module named 'scripts'\n",
                encoding="utf-8",
            )
            contract = hardware_preflight.build_preflight_child_contract(
                "gcn",
                exit_code=1,
                log_path=log,
                root=td,
            )
            self.assertEqual(contract["error_type"], "CHILD_EXCEPTION")
            self.assertEqual(contract["exception_type"], "ModuleNotFoundError")
            self.assertIn("No module named 'scripts'", contract["error_message"])
            self.assertFalse(contract["artifact_written"])

    def test_missing_artifact_does_not_become_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "child.log"
            log.write_text(
                "Traceback (most recent call last):\n"
                "ModuleNotFoundError: No module named 'scripts'\n",
                encoding="utf-8",
            )
            contract = hardware_preflight.build_preflight_child_contract(
                "gcn",
                exit_code=1,
                log_path=log,
                root=td,
                expected_identity={"model_id": "gcn", "protocol_hash": "frozen"},
            )
            self.assertEqual(contract["error_type"], "CHILD_EXCEPTION")
            self.assertEqual(contract["identity_mismatch_fields"], [])

    def test_formal_train_accepts_attempt_binding_arguments(self):
        parameters = inspect.signature(model_cli.formal_train).parameters
        self.assertIn("preflight_attempt_id", parameters)
        self.assertIn("preflight_artifact_sha256", parameters)


if __name__ == "__main__":
    unittest.main()
