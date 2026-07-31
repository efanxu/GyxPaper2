import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
