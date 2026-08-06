from __future__ import annotations

import unittest
import os
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
E5_DOCS = PROJECT_ROOT / "custom_models/docs/benchmark_v2/E5"


def _msys_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").casefold()
    tail = resolved.as_posix().split(":", 1)[-1]
    return f"/{drive}{tail}"


class Batch4LauncherTests(unittest.TestCase):
    def test_fresh_linux_output_roots_are_created(self) -> None:
        script = (E5_DOCS / "E5_RUN_ALL_27_BATCH4_LINUX.sh").read_text(encoding="utf-8")
        for command in (
            'mkdir -p "$OUTPUT_ROOT"',
            'mkdir -p "$AUDIT_ROOT"',
            'mkdir -p "$PREFLIGHT_ROOT"',
            'mkdir -p "$RUN_LOG_ROOT"',
        ):
            self.assertIn(command, script)
        self.assertNotIn("exit 67", script)
        self.assertNotIn("output-root parent is missing", script)
        self.assertIn('if [[ "$lock_status" != "ABSENT" ]]', script)

    def test_autoshutdown_mock_hooks_preserve_required_order(self) -> None:
        for name, runner_token, exit_file in (
            ("E5_RUN_ALL_27_BATCH4_LINUX_AUTOSHUTDOWN.sh", '"$BASH_EXECUTABLE" "$RUN_ALL"', "e5_scope27_autoshutdown.exitcode"),
            ("E5_A8_BATCH4_LINUX_AUTOSHUTDOWN.sh", '"$BASH_EXECUTABLE" "$RUNNER"', "a8_batch4_autoshutdown.exitcode"),
        ):
            with self.subTest(name=name):
                script = (E5_DOCS / name).read_text(encoding="utf-8")
                order = (
                    script.index(runner_token),
                    script.index("run_code=${PIPESTATUS[0]}") if "run_code=${PIPESTATUS[0]}" in script else script.index("main_code=${PIPESTATUS[0]}"),
                    script.index(exit_file),
                    script.index('"$SYNC_COMMAND"'),
                    script.index('"$SHUTDOWN_COMMAND" -h now'),
                )
                self.assertEqual(tuple(sorted(order)), order)
                self.assertIn("SHUTDOWN_COMMAND=${SHUTDOWN_COMMAND:-/usr/bin/shutdown}", script)
                self.assertIn("SYNC_COMMAND=${SYNC_COMMAND:-sync}", script)

    @unittest.skipUnless(os.environ.get("BASH_EXE"), "GNU bash path is required for the mock shutdown fixture")
    def test_autoshutdown_mock_executes_run_exitcode_sync_shutdown(self) -> None:
        bash = Path(os.environ["BASH_EXE"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_log = root / "events.log"
            fake_sync = root / "fake-sync.sh"
            fake_shutdown = root / "fake-shutdown.sh"
            fake_sync.write_text('#!/bin/sh\necho SYNC >> "$EVENT_LOG"\n', encoding="utf-8")
            fake_shutdown.write_text('#!/bin/sh\necho SHUTDOWN >> "$EVENT_LOG"\n', encoding="utf-8")
            for wrapper_name, runner_name, exit_file in (
                ("E5_RUN_ALL_27_BATCH4_LINUX_AUTOSHUTDOWN.sh", "E5_RUN_ALL_27_BATCH4_LINUX.sh", "custom_models/logs/uniform_bs4/audit/e5_scope27/e5_scope27_autoshutdown.exitcode"),
                ("E5_A8_BATCH4_LINUX_AUTOSHUTDOWN.sh", "E5_A8_BATCH4_LINUX.sh", "custom_models/logs/uniform_bs4/audit/st_mgprompt_a8_batch4/a8_batch4_autoshutdown.exitcode"),
            ):
                runner = root / "custom_models/docs/benchmark_v2/E5" / runner_name
                runner.parent.mkdir(parents=True, exist_ok=True)
                for simulated_code in (0, 7):
                    with self.subTest(wrapper=wrapper_name, exit_code=simulated_code):
                        runner.write_text(f'#!/bin/sh\necho RUN >> "$EVENT_LOG"\nexit {simulated_code}\n', encoding="utf-8")
                        event_log.write_text("", encoding="utf-8")
                        env = {
                            **os.environ,
                            "PROJECT_ROOT": _msys_path(root),
                            "BASH_EXECUTABLE": _msys_path(bash),
                            "SYNC_COMMAND": _msys_path(fake_sync),
                            "SHUTDOWN_COMMAND": _msys_path(fake_shutdown),
                            "EVENT_LOG": _msys_path(event_log),
                            "PATH": f"{_msys_path(bash.parent)}:{os.environ.get('PATH', '')}",
                        }
                        completed = subprocess.run(
                            [str(bash), str(E5_DOCS / wrapper_name)],
                            cwd=PROJECT_ROOT, env=env, check=False,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                        )
                        self.assertEqual(completed.returncode, simulated_code, completed.stdout)
                        self.assertEqual(event_log.read_text(encoding="utf-8").splitlines(), ["RUN", "SYNC", "SHUTDOWN"])
                        selected_exit = root / exit_file
                        self.assertTrue(selected_exit.is_file(), {"found": [str(path) for path in root.rglob("*exitcode*")], "stdout": completed.stdout})
                        self.assertEqual(selected_exit.read_text(encoding="utf-8").strip(), f"exit_code={simulated_code}")

    @unittest.skipUnless(os.environ.get("BASH_EXE"), "GNU bash path is required for the fresh-clone fixture")
    def test_fresh_clone_runner_creates_roots_before_mock_preflight(self) -> None:
        bash = Path(os.environ["BASH_EXE"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                "custom_models/docs/benchmark_v2/E5/E5_SCOPE27_VARIANT_MANIFEST.json",
                "scripts/e5_batch4_scope27_gate.py",
                "scripts/st_mgprompt_a8_batch4_gate.py",
                "dataset/sdwpf_model_input_base.parquet",
                "dataset/sdwpf_eval_target.parquet",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            fake_python = root / "fake-python.sh"
            fake_python.write_text(
                "#!/bin/sh\n"
                "case \" $* \" in\n"
                "  *\" -c \"*) echo ABSENT; exit 0 ;;\n"
                "  *\" lock-status \"*) echo '{\"status\":\"ABSENT\"}'; exit 0 ;;\n"
                "  *\" preflight \"*) exit 91 ;;\n"
                "  *) echo '{}'; exit 0 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            fake_tee = root / "tee"
            fake_tee.write_text(
                "#!/bin/sh\n"
                "log=''\n"
                "for value in \"$@\"; do case \"$value\" in -*) ;; *) log=\"$value\" ;; esac; done\n"
                "while IFS= read -r line; do echo \"$line\"; if [ -n \"$log\" ]; then echo \"$line\" >> \"$log\"; fi; done\n",
                encoding="utf-8",
            )
            env = {
                **os.environ,
                "PROJECT_ROOT": _msys_path(root),
                "PYTHON": _msys_path(fake_python),
                "PATH": f"{_msys_path(fake_python.parent)}:{_msys_path(bash.parent)}:/usr/bin",
            }
            completed = subprocess.run(
                [str(bash), str(E5_DOCS / "E5_RUN_ALL_27_BATCH4_LINUX.sh")],
                cwd=PROJECT_ROOT, env=env, check=False,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            self.assertEqual(completed.returncode, 91, completed.stdout)
            for relative in (
                "custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026",
                "custom_models/logs/uniform_bs4/audit/e5_scope27",
                "custom_models/logs/uniform_bs4/audit/e5_scope27/preflight",
                "custom_models/logs/uniform_bs4/audit/e5_scope27/runs",
            ):
                self.assertTrue((root / relative).is_dir(), relative)

    def test_ordinary_launchers_never_shutdown_and_windows_uses_ready(self) -> None:
        for name in ("E5_A8_BATCH4_LINUX.sh", "E5_RUN_ALL_27_BATCH4_LINUX.sh"):
            script = (E5_DOCS / name).read_text(encoding="utf-8")
            self.assertNotIn("shutdown -h", script)
            self.assertNotIn("/usr/bin/shutdown", script)
        windows = (E5_DOCS / "E5_A8_BATCH4_WINDOWS_FORMAL_COMMANDS.ps1").read_text(encoding="utf-8")
        self.assertNotIn("READY_A8_" + "BATCH4_PREREQUISITE", windows)
        self.assertNotIn("'--model', 'st_mgprompt_a8'", windows)
        self.assertIn("$result.Json.readiness.status -ne 'READY'", windows)


if __name__ == "__main__":
    unittest.main()
