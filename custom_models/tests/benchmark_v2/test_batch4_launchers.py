from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
A7_DOCS = PROJECT_ROOT / "custom_models/docs/benchmark_v2/A7"


class Batch4LauncherTests(unittest.TestCase):
    def test_a7_autoshutdown_preserves_required_order(self) -> None:
        script = (A7_DOCS / "A7_BATCH4_LINUX_AUTOSHUTDOWN.sh").read_text(
            encoding="utf-8"
        )
        order = (
            script.index('"$BASH_EXECUTABLE" "$RUNNER"'),
            script.index("main_code=${PIPESTATUS[0]}"),
            script.index("a7_batch4_autoshutdown.exitcode"),
            script.index('"$SYNC_COMMAND"'),
            script.index('"$SHUTDOWN_COMMAND" -h now'),
        )
        self.assertEqual(tuple(sorted(order)), order)

    def test_ordinary_a7_launchers_never_shutdown(self) -> None:
        linux = (A7_DOCS / "A7_BATCH4_LINUX.sh").read_text(encoding="utf-8")
        windows = (A7_DOCS / "A7_BATCH4_WINDOWS_FORMAL_COMMANDS.ps1").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("shutdown -h", linux)
        self.assertNotIn("/usr/bin/shutdown", linux)
        self.assertNotIn("PREREQUISITE", windows)
        self.assertIn("$result.Json.readiness.status -ne 'READY'", windows)


if __name__ == "__main__":
    unittest.main()
