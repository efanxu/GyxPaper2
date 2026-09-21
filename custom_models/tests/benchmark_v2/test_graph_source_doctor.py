from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmark_v2.graph.source_doctor import inspect_graph_source, repair_graph_source


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCATION_RELATIVE_PATH = "dataset/sdwpf_turb_location_elevation.csv"


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


class GraphSourceDoctorTests(unittest.TestCase):
    def _temporary_repo(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        holder = tempfile.TemporaryDirectory()
        repo = Path(holder.name)
        (repo / "dataset").mkdir()
        (repo / "custom_models/src/benchmark_v2").mkdir(parents=True)
        shutil.copy2(
            PROJECT_ROOT / LOCATION_RELATIVE_PATH,
            repo / LOCATION_RELATIVE_PATH,
        )
        shutil.copytree(
            PROJECT_ROOT / "custom_models/src/benchmark_v2/protocol",
            repo / "custom_models/src/benchmark_v2/protocol",
        )
        shutil.copy2(PROJECT_ROOT / ".gitattributes", repo / ".gitattributes")
        _git(repo, "init")
        _git(repo, "config", "user.email", "test@example.invalid")
        _git(repo, "config", "user.name", "Graph Source Doctor Test")
        _git(repo, "add", "--", ".")
        _git(repo, "commit", "-m", "fixture")
        return holder, repo

    def test_repository_rule_and_frozen_source_status(self):
        text = (PROJECT_ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn(
            "dataset/sdwpf_turb_location_elevation.csv text eol=lf",
            text.splitlines(),
        )
        status = inspect_graph_source(PROJECT_ROOT)
        self.assertEqual(status["status"], "PASS", status)
        self.assertEqual(status["relative_path"], LOCATION_RELATIVE_PATH)
        self.assertEqual(status["worktree_sha256"], status["expected_sha256"])
        self.assertEqual(status["git_head_blob_sha256"], status["expected_sha256"])
        self.assertFalse(status["contains_crlf"])
        self.assertFalse(status["repair_required"])

    def test_crlf_preview_apply_and_real_content_refusal(self):
        holder, repo = self._temporary_repo()
        try:
            _git(repo, "config", "core.autocrlf", "true")
            path = repo / LOCATION_RELATIVE_PATH
            original = path.read_bytes()
            path.write_bytes(original.replace(b"\n", b"\r\n"))

            status = inspect_graph_source(repo)
            self.assertEqual(status["status"], "REPAIR_REQUIRED", status)
            self.assertTrue(status["repair_safe"])
            self.assertTrue(status["contains_crlf"])
            preview = repair_graph_source(
                repo,
                apply=False,
                receipt_root=repo / "receipts",
            )
            self.assertEqual(preview["status"], "PREVIEW_ONLY", preview)
            self.assertEqual(path.read_bytes(), original.replace(b"\n", b"\r\n"))

            applied = repair_graph_source(
                repo,
                apply=True,
                receipt_root=repo / "receipts",
            )
            self.assertEqual(applied["status"], "PASS", applied)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(inspect_graph_source(repo)["status"], "PASS")

            modified = original.replace(b"0", b"1", 1)
            path.write_bytes(modified)
            unsafe = inspect_graph_source(repo)
            self.assertEqual(unsafe["status"], "REPAIR_REQUIRED", unsafe)
            self.assertFalse(unsafe["repair_safe"])
            refused = repair_graph_source(
                repo,
                apply=True,
                receipt_root=repo / "receipts",
            )
            self.assertEqual(refused["status"], "REPAIR_REFUSED", refused)
            self.assertEqual(path.read_bytes(), modified)
        finally:
            holder.cleanup()

    def test_head_mismatch_is_repository_inconsistency_and_autocrlf_false_is_clean(self):
        holder, repo = self._temporary_repo()
        try:
            _git(repo, "config", "core.autocrlf", "false")
            path = repo / LOCATION_RELATIVE_PATH
            original = path.read_bytes()
            path.write_bytes(original.replace(b"0", b"1", 1))
            _git(repo, "add", "--", LOCATION_RELATIVE_PATH)
            _git(repo, "commit", "-m", "invalid frozen source")
            path.write_bytes(original.replace(b"0", b"1", 1).replace(b"\n", b"\r\n"))
            status = inspect_graph_source(repo)
            self.assertEqual(status["status"], "REPOSITORY_PROTOCOL_INCONSISTENCY", status)
            before = path.read_bytes()
            refused = repair_graph_source(
                repo,
                apply=True,
                receipt_root=repo / "receipts",
            )
            self.assertEqual(refused["status"], "REPOSITORY_PROTOCOL_INCONSISTENCY")
            self.assertEqual(path.read_bytes(), before)
        finally:
            holder.cleanup()


if __name__ == "__main__":
    unittest.main()
