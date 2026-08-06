import tempfile
import unittest
from pathlib import Path
from benchmark_v2.artifacts import PROFILES, atomic_write_json, safe_run_dir, validate_run, write_status
from benchmark_v2.errors import ArtifactError

class ArtifactTests(unittest.TestCase):

    def test_path_escape_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'root'
            run = safe_run_dir(root, 'suite', 'run')
            run.mkdir(parents=True)
            with self.assertRaises(ArtifactError):
                safe_run_dir(root, 'suite', 'run')
            with self.assertRaises(ArtifactError):
                safe_run_dir(root, '..', 'outside')

    def test_failed_and_incomplete_are_not_completed(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / 'run'
            run.mkdir()
            write_status(run, status='FAILED', run_mode='smoke', artifact_profile='FAILED', error_type='X', error_message='bad')
            self.assertEqual(validate_run(run)['run_status'], 'FAILED')
            run2 = Path(td) / 'run2'
            run2.mkdir()
            write_status(run2, status='COMPLETED', run_mode='smoke', artifact_profile='SMOKE')
            with self.assertRaises(ArtifactError):
                validate_run(run2)
if __name__ == '__main__':
    unittest.main()
