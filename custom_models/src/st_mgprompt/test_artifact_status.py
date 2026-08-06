from __future__ import annotations
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from st_mgprompt.artifact_status import inspect_variant_artifacts
from st_mgprompt.experiment_protocol import apply_variant
from st_mgprompt.formal_runner import run_family

def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

class ArtifactDrivenStatusTests(unittest.TestCase):

    def test_p5_dry_run_is_independent_of_p4_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_family('precision', ['--variants', 'P5', '--dry-run', '--output-root', tmp, '--run-id', 'precision_ablation_fixed_dual_seed2026'])
        self.assertEqual(result['failed_variants'], [])
        self.assertEqual(result['statuses'][0]['final_status'], 'pending')
if __name__ == '__main__':
    unittest.main()
