from __future__ import annotations
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from st_mgprompt.artifact_status import inspect_variant_artifacts
from st_mgprompt.experiment_protocol import apply_variant, get_variant
from st_mgprompt.formal_runner import _command, _resume_plan, run_family

def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

class ArtifactDrivenStatusTests(unittest.TestCase):

    def test_p5_dry_run_is_independent_of_p4_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_family('precision', ['--variants', 'P5', '--dry-run', '--output-root', tmp, '--run-id', 'precision_ablation_fixed_dual_seed2026'])
        self.assertEqual(result['failed_variants'], [])
        self.assertEqual(result['statuses'][0]['final_status'], 'pending')

    def test_resume_uses_last_checkpoint_only_for_matching_artifacts(self) -> None:
        args = SimpleNamespace(resume=True, run_smoke=False, full_shape_smoke=False, extra_args=None)
        variant = get_variant('A4', 'component_ablation')
        with tempfile.TemporaryDirectory() as tmp:
            variant_dir = Path(tmp) / 'A4' / 'STMGPrompt_ComponentAblation'
            variant_dir.mkdir(parents=True)
            audit = {'config_match': True, 'protocol_consistent': True}
            (variant_dir / 'last_checkpoint.pt').write_bytes(b'checkpoint-placeholder')
            self.assertEqual(
                _resume_plan(args, audit, variant_dir, 'A4'),
                (True, 'last_checkpoint'),
            )
            command = _command(args, 'component_ablation', Path(tmp), variant, resume=True)
            self.assertIn('--resume', command)

            with self.assertRaisesRegex(ValueError, 'refusing --resume'):
                _resume_plan(
                    args,
                    {'config_match': False, 'protocol_consistent': False, 'config_differences': ['macro_prompt_len']},
                    variant_dir,
                    'A4',
                )
if __name__ == '__main__':
    unittest.main()
