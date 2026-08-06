from __future__ import annotations
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from benchmark_v2 import model_cli
from benchmark_v2.cli import main
from benchmark_v2.data import SDWPFDataProvider
from benchmark_v2.original_scope26 import CURRENT_SCOPE26_ID
from benchmark_v2.protocol import load_protocol

class OriginalPersistenceFormalFixtureTests(unittest.TestCase):

    def test_tiny_formal_evaluate_only_completes_with_non_trainable_artifacts(self):
        protocol = load_protocol()
        rng = np.random.default_rng(2026)
        timestamp_count = 1600
        node_count = 4
        inputs = rng.normal(size=(timestamp_count, node_count, 16)).astype(np.float32)
        inputs[..., 15] = rng.uniform(10.0, 1000.0, size=(timestamp_count, node_count)).astype(np.float32)
        targets = rng.uniform(10.0, 1000.0, size=(timestamp_count, node_count)).astype(np.float32)
        mask = np.ones_like(targets, dtype=bool)
        timestamps = pd.date_range('2026-01-01', periods=timestamp_count, freq='10min').tolist()
        provider = SDWPFDataProvider.from_arrays(inputs, targets, mask, node_ids=list(range(1, node_count + 1)), timestamps=timestamps, protocol=protocol)

        def apply_fixture_scope_identity(runtime, **kwargs):
            runtime.effective_config.update({'active_scope_id': CURRENT_SCOPE26_ID, 'run_id': kwargs['run_id'], 'output_root': str(kwargs['output_root']), 'formal_training': False})
            return runtime
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / 'results'
            with patch.object(model_cli.SDWPFDataProvider, 'from_files', return_value=provider), patch.object(model_cli, 'apply_current_scope_identity', side_effect=apply_fixture_scope_identity), patch('benchmark_v2.hardware_preflight.preflight_identity', return_value={'scope_id': CURRENT_SCOPE26_ID}):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = main(['evaluate-only', '--model', 'persistence', '--input-path', str(Path(temp) / 'fixture-input.parquet'), '--target-path', str(Path(temp) / 'fixture-target.parquet'), '--output-root', str(output_root), '--run-id', 'Persistence_bs4_seed2026', '--device', 'cpu', '--training-profile', 'uniform_train_batch4_v1', '--formal-scope-id', CURRENT_SCOPE26_ID, '--source-revision', 'fixture-revision'])
                result = json.loads(stdout.getvalue())
            run_dir = output_root / 'Persistence_bs4_seed2026'
            status = json.loads((run_dir / 'run_status.json').read_text(encoding='utf-8'))
            effective = json.loads((run_dir / 'effective_config.json').read_text(encoding='utf-8'))
            artifact = json.loads((run_dir / 'artifact_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(exit_code, 0)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['artifact_validation']['status'], 'PASS')
            self.assertEqual(status['status'], 'COMPLETED')
            self.assertEqual(status['exit_code'], 0)
            self.assertFalse(status['formal_training'])
            self.assertEqual(status['artifact_profile'], 'NON_TRAINABLE')
            self.assertFalse(effective['formal_training'])
            self.assertEqual(effective['active_scope_id'], CURRENT_SCOPE26_ID)
            self.assertEqual(effective['training_batch_profile_id'], 'uniform_train_batch4_v1')
            self.assertEqual(artifact['artifact_profile'], 'NON_TRAINABLE')
            for horizon in (3, 6, 10):
                metrics_path = run_dir / f'metrics_eval_h{horizon}.json'
                self.assertTrue(metrics_path.is_file())
                metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
                for metric in ('MAE', 'RMSE', 'R2', 'Score'):
                    self.assertTrue(math.isfinite(float(metrics[metric])), f'non-finite H{horizon} {metric}: {metrics}')
                self.assertGreater(int(metrics['valid_target_count']), 0)
            self.assertEqual(effective['test_batch_size'], 4)
            self.assertTrue((run_dir / 'metrics.csv').is_file())
            self.assertFalse((run_dir / 'best_checkpoint.pt').exists())
if __name__ == '__main__':
    unittest.main()
