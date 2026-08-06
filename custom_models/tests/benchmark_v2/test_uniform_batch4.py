import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from benchmark_v2.artifacts import validate_run, write_status
from benchmark_v2.checkpointing import CheckpointManager
from benchmark_v2.experiments.e5_common_loss import a8_reference, aggregation
from benchmark_v2.experiments.e5_common_loss.a8_reference import validate_a8_reference
from benchmark_v2.experiments.e5_common_loss.contracts import BATCH4_A8_REFERENCE_ID
from benchmark_v2.experiments.e5_common_loss.readiness import build_readiness
from benchmark_v2.hardware_preflight import preflight_identity
from benchmark_v2.model_cli import _write_common
from benchmark_v2.model_runtime import build_model_runtime
from benchmark_v2.protocol import load_protocol
from benchmark_v2.registry import load_registry
from benchmark_v2.runtime import ProviderBatchIterable
from benchmark_v2.training_profiles import PROFILE_ALLOWLIST, PROFILE_SCHEMA_VERSION, UNIFORM_BATCH4_PROFILE_ID, apply_training_profile, load_training_profile, resolved_batch_sizes
PROFILE_ID = UNIFORM_BATCH4_PROFILE_ID

class UniformBatch4ProfileTests(unittest.TestCase):

    def test_provider_batches_are_exactly_four_without_sample_change(self):

        class FakeProvider:

            def __init__(self):
                self.starts = {'train': list(range(10))}

            def batches(self, split, batch_size):
                return [list(self.starts[split])]
        provider = FakeProvider()
        batches = list(ProviderBatchIterable(provider, 'train', 4))
        self.assertEqual([len(batch) for batch in batches], [4, 4, 2])
        self.assertEqual(provider.starts['train'], list(range(10)))
if __name__ == '__main__':
    unittest.main()
