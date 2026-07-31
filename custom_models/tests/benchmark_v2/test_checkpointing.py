import tempfile
import unittest
from pathlib import Path

import torch

from benchmark_v2.checkpointing import CheckpointManager
from benchmark_v2.errors import ArtifactError


class CheckpointTests(unittest.TestCase):
    def test_atomic_checkpoint_and_hash_resume(self):
        with tempfile.TemporaryDirectory() as td:
            model = torch.nn.Linear(2,1)
            m = CheckpointManager(Path(td), protocol_hash="a"*64, model_id="x", resolved_config={"a":1}, effective_config={"b":2})
            path = m.save("best_checkpoint.pt", epoch=1, global_step=1, monitor_value=0.5, model=model)
            self.assertTrue(path.exists())
            m.load(path, model)
            bad = CheckpointManager(Path(td), protocol_hash="b"*64, model_id="x", resolved_config={"a":1}, effective_config={"b":2})
            with self.assertRaises(ArtifactError): bad.load(path, model)


if __name__ == "__main__": unittest.main()

