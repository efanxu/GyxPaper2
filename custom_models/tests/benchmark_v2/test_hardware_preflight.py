from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_v2 import hardware_preflight
from benchmark_v2.original_scope26 import CURRENT_SCOPE26_ID


class HardwarePreflightTests(unittest.TestCase):
    def test_formal_shape_and_explicit_identity(self) -> None:
        value = hardware_preflight.preflight_identity("gcn", formal_scope_id=CURRENT_SCOPE26_ID)
        self.assertEqual((value["batch_size"], value["lookback"], value["node_count"], value["feature_count"], value["horizon"]), (4, 144, 134, 16, 10))

    def test_pass_requires_forward_backward_finite_and_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = hardware_preflight.preflight_identity("gcn", formal_scope_id=CURRENT_SCOPE26_ID)
            path = hardware_preflight.preflight_artifact_path("gcn", root=root)
            path.parent.mkdir(parents=True)
            payload = {**value, "status": "PASS", "forward_pass": True, "backward_pass": True, "finite": True, "output_shape": [4, 134, 10]}
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNotNone(hardware_preflight.read_matching_pass("gcn", root=root, formal_scope_id=CURRENT_SCOPE26_ID, source_revision="one"))
            payload["finite"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(hardware_preflight.read_matching_pass("gcn", root=root, formal_scope_id=CURRENT_SCOPE26_ID, source_revision="two"))


if __name__ == "__main__":
    unittest.main()
