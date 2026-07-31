import json
import unittest
from pathlib import Path

from benchmark_v2.data.signatures import feature_order_hash
from benchmark_v2.protocol import check_protocol, load_protocol


class ProtocolTests(unittest.TestCase):
    def test_json_hash_features_and_fixed_fields(self):
        path = Path(__file__).parents[2] / "src" / "benchmark_v2" / "protocol" / "benchmark_protocol_v1.json"
        json.loads(path.read_text(encoding="utf-8"))
        p1, p2 = load_protocol(), load_protocol()
        self.assertEqual(p1.protocol_hash, p2.protocol_hash)
        self.assertEqual(feature_order_hash(p1["ordered_input_features"]), p1["feature_order_hash"])
        self.assertEqual(check_protocol(p1)["status"], "PASS")
        self.assertEqual(p1["eval_horizons"], [3, 6, 10])
        self.assertEqual(p1["target_loss_space"], "normalized_target_space")
        self.assertEqual(p1["future_observed_covariates"], "disabled")


if __name__ == "__main__":
    unittest.main()

