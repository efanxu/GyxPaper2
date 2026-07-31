import inspect
import unittest

from benchmark_v2.engine import Trainer


class LeakageTests(unittest.TestCase):
    def test_fit_has_no_test_loader_parameter(self):
        params = list(inspect.signature(Trainer.fit).parameters)
        self.assertEqual(params, ["self", "train_loader", "val_loader"])
        self.assertNotIn("test_loader", params)


if __name__ == "__main__": unittest.main()

