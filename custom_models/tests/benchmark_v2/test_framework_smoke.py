import unittest

from benchmark_v2.adapters import NodeSharedAdapter
from benchmark_v2.contracts import BenchmarkBatch


class FrameworkSmokeContractTests(unittest.TestCase):
    def test_smoke_shape_contract_is_not_formal_node_count(self):
        import torch
        b = BenchmarkBatch(torch.zeros(2,12,4,16), torch.zeros(2,4,10), torch.ones(2,4,10,dtype=torch.bool), [1,2], [12,12], [1,2,3,4], metadata={"contains_future_target":False})
        b.validate(expected_nodes=4)
        self.assertEqual(b.node_count, 4)
        self.assertEqual(NodeSharedAdapter.__doc__.split(":",1)[0], "Shared parameters per node")


if __name__ == "__main__": unittest.main()

