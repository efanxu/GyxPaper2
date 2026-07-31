import unittest

import torch

from benchmark_v2.adapters import NativeSpatiotemporalAdapter, NodeSharedAdapter
from benchmark_v2.contracts import BenchmarkBatch, GraphContext
from benchmark_v2.errors import ContractError


def batch(B=2, N=4, T=12, C=16, H=10):
    return BenchmarkBatch(torch.randn(B,T,N,C), torch.randn(B,N,H), torch.ones(B,N,H,dtype=torch.bool), list(range(B)), list(range(B)), list(range(N)), metadata={"contains_future_target":False})


class Tiny(torch.nn.Module):
    def __init__(self, H=10):
        super().__init__(); self.h = torch.nn.Linear(16,H)
    def forward(self, x): return self.h(x[:,-1])


class AdapterTests(unittest.TestCase):
    def test_node_shared_roundtrip(self):
        b = batch(); out = NodeSharedAdapter()(Tiny(), b)
        self.assertEqual(tuple(out.prediction.shape), (2,4,10))

    def test_bad_b_n_and_multivariable_output_rejected(self):
        b = batch()
        class Bad(torch.nn.Module):
            def forward(self, x): return torch.zeros(3, 10)
        with self.assertRaises(ContractError): NodeSharedAdapter()(Bad(), b)
        class Variables(torch.nn.Module):
            def forward(self, x): return torch.zeros(8, 10, 16)
        with self.assertRaises(ContractError): NodeSharedAdapter()(Variables(), b)

    def test_native_and_graph_context(self):
        b = batch()
        class Native(torch.nn.Module):
            def forward(self, x): return torch.zeros(x.shape[0], x.shape[2], 10)
        self.assertEqual(tuple(NativeSpatiotemporalAdapter()(Native(), b).prediction.shape), (2,4,10))
        graph = NativeSpatiotemporalAdapter(requires_graph=True)
        with self.assertRaises(ContractError): graph(Native(), b)
        context = GraphContext(tuple(range(4)), {"source":"explicit"}, "a"*64, False, False, {"normalization":"none"})
        self.assertEqual(tuple(graph(Native(), b, graph_context=context).prediction.shape), (2,4,10))


if __name__ == "__main__": unittest.main()

