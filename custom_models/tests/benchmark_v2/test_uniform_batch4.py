import unittest
from benchmark_v2.runtime import ProviderBatchIterable

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
