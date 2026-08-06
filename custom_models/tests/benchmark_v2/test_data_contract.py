import unittest
import numpy as np
from benchmark_v2.data import SDWPFDataProvider, StandardScaler
from benchmark_v2.data.windows import split_indices, window_start_indices

class DataContractTests(unittest.TestCase):

    def test_split_windows_stride_and_train_only_scaler(self):
        self.assertEqual(split_indices(100), {'train': (0, 80), 'val': (80, 90), 'test': (90, 100)})
        self.assertEqual(window_start_indices(0, 30, 5, 3, 6), [5, 11, 17, 23])
        values = np.arange(10, dtype=np.float32)[:, None]
        s = StandardScaler().fit(values[:8])
        self.assertAlmostEqual(float(s.mean[0]), 3.5)
        self.assertAlmostEqual(float(s.inverse_transform(s.transform(values))[0, 0]), 0.0, places=6)
if __name__ == '__main__':
    unittest.main()
