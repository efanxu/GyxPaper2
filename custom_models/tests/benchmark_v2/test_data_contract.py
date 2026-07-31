import unittest
import numpy as np

from benchmark_v2.data import SDWPFDataProvider, StandardScaler
from benchmark_v2.data.signatures import feature_order_hash, node_order_hash
from benchmark_v2.data.windows import split_indices, window_start_indices


class DataContractTests(unittest.TestCase):
    def test_split_windows_stride_and_train_only_scaler(self):
        self.assertEqual(split_indices(100), {"train":(0,80),"val":(80,90),"test":(90,100)})
        self.assertEqual(window_start_indices(0, 30, 5, 3, 6), [5,11,17,23])
        values = np.arange(10, dtype=np.float32)[:,None]
        s = StandardScaler().fit(values[:8])
        self.assertAlmostEqual(float(s.mean[0]), 3.5)
        self.assertAlmostEqual(float(s.inverse_transform(s.transform(values))[0,0]), 0.0, places=6)

    def test_arrays_alignment_mask_and_signatures(self):
        rng = np.random.default_rng(2026)
        x = rng.normal(size=(80,4,16)).astype(np.float32)
        y = rng.normal(size=(80,4)).astype(np.float32)
        mask = np.ones((80,4), dtype=bool); mask[0,0] = False
        p = SDWPFDataProvider.from_arrays(x,y,mask,node_ids=[10,11,12,13],timestamps=list(range(80)))
        self.assertEqual(p.signature()["node_order_hash"], node_order_hash([10,11,12,13]))
        self.assertEqual(p.signature()["feature_order_hash"], feature_order_hash(p.feature_names))
        for split in ("train","val","test"):
            for w in p.windows(split):
                self.assertEqual(w.target.shape, w.mask.shape)
                self.assertEqual(w.x.shape[-1], 16)
                self.assertTrue(w.end_index >= p.split_indices[split][0] + 144)


if __name__ == "__main__": unittest.main()
