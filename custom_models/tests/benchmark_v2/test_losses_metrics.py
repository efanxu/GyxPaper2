import unittest
import numpy as np
import torch
from benchmark_v2.losses import get_loss
from benchmark_v2.metrics import evaluate_horizons, official_score, regression_metrics

class LossMetricsTests(unittest.TestCase):

    def test_masked_losses_zero_valid_and_backward(self):
        p = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]], requires_grad=True)
        y = torch.zeros_like(p)
        m = torch.tensor([[[1, 0, 1], [1, 1, 0]]], dtype=torch.bool)
        loss = get_loss('masked_mse')(p, y, m)
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertIsNotNone(p.grad)
        self.assertIsNone(get_loss('masked_mse')(p.detach(), y, torch.zeros_like(m)))

    def test_metrics_horizons_clip_and_parity(self):
        p = np.array([[[-10, 100, 1600], [20, 110, 1500]]], dtype=float)
        y = np.array([[[0, 100, 1500], [10, 100, 1400]]], dtype=float)
        m = np.ones_like(p, dtype=bool)
        m[0, 1, 1] = False
        rows = evaluate_horizons(p, y, m, (3, 6, 10), num_nodes=2, physical_clip=(0, 1500))
        self.assertEqual([r['horizon'] for r in rows], [3, 6, 10])
        self.assertLessEqual(rows[0]['Score'], rows[0]['Score'] + 1e-09)
        self.assertEqual(regression_metrics(p, y, np.zeros_like(m), num_nodes=2)['status'], 'NO_VALID_TARGET')
        sys_path = __import__('pathlib').Path(__file__).parents[2] / 'src'
        import sys
        sys.path.insert(0, str(sys_path))
        from st_mgprompt.metrics import masked_official_align_score_kw
        self.assertAlmostEqual(official_score(p, y, m, 2), masked_official_align_score_kw(y.transpose(0, 2, 1), p.transpose(0, 2, 1), m.transpose(0, 2, 1), 2), places=12)

    def test_official_score_ignores_masked_nan_targets(self):
        pred = np.array([[[100.0, 200.0, 300.0]]])
        target = np.array([[[100.0, np.nan, 300.0]]])
        mask = np.array([[[True, False, True]]])
        self.assertEqual(official_score(pred, target, mask, num_nodes=1), 0.0)
if __name__ == '__main__':
    unittest.main()
