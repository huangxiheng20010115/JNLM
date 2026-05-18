import unittest

import numpy as np

from jnlm.normalize import inv_robust_norm, robust_norm
from jnlm.weights import offset_list, smooth_patch_distance


class TestWeights(unittest.TestCase):
    def test_offset_count(self):
        self.assertEqual(len(offset_list(5)), 24)

    def test_robust_norm_inverse(self):
        x = np.arange(25, dtype=np.float32).reshape(5, 5)
        y, st = robust_norm(x)
        xr = inv_robust_norm(y, st)
        self.assertTrue(np.allclose(xr, x, atol=1e-5))

    def test_smooth_shape(self):
        x = np.ones((7, 9), dtype=np.float32)
        y = smooth_patch_distance(x, patch_size=3, gauss_ps=1.0)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(y)))


if __name__ == "__main__":
    unittest.main()
