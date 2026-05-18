import unittest

import numpy as np

from jnlm.config import JNLMConfig
from jnlm.core import jnlm_filter_slc_pair


class TestCore(unittest.TestCase):
    def test_shape_and_finite(self):
        rng = np.random.default_rng(42)
        master = (rng.normal(size=(16, 18)) + 1j * rng.normal(size=(16, 18))).astype(np.complex64)
        slave = (rng.normal(size=(16, 18)) + 1j * rng.normal(size=(16, 18))).astype(np.complex64)
        cfg = JNLMConfig(patch_size=3, search_window_size=5, use_single=True)
        result = jnlm_filter_slc_pair(master, slave, config=cfg)
        self.assertEqual(result.master_after.shape, master.shape)
        self.assertEqual(result.slave_after.shape, slave.shape)
        self.assertTrue(np.all(np.isfinite(result.ifg_after)))
        self.assertTrue(np.all(np.isfinite(result.phase_after)))

    def test_constant_region_stability(self):
        master = np.full((12, 12), 2.0 + 1.0j, dtype=np.complex64)
        slave = np.full((12, 12), 0.5 - 0.25j, dtype=np.complex64)
        cfg = JNLMConfig(patch_size=3, search_window_size=5, use_single=True)
        result = jnlm_filter_slc_pair(master, slave, config=cfg)
        self.assertTrue(np.allclose(result.master_after, master, atol=1e-5))
        self.assertTrue(np.allclose(result.slave_after, slave, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
