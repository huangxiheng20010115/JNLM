import unittest

import numpy as np

from jnlm.config import JNLMConfig
from jnlm.core import jnlm_filter_insar, jnlm_filter_slc_pair


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

    def test_insar_wrapper_matches_guided_pair(self):
        rng = np.random.default_rng(7)
        amp_m = np.abs(rng.normal(size=(10, 11))).astype(np.float32) + 0.1
        amp_s = np.abs(rng.normal(size=(10, 11))).astype(np.float32) + 0.1
        phase = rng.normal(size=(10, 11)).astype(np.float32)
        cfg = JNLMConfig(patch_size=3, search_window_size=5, use_single=True)

        wrapped = jnlm_filter_insar(amp_m, amp_s, phase, config=cfg)
        direct = jnlm_filter_slc_pair(
            amp_m.astype(np.complex64),
            (amp_s * np.exp(-1j * phase)).astype(np.complex64),
            config=cfg,
        )

        self.assertTrue(np.allclose(wrapped.master_after, direct.master_after, atol=1e-6))
        self.assertTrue(np.allclose(wrapped.slave_after, direct.slave_after, atol=1e-6))
        self.assertTrue(np.allclose(wrapped.phase_after, direct.phase_after, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
