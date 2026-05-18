import unittest

import numpy as np

from jnlm.coherence import local_coherence
from jnlm.residue import residue_density, wrapped_phase_difference


class TestMetrics(unittest.TestCase):
    def test_coherence_identical_is_one(self):
        x = np.ones((16, 16), dtype=np.complex64) * (1 + 1j)
        coh = local_coherence(x, x, win=5)
        self.assertGreater(float(np.mean(coh[3:-3, 3:-3])), 0.999)

    def test_residue_flat_zero(self):
        phase = np.zeros((10, 10), dtype=np.float32)
        self.assertEqual(residue_density(phase), 0.0)

    def test_wrapped_difference(self):
        before = np.array([np.pi - 0.1])
        after = np.array([-np.pi + 0.1])
        diff = wrapped_phase_difference(after, before)
        self.assertAlmostEqual(float(diff[0]), 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
