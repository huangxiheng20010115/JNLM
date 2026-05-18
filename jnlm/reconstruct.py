from __future__ import annotations

import numpy as np


def finalize_weighted_channels(
    wsum: np.ndarray,
    mr_acc: np.ndarray,
    mi_acc: np.ndarray,
    sr_acc: np.ndarray,
    si_acc: np.ndarray,
    *,
    eps: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wsum = np.maximum(wsum, np.asarray(eps, dtype=wsum.dtype))
    return mr_acc / wsum, mi_acc / wsum, sr_acc / wsum, si_acc / wsum
