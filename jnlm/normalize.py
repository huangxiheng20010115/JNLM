from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustNormState:
    med: float
    sc: float
    dtype: np.dtype


def robust_norm(x: np.ndarray) -> tuple[np.ndarray, RobustNormState]:
    """MATLAB-compatible robust normalization.

    Mirrors `robust_norm_matlab.m`: median/MAD scaling with standard-deviation
    fallback and final cast back to the input dtype.
    """

    arr = np.asarray(x)
    dtype = arr.dtype
    xd = arr.astype(np.float64, copy=False).ravel()
    med = float(np.median(xd))
    madv = float(np.median(np.abs(xd - med)))
    sc = 1.4826 * madv

    if not np.isfinite(sc) or sc < 1.0e-6:
        sc = float(np.std(xd))
        if not np.isfinite(sc) or sc < 1.0e-6:
            sc = 1.0

    y = (arr.astype(np.float64, copy=False) - med) / sc
    return y.astype(dtype, copy=False), RobustNormState(med=med, sc=sc, dtype=np.dtype(dtype))


def inv_robust_norm(y: np.ndarray, state: RobustNormState) -> np.ndarray:
    """Inverse of :func:`robust_norm`, matching `inv_robust_norm_matlab.m`."""

    arr = np.asarray(y)
    x = arr.astype(np.float64, copy=False) * state.sc + state.med
    return x.astype(arr.dtype, copy=False)
