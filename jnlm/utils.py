from __future__ import annotations

import numpy as np


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap radians to [-pi, pi)."""

    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def as_bool_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    arr = np.asarray(mask)
    if arr.shape != shape:
        raise ValueError(f"valid_mask shape {arr.shape} does not match {shape}")
    return arr.astype(bool, copy=False)


def validate_slc_pair(master: np.ndarray, slave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(master)
    s = np.asarray(slave)
    if m.shape != s.shape:
        raise ValueError(f"master/slave shapes differ: {m.shape} vs {s.shape}")
    if m.ndim != 2:
        raise ValueError(f"master/slave must be 2-D arrays, got ndim={m.ndim}")
    if not np.iscomplexobj(m) or not np.iscomplexobj(s):
        raise ValueError("master_cpx and slave_cpx must be complex arrays")
    if not np.all(np.isfinite(m)) or not np.all(np.isfinite(s)):
        raise ValueError("master_cpx and slave_cpx contain NaN or Inf")
    return m, s
