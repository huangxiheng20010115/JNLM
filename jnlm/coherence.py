from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from .utils import as_bool_mask


def local_coherence(
    master: np.ndarray,
    slave: np.ndarray,
    valid_mask: np.ndarray | None = None,
    win: int = 11,
) -> np.ndarray:
    """Standard masked InSAR coherence map.

    `|mean(M * conj(S))| / sqrt(mean(|M|^2) * mean(|S|^2))`
    using only valid pixels in each local window.
    """

    if win < 1 or win % 2 != 1:
        raise ValueError("win must be odd and >= 1")
    m = np.asarray(master)
    s = np.asarray(slave)
    if m.shape != s.shape:
        raise ValueError(f"master/slave shapes differ: {m.shape} vs {s.shape}")
    mask = as_bool_mask(valid_mask, m.shape).astype(np.float64)
    ker = np.ones((win, win), dtype=np.float64)

    num = convolve2d((m * np.conj(s)) * mask, ker, mode="same", boundary="fill", fillvalue=0)
    den1 = convolve2d((np.abs(m) ** 2) * mask, ker, mode="same", boundary="fill", fillvalue=0)
    den2 = convolve2d((np.abs(s) ** 2) * mask, ker, mode="same", boundary="fill", fillvalue=0)
    cnt = convolve2d(mask, ker, mode="same", boundary="fill", fillvalue=0)
    cnt = np.maximum(cnt, 1.0)

    num = num / cnt
    den1 = den1 / cnt
    den2 = den2 / cnt
    coh = np.abs(num) / np.sqrt(den1 * den2 + np.finfo(np.float64).eps)
    coh[~np.isfinite(coh)] = 0
    return coh.astype(np.float32)


def coherence_stats(coh: np.ndarray, valid_mask: np.ndarray | None = None) -> dict[str, float]:
    arr = np.asarray(coh)
    mask = as_bool_mask(valid_mask, arr.shape) if valid_mask is not None else np.isfinite(arr)
    vals = arr[mask & np.isfinite(arr)]
    if vals.size == 0:
        return {"coh_mean": float("nan"), "coh_p25": float("nan")}
    return {"coh_mean": float(np.mean(vals)), "coh_p25": float(np.percentile(vals, 25))}
