from __future__ import annotations

import numpy as np
from scipy import ndimage


def offset_list(search_window_size: int) -> list[tuple[int, int]]:
    hsw = search_window_size // 2
    return [
        (dy, dx)
        for dy in range(-hsw, hsw + 1)
        for dx in range(-hsw, hsw + 1)
        if not (dy == 0 and dx == 0)
    ]


def smooth_patch_distance(d: np.ndarray, patch_size: int, gauss_ps: float) -> np.ndarray:
    """Smooth per-pixel channel distance over the patch.

    MATLAB uses `imgaussfilt(..., FilterSize=patch_size, Padding='symmetric')`
    or `imboxfilt(..., Padding='symmetric')`. SciPy boundary handling is very
    close but not bit-identical at borders; this is documented in README.
    """

    if gauss_ps > 0:
        g = gaussian_kernel_1d(patch_size, gauss_ps, d.dtype)
        out = ndimage.convolve1d(d, g, axis=0, mode="reflect")
        out = ndimage.convolve1d(out, g, axis=1, mode="reflect")
        return out
    return ndimage.uniform_filter(d, size=patch_size, mode="reflect")


def gaussian_kernel_1d(patch_size: int, gauss_ps: float, dtype: np.dtype) -> np.ndarray:
    radius = patch_size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    sigma = float(gauss_ps)
    g = np.exp(-(x * x) / (2.0 * sigma * sigma))
    g = g / np.sum(g)
    return g.astype(dtype, copy=False)


def smooth_patch_distance_precomputed(
    d: np.ndarray,
    *,
    patch_size: int,
    gauss_ps: float,
    gaussian_kernel: np.ndarray | None,
) -> np.ndarray:
    if gauss_ps > 0:
        if gaussian_kernel is None:
            gaussian_kernel = gaussian_kernel_1d(patch_size, gauss_ps, d.dtype)
        out = ndimage.convolve1d(d, gaussian_kernel, axis=0, mode="reflect")
        out = ndimage.convolve1d(out, gaussian_kernel, axis=1, mode="reflect")
        return out
    return ndimage.uniform_filter(d, size=patch_size, mode="reflect")


def shared_weight_from_distance(
    d: np.ndarray,
    *,
    patch_size: int,
    h: float,
    gauss_ps: float,
    dtype: np.dtype,
    gaussian_kernel: np.ndarray | None = None,
    invden: float | None = None,
) -> np.ndarray:
    if invden is None:
        patch_n = float(patch_size * patch_size)
        denom = (float(h) * float(h)) * (4.0 * patch_n) + np.finfo(np.float64).eps
        invden = 1.0 / denom
    d2 = smooth_patch_distance_precomputed(
        d,
        patch_size=patch_size,
        gauss_ps=gauss_ps,
        gaussian_kernel=gaussian_kernel,
    )
    w = np.exp(-d2.astype(np.float64, copy=False) * float(invden))
    return w.astype(dtype, copy=False)
