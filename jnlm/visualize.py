from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from .residue import residue_map, wrapped_phase_difference




def phase_rgb(phi: np.ndarray) -> np.ndarray:
    """Render wrapped phase with a circular HSV color wheel."""

    hue = ((phi + np.pi) / (2 * np.pi)).astype(np.float32)
    return hsv_to_rgb(np.dstack([hue, np.ones_like(hue), np.ones_like(hue)]))


def _decimate(arr: np.ndarray, max_side: int = 900) -> np.ndarray:
    h, w = arr.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / max_side)))
    return arr[::step, ::step]


def save_comparison_png(
    path: str | Path,
    *,
    phase_before: np.ndarray,
    phase_after: np.ndarray,
    coh_before: np.ndarray,
    coh_after: np.ndarray,
    valid_mask: np.ndarray,
    title: str = "",
) -> None:
    """Save a compact 2x4 comparison figure for a tile."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    delta = wrapped_phase_difference(phase_after, phase_before)
    res_before, res_valid = residue_map(phase_before, valid_mask)
    res_after, _ = residue_map(phase_after, valid_mask)

    res_after_display = np.abs(res_after).astype(float) * res_valid.astype(float)
    panels = [
        ("Raw phase", _decimate(phase_rgb(phase_before)), None, None),
        ("JNLM phase", _decimate(phase_rgb(phase_after)), None, None),
        ("Wrapped diff", _decimate(delta), "twilight", (-np.pi, np.pi)),
        ("Valid mask", _decimate(valid_mask.astype(float)), "gray", (0, 1)),
        ("Raw coherence", _decimate(coh_before), "viridis", (0, 1)),
        ("JNLM coherence", _decimate(coh_after), "viridis", (0, 1)),
        ("Raw residue", _decimate(np.abs(res_before).astype(float)), "magma", (0, 1)),
        ("JNLM residue", _decimate(res_after_display), "magma", (0, 1)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=11)
    for ax, (name, img, cmap, lim) in zip(axes.flat, panels):
        if cmap is None:
            ax.imshow(img)
        else:
            im = ax.imshow(img, cmap=cmap, vmin=lim[0], vmax=lim[1])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(name, fontsize=10)
        ax.set_axis_off()
    fig.savefig(out, dpi=140)
    plt.close(fig)
