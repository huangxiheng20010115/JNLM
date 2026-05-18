from __future__ import annotations

import numpy as np

from .utils import as_bool_mask, wrap_to_pi


def residue_map(phase: np.ndarray, valid_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return wrapped 2x2 residue charge map and valid-cell mask."""

    phi = wrap_to_pi(np.asarray(phase))
    mask = as_bool_mask(valid_mask, phi.shape)
    a = phi[:-1, :-1]
    b = phi[:-1, 1:]
    e = phi[1:, 1:]
    d = phi[1:, :-1]
    s = wrap_to_pi(b - a) + wrap_to_pi(e - b) + wrap_to_pi(d - e) + wrap_to_pi(a - d)
    charge = np.rint(s / (2.0 * np.pi)).astype(np.int8)
    cell_valid = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, 1:] & mask[1:, :-1]
    charge = np.where(cell_valid, charge, 0).astype(np.int8)
    return charge, cell_valid


def residue_density(phase: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    charge, cell_valid = residue_map(phase, valid_mask)
    total = int(np.count_nonzero(cell_valid))
    if total == 0:
        return 0.0
    return float(np.count_nonzero(charge[cell_valid] != 0) / total)


def wrapped_phase_difference(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (np.asarray(after) - np.asarray(before))))
