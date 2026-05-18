from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .config import JNLMConfig
from .normalize import RobustNormState, inv_robust_norm, robust_norm
from .reconstruct import finalize_weighted_channels
from .utils import as_bool_mask, validate_slc_pair
from .weights import gaussian_kernel_1d, offset_list, shared_weight_from_distance


@dataclass
class FilterResult:
    master_after: np.ndarray
    slave_after: np.ndarray
    ifg_after: np.ndarray
    phase_after: np.ndarray
    valid_mask: np.ndarray | None = None
    config: dict[str, Any] | None = None
    timings: dict[str, float] | None = None


def _split_channels(master: np.ndarray, slave: np.ndarray, dtype: np.dtype) -> tuple[np.ndarray, ...]:
    return (
        np.asarray(master.real, dtype=dtype),
        np.asarray(master.imag, dtype=dtype),
        np.asarray(slave.real, dtype=dtype),
        np.asarray(slave.imag, dtype=dtype),
    )


def _maybe_norm(channels: tuple[np.ndarray, ...], do_norm: bool) -> tuple[tuple[np.ndarray, ...], tuple[RobustNormState | None, ...]]:
    if not do_norm:
        return channels, (None, None, None, None)
    normed = []
    states = []
    for ch in channels:
        y, st = robust_norm(ch)
        normed.append(y)
        states.append(st)
    return tuple(normed), tuple(states)


def _maybe_inv_norm(channels: tuple[np.ndarray, ...], states: tuple[RobustNormState | None, ...]) -> tuple[np.ndarray, ...]:
    out = []
    for ch, st in zip(channels, states):
        out.append(ch if st is None else inv_robust_norm(ch, st))
    return tuple(out)


def jnlm_filter_slc_pair(
    master_cpx: np.ndarray,
    slave_cpx: np.ndarray,
    valid_mask: np.ndarray | None = None,
    config: JNLMConfig | dict[str, Any] | None = None,
) -> FilterResult:
    """Filter a registered complex SLC pair with MATLAB-style shared JNLM.

    The core algorithm follows `jnlm_pair_complex_matlab.m`. `valid_mask` is
    carried through for downstream metrics/IO; the MATLAB reference does not
    mask weight computation, so v1 also leaves the filter itself unmasked.
    """

    if config is None:
        cfg = JNLMConfig()
    elif isinstance(config, dict):
        cfg = JNLMConfig(**config)
    else:
        cfg = config
    cfg.validate()
    t_total0 = time.perf_counter()
    timings: dict[str, float] = {
        "validate_sec": 0.0,
        "normalize_sec": 0.0,
        "pad_sec": 0.0,
        "distance_sec": 0.0,
        "weight_smooth_exp_sec": 0.0,
        "reconstruct_accumulate_sec": 0.0,
        "finalize_sec": 0.0,
    }

    t0 = time.perf_counter()
    master, slave = validate_slc_pair(master_cpx, slave_cpx)
    mask = as_bool_mask(valid_mask, master.shape) if valid_mask is not None else None
    dtype = np.dtype(np.float32 if cfg.use_single else np.float64)
    timings["validate_sec"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    mr, mi, sr, si = _split_channels(master, slave, dtype)
    (mr, mi, sr, si), states = _maybe_norm((mr, mi, sr, si), cfg.do_norm)
    timings["normalize_sec"] += time.perf_counter() - t0

    hs = cfg.patch_size // 2
    hsw = cfg.search_window_size // 2
    pad = hs + hsw
    pad_width = ((pad, pad), (pad, pad))

    t0 = time.perf_counter()
    mrp = np.pad(mr, pad_width, mode="symmetric")
    mip = np.pad(mi, pad_width, mode="symmetric")
    srp = np.pad(sr, pad_width, mode="symmetric")
    sip = np.pad(si, pad_width, mode="symmetric")
    timings["pad_sec"] += time.perf_counter() - t0

    hgt, wid = mr.shape
    r0 = pad
    c0 = pad

    mr0 = mrp[r0 : r0 + hgt, c0 : c0 + wid]
    mi0 = mip[r0 : r0 + hgt, c0 : c0 + wid]
    sr0 = srp[r0 : r0 + hgt, c0 : c0 + wid]
    si0 = sip[r0 : r0 + hgt, c0 : c0 + wid]

    wsum = np.ones((hgt, wid), dtype=dtype)
    mr_acc = mr0.copy()
    mi_acc = mi0.copy()
    sr_acc = sr0.copy()
    si_acc = si0.copy()
    d = np.empty((hgt, wid), dtype=dtype)
    scratch = np.empty((hgt, wid), dtype=dtype)
    gaussian_kernel = gaussian_kernel_1d(cfg.patch_size, cfg.gauss_ps, dtype) if cfg.gauss_ps > 0 else None
    patch_n = float(cfg.patch_size * cfg.patch_size)
    invden = 1.0 / ((float(cfg.h) * float(cfg.h)) * (4.0 * patch_n) + np.finfo(np.float64).eps)

    for dy, dx in offset_list(cfg.search_window_size):
        rs = r0 + dy
        cs = c0 + dx
        mrsh = mrp[rs : rs + hgt, cs : cs + wid]
        mish = mip[rs : rs + hgt, cs : cs + wid]
        srsh = srp[rs : rs + hgt, cs : cs + wid]
        sish = sip[rs : rs + hgt, cs : cs + wid]

        t0 = time.perf_counter()
        np.subtract(mr0, mrsh, out=d)
        np.square(d, out=d)
        np.subtract(mi0, mish, out=scratch)
        np.square(scratch, out=scratch)
        d += scratch
        np.subtract(sr0, srsh, out=scratch)
        np.square(scratch, out=scratch)
        d += scratch
        np.subtract(si0, sish, out=scratch)
        np.square(scratch, out=scratch)
        d += scratch
        timings["distance_sec"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        w = shared_weight_from_distance(
            d,
            patch_size=cfg.patch_size,
            h=cfg.h,
            gauss_ps=cfg.gauss_ps,
            dtype=dtype,
            gaussian_kernel=gaussian_kernel,
            invden=invden,
        )
        timings["weight_smooth_exp_sec"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        wsum += w
        np.multiply(w, mrsh, out=scratch)
        mr_acc += scratch
        np.multiply(w, mish, out=scratch)
        mi_acc += scratch
        np.multiply(w, srsh, out=scratch)
        sr_acc += scratch
        np.multiply(w, sish, out=scratch)
        si_acc += scratch
        timings["reconstruct_accumulate_sec"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    mr_out, mi_out, sr_out, si_out = finalize_weighted_channels(
        wsum, mr_acc, mi_acc, sr_acc, si_acc, eps=cfg.eps
    )
    mr_out, mi_out, sr_out, si_out = _maybe_inv_norm((mr_out, mi_out, sr_out, si_out), states)

    master_after = np.asarray(mr_out + 1j * mi_out, dtype=np.complex64 if cfg.use_single else np.complex128)
    slave_after = np.asarray(sr_out + 1j * si_out, dtype=np.complex64 if cfg.use_single else np.complex128)
    ifg_after = master_after * np.conj(slave_after)
    phase_after = np.angle(ifg_after).astype(dtype, copy=False)
    timings["finalize_sec"] += time.perf_counter() - t0
    timings["total_filter_sec"] = time.perf_counter() - t_total0

    return FilterResult(
        master_after=master_after,
        slave_after=slave_after,
        ifg_after=ifg_after,
        phase_after=phase_after,
        valid_mask=mask,
        config=cfg.to_dict(),
        timings=timings,
    )


def jnlm_filter_insar(
    amplitude_master: np.ndarray,
    amplitude_slave: np.ndarray,
    phase: np.ndarray,
    valid_mask: np.ndarray | None = None,
    config: JNLMConfig | dict[str, Any] | None = None,
) -> FilterResult:
    """Filter an InSAR amplitude/phase representation with official JNLM.

    This is the Python equivalent of the MATLAB ``jnlm_insar_matlab`` wrapper:
    it constructs a guided complex pair

    ``M = amplitude_master + 0j``
    ``S = amplitude_slave * exp(-1j * phase)``

    and then runs :func:`jnlm_filter_slc_pair` on that pair. Use this entry point
    when the available/reference workflow is defined by amplitudes plus the raw
    interferometric phase, rather than by direct filtering of the original
    registered complex SLC pair.
    """

    amp_m = np.asarray(amplitude_master)
    amp_s = np.asarray(amplitude_slave)
    ph = np.asarray(phase)
    if amp_m.ndim != 2 or amp_s.ndim != 2 or ph.ndim != 2:
        raise ValueError("amplitude_master, amplitude_slave, and phase must be 2-D arrays")
    if amp_m.shape != amp_s.shape or amp_m.shape != ph.shape:
        raise ValueError(
            "amplitude_master, amplitude_slave, and phase must have identical shapes: "
            f"{amp_m.shape}, {amp_s.shape}, {ph.shape}"
        )
    if np.iscomplexobj(amp_m) or np.iscomplexobj(amp_s):
        raise ValueError("amplitude_master and amplitude_slave must be real-valued arrays")

    if isinstance(config, JNLMConfig):
        use_single = config.use_single
    elif isinstance(config, dict):
        use_single = bool(config.get("use_single", JNLMConfig().use_single))
    else:
        use_single = JNLMConfig().use_single
    real_dtype = np.float32 if use_single else np.float64
    complex_dtype = np.complex64 if use_single else np.complex128

    guided_master = np.asarray(amp_m, dtype=real_dtype).astype(complex_dtype)
    guided_slave = np.asarray(amp_s, dtype=real_dtype) * np.exp(-1j * np.asarray(ph, dtype=real_dtype))
    guided_slave = np.asarray(guided_slave, dtype=complex_dtype)

    return jnlm_filter_slc_pair(
        guided_master,
        guided_slave,
        valid_mask=valid_mask,
        config=config,
    )
