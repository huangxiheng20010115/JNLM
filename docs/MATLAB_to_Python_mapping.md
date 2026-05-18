# MATLAB to Python JNLM Mapping Proposal

## Goal

Build a clean official Python implementation under `D:\14\14_py` that reproduces the MATLAB JNLM complex-pair behavior first, then adds runners and diagnostics around it. The official v1 should be small, testable, and separate from registration and experiment-specific scripts.

## Proposed Python Package Layout

| Proposed module | Responsibility | Source mapping |
|---|---|---|
| `jnlm/core.py` | Public filtering APIs: `filter_pair(master, slave, config)` and `filter_from_amplitude_phase(ampl_master, ampl_slave, phase, config)`. | `jnlm_pair_complex_matlab.m`, `jnlm_insar_matlab.m`, Python `_jnlm_pair_impl`. |
| `jnlm/weights.py` | Robust normalization, padding, patch-distance smoothing, shared four-channel weight computation. | `robust_norm_matlab.m`, `inv_robust_norm_matlab.m`, distance/weight loop from `jnlm_pair_complex_matlab.m`. |
| `jnlm/reconstruct.py` | Weighted accumulation and reconstruction of `M_dn/S_dn`; conversion to IFG and phase. | Accumulator/reconstruction sections from `jnlm_pair_complex_matlab.m`; phase output from `jnlm_insar_matlab.m`. |
| `jnlm/coherence.py` | Masked standard InSAR coherence and optional phase-consistency diagnostic. | `local_coherence_masked.m`; Python `local_coh`; `phase_consistency_map.m` / Python `phase_consistency`. |
| `jnlm/residue.py` | Wrapped phase residue map and residue density over 2x2 fully valid cells. | `residue_density.m`; Python `residue_rr`. |
| `jnlm/io.py` | MAT v7.3, MAT v5, `.npy/.npz`, tile dataset loading; validation of complex dtype and shape. | `V73Reader`, `ENVIReader`, tile readers in current Python dataset scripts. |
| `jnlm/visualize.py` | Phase RGB, coherence/residue/valid-mask comparison figures. | `phase_rgb` and plotting logic from `jnlm_before_after.py`; current tile JNLM comparison scripts. |
| `scripts/run_jnlm_single_tile.py` | Run official JNLM on one SLC-pair tile MAT. | New script; should use `jnlm.io`, `jnlm.core`, `jnlm.coherence`, `jnlm.residue`, `jnlm.visualize`. |
| `scripts/run_jnlm_dataset.py` | Batch over accepted/rejected SLC-pair tile dataset, write metrics CSV/JSON and PNGs. | Current runner patterns plus `run_airborne_experiments.py` aggregation style. |

## Function-Level Mapping

| MATLAB / current Python source | Official Python target | Notes |
|---|---|---|
| `robust_norm_matlab(x)` | `jnlm.weights.robust_norm(x) -> (y, state)` | Preserve median/MAD with std fallback. |
| `inv_robust_norm_matlab(y, st)` | `jnlm.weights.inv_robust_norm(y, state)` | Preserve output dtype behavior where practical. |
| `jnlm_pair_complex_matlab(M, S, ...)` | `jnlm.core.filter_pair(master, slave, config) -> FilterResult` | Primary v1 implementation target. |
| `jnlm_insar_matlab(ampl_master, ampl_slave, phase, ...)` | `jnlm.core.filter_from_amplitude_phase(...) -> FilterResult` | Wrapper only; do not use it to fake master/slave when true SLC pair is required. |
| MATLAB distance loop over offsets | `jnlm.weights.compute_weighted_accumulators(...)` or private helper in `core.py` | Keep simple in v1; split only if it improves tests. |
| `imgaussfilt(d, gauss_ps, FilterSize=patch_size, Padding=symmetric)` | `scipy.ndimage.gaussian_filter(d, sigma=gauss_ps, mode="reflect", truncate=...)` | Match MATLAB as closely as practical; document small boundary differences. |
| `imboxfilt(d, patch_size, Padding=symmetric)` | `scipy.ndimage.uniform_filter(d, size=patch_size, mode="reflect")` | Used when `gauss_ps <= 0`. |
| `local_coherence_masked(S1, S2, valid_mask, win)` | `jnlm.coherence.local_coherence(master, slave, valid_mask, win)` | Required for metrics and output products. |
| `phase_consistency_map(phi, valid_mask, win)` | `jnlm.coherence.phase_consistency(phi, valid_mask, win)` | Diagnostic only; do not call it standard coherence. |
| `residue_density(phi, valid_mask)` | `jnlm.residue.residue_density(phi, valid_mask)` and `residue_map(...)` | Vectorize in Python while preserving fully valid 2x2 rule. |
| `V73Reader`, `ENVIReader` in `jnlm_before_after.py` | `jnlm.io` | Keep IO separate from filtering. |
| `phase_rgb` and comparison plotting | `jnlm.visualize` | Keep plotting out of `core.py`. |

## Source Priority

### Use `D:\14\14所\algorithms` as the authority for v1

- `jnlm_pair_complex_matlab.m` defines the official shared-weight complex-pair algorithm.
- `robust_norm_matlab.m` and `inv_robust_norm_matlab.m` define the normalization behavior.
- `jnlm_insar_matlab.m` defines the amplitude/phase wrapper behavior.

### Use `D:\14\星载` as supplementary reference

- Use `jnlm_before_after.py` to speed up Python translation of:
  - MAT/ENVI readers,
  - registration-free evaluation pipeline pieces,
  - `local_coh`,
  - `phase_consistency`,
  - `residue_rr`,
  - plotting and output conventions.
- Treat `jnlm_pair(..., mean_mode="mean")` as the closest Python prototype to MATLAB.
- Treat `jnlm_pair(..., mean_mode="sharpened")` as experimental.

### Do not use these as official v1 algorithm sources

- `slc_power_after_jnlm.py`: shared-affinity/SANLE research path, not minimal JNLM filtering.
- `scan_sanle_params.py`: hyperparameter scan runner.
- `run_airborne_experiments.py`: batch launcher.
- Registration utilities: important upstream but separate package/functionality.

## Official Implementation v1 Scope

Official v1 should include only:

- A typed/configured complex-pair JNLM filter:
  - input: registered `master_cpx`, `slave_cpx`, optional `valid_mask`;
  - output: `master_jnlm`, `slave_jnlm`, `ifg_jnlm`, `phase_jnlm`.
- MATLAB-compatible robust normalization.
- MATLAB-compatible four-channel shared patch distance and weight formula.
- Gaussian patch smoothing and box patch smoothing options.
- Standard masked coherence.
- Wrapped phase residue density and residue map.
- Basic tile IO for the new VV_VV SLC-pair dataset.
- A single-tile runner and a dataset runner.
- Minimal PNG comparison figures and CSV/JSON metrics.

## Explicitly Out of Scope for v1

- Registration, local matching, phase correlation, RANSAC, dense offset estimation, or resampling.
- HOG/texture matching or two-stage registration variants.
- SANLE/power-domain reconstruction, adaptive shrinkage, top-k affinity, ROI ENL/CV ranking.
- Multi-scene automatic discovery beyond the existing tile dataset contract.
- Training model integration or SwinInSAR neural network inference.
- IFG-only workflows that fabricate master/slave from `ifg_cpx`.
- Experimental `sharpened` JNLM weights as default behavior.

## Recommended v1 Defaults

- `search_window_size = 21`
- `patch_size = 7`
- `h = 0.15`
- `gauss_ps = 1.0`
- `do_norm = True`
- `use_single = True`
- `coh_win = 11`
- `weight_mode = "mean"` to match MATLAB; optionally expose `"sharpened"` later under an experimental flag.

## Suggested Development Sequence

1. Implement `jnlm.weights.robust_norm` and inverse normalization with unit tests against simple arrays.
2. Implement `jnlm.core.filter_pair` for small arrays and compare against MATLAB on a tiny saved test case.
3. Add `jnlm.coherence` and `jnlm.residue` with tests for known phase patterns and masks.
4. Add `jnlm.io.load_slc_pair_tile`.
5. Add `scripts/run_jnlm_single_tile.py`.
6. Add `scripts/run_jnlm_dataset.py`.
7. Only after v1 is stable, evaluate whether to port SANLE/top-k/adaptive shrinkage as a separate experimental module.
