# JNLM Official Python v1

## Scope

This is the first clean Python implementation of the JNLM complex SLC-pair
filter under `D:\14\14_py`. The implementation follows the MATLAB reference:

`D:\14\14所\algorithms\jnlm\jnlm_pair_complex_matlab.m`

v1 includes only:

- four-channel registered SLC-pair JNLM;
- robust normalization and inverse normalization;
- shared patch weights over `[Re(M), Im(M), Re(S), Im(S)]`;
- weighted same-channel reconstruction for master and slave;
- standard masked coherence;
- wrapped phase residue density;
- tile MAT IO;
- single-tile and dataset runners;
- MATLAB comparison script skeleton.

v1 does not include registration, RANSAC, HOG, two-stage workflows, SANLE,
power-domain extensions, adaptive shrinkage, parameter scans, or model training.

## IFG-Guided Variant

An independent IFG-guided variant is implemented under:

`jnlm/variants/ifg_guided.py`

This variant exists to reproduce and evaluate an older experimental idea that
had better visual/metric behavior in some runs. It is not a replacement for
official v1.

Official v1 filters the original registered complex SLC pair directly:

`[Re(M), Im(M), Re(S), Im(S)]`

The IFG-guided variant first builds:

```text
amp_m = abs(M)
amp_s = abs(S)
ph_ifg = angle(M * conj(S))
M_guided = amp_m + 0j
S_guided = amp_s * exp(-1j * ph_ifg)
```

Then it applies the shared-weight / same-channel JNLM reconstruction to
`M_guided` and `S_guided`.

Because the interferometric phase is injected into the guided input before
filtering, this variant must not be described as raw SLC-pair JNLM. It should be
reported as `IFG-guided JNLM variant`.

Variant config:

`configs/jnlm_ifg_guided.yaml`

The default weight mode follows the older experiment:

```yaml
mean_mode: sharpened
weight_floor: 0.05
weight_power: 2.0
```

For raw exponential weights, use:

```yaml
mean_mode: legacy_mean
```

Single-tile diagnosis:

```powershell
python D:\14\14_py\scripts\run_ifg_guided_diagnosis_single_tile.py
```

Default output:

`D:\14\14_py\outputs\jnlm_official_v1\ifg_guided_diagnosis_single_tile`

## Method Definition

Given registered complex SLC images `M` and `S`, the filter splits them into four
real-valued channels:

`[Re(M), Im(M), Re(S), Im(S)]`

For each search-window offset, it computes the shared squared distance:

`d = (Mr0-Mr1)^2 + (Mi0-Mi1)^2 + (Sr0-Sr1)^2 + (Si0-Si1)^2`

The distance is smoothed over the patch by a Gaussian filter when `gauss_ps > 0`,
otherwise by a box filter. The neighbor weight is:

`w = exp(-d2 / (h^2 * 4 * patch_size^2 + eps))`

The same scalar weight is used to reconstruct each master/slave real/imag channel.

## Inputs and Outputs

Main API:

```python
from jnlm import JNLMConfig, jnlm_filter_slc_pair

result = jnlm_filter_slc_pair(master_cpx, slave_cpx, valid_mask=None, config=JNLMConfig())
```

`master_cpx` and `slave_cpx` must be 2-D complex arrays with the same shape.
`valid_mask` is optional and is carried through for metrics. The MATLAB reference
does not mask the core weight computation, so v1 does not mask filtering either.
The dataset runners handle real tile edge artifacts by marking non-finite
master/slave pixels invalid and zero-filling those pixels before calling the
core filter. This is input sanitation for invalid pixels, not a change to the
JNLM formula.

Returned fields:

- `master_after`
- `slave_after`
- `ifg_after = master_after * conj(slave_after)`
- `phase_after = angle(ifg_after)`
- `valid_mask`
- `config`

## Configuration

Default config is in `configs/jnlm_default.yaml`:

```yaml
patch_size: 7
search_window_size: 21
h: 0.15
gauss_ps: 1.0
do_norm: true
use_single: true
coh_win: 11
```

`configs/jnlm_debug.yaml` uses smaller windows for quick tests.

## paper_v1 Parameters

`configs/jnlm_paper_v1.yaml` is the frozen v1 paper/official-run config:

```yaml
patch_size: 7
search_window_size: 21
h: 0.15
gauss_ps: 1.0
do_norm: true
use_single: true
coh_win: 11
```

It is intentionally the same as the MATLAB-mapping recommended default. It is
much heavier than `jnlm_debug.yaml`: `search_window_size=21` evaluates 440
neighbor offsets, while the debug config uses `search_window_size=5` and only
24 neighbor offsets. Use `debug` for workflow checks and `paper_v1` for formal
accepted-tile runs.

Do not change `paper_v1` for speed experiments. If engineering optimizations are
tested, keep the parameter file fixed and compare output metrics and arrays.

## Running

Single tile:

```powershell
python D:\14\14_py\scripts\run_jnlm_single_tile.py `
  --tile "D:\14\14_py\datasets\平地\VV_VV\block12288_tile2048_slc_pair\samples\<tile>.mat" `
  --config "D:\14\14_py\configs\jnlm_debug.yaml" `
  --output_dir "D:\14\14_py\outputs\jnlm_official_v1\single_tile_debug"
```

Dataset smoke test:

```powershell
python D:\14\14_py\scripts\run_jnlm_dataset.py `
  --dataset_dir "D:\14\14_py\datasets\平地\VV_VV\block12288_tile2048_slc_pair" `
  --split samples `
  --max_tiles 2 `
  --config "D:\14\14_py\configs\jnlm_debug.yaml" `
  --output_dir "D:\14\14_py\outputs\jnlm_official_v1\dataset_smoke"
```

Full-size `2048 x 2048` tiles with the default `21 x 21` search window are
computationally heavy in this pure Python v1. Use debug config for workflow
validation before long runs.

## MATLAB Relationship

The algorithm structure, robust normalization, four-channel distance, weight
formula, self-weight initialization, and weighted reconstruction follow
`jnlm_pair_complex_matlab.m`.

Known implementation note: MATLAB `imgaussfilt`/`imboxfilt` with symmetric
padding is approximated with SciPy convolution/filtering using reflect-style
boundaries. This should be numerically close but should not be described as
bit-identical until `scripts/compare_jnlm_matlab_python.py` is run on exported
MATLAB reference cases.

## MATLAB Alignment Test Skeleton

After exporting a small MATLAB reference MAT containing `M`, `S`, `M_dn`, and
`S_dn`, run:

```powershell
python D:\14\14_py\scripts\compare_jnlm_matlab_python.py `
  --matlab_reference "D:\path\to\jnlm_reference_case.mat" `
  --config "D:\14\14_py\configs\jnlm_debug.yaml"
```

This reports mean relative error. It does not claim full MATLAB equivalence by
itself; that conclusion requires explicit reference cases and tolerances.
