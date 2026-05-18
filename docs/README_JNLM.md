# JNLM Official Python v1

## Scope

This is the clean public Python implementation of the JNLM filtering workflow.
It follows the MATLAB references `jnlm_pair_complex_matlab.m` and
`jnlm_insar_matlab.m`.

v1 includes only:

- four-channel registered SLC-pair JNLM;
- the MATLAB-equivalent InSAR wrapper `jnlm_filter_insar(...)`;
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

## Official Public Entry Points

The release exposes two official APIs over the same JNLM core:

```python
from jnlm import jnlm_filter_insar, jnlm_filter_slc_pair
```

`jnlm_filter_insar(...)` matches MATLAB `jnlm_insar_matlab.m` by constructing:

```text
M = amplitude_master + 0j
S = amplitude_slave * exp(-1j * phase)
```

and then calling the shared-weight JNLM core. This is the recommended entry
point when reproducing the historical InSAR pipeline and the HH_HH benchmark.

`jnlm_filter_slc_pair(...)` directly filters the original registered complex SLC
pair. It is useful when direct pair filtering is the intended experiment, but it
is not the same input representation as the MATLAB InSAR wrapper.

Research-only variants such as sharpened IFG-guided weighting, two-stage
refinement, and BM3D-inspired branches are intentionally excluded from this
public release.

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

Main APIs:

```python
from jnlm import JNLMConfig, jnlm_filter_insar, jnlm_filter_slc_pair

result_insar = jnlm_filter_insar(
    amplitude_master,
    amplitude_slave,
    phase_raw,
    valid_mask=None,
    config=JNLMConfig(),
)

result_pair = jnlm_filter_slc_pair(
    master_cpx,
    slave_cpx,
    valid_mask=None,
    config=JNLMConfig(),
)
```

For `jnlm_filter_slc_pair`, `master_cpx` and `slave_cpx` must be 2-D complex arrays with the same shape.
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
python scripts/run_jnlm_single_tile.py `
  --tile "D:\path\to\registered_slc_pair.mat" `
  --config "configs\jnlm_debug.yaml" `
  --mode insar `
  --output_dir "outputs\single_tile_debug"
```

Dataset smoke test:

```powershell
python scripts/run_jnlm_dataset.py `
  --dataset_dir "D:\path\to\dataset_root" `
  --split samples `
  --max_tiles 2 `
  --config "configs\jnlm_debug.yaml" `
  --mode insar `
  --output_dir "outputs\dataset_smoke"
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
python scripts/compare_jnlm_matlab_python.py `
  --matlab_reference "D:\path\to\jnlm_reference_case.mat" `
  --config "configs\jnlm_debug.yaml"
```

This reports mean relative error. It does not claim full MATLAB equivalence by
itself; that conclusion requires explicit reference cases and tolerances.
