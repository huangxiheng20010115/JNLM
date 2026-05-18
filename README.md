# JNLM Official Python Implementation

This repository provides the official Python implementation of **Joint Nonlocal Means (JNLM)** for two closely related workflows:

- direct filtering of a registered complex SLC pair;
- the InSAR-guided amplitude/phase formulation used by the MATLAB `jnlm_insar_matlab` reference path.

The implementation follows the MATLAB reference formulation used for JNLM:

- four real channels: `[Re(M), Im(M), Re(S), Im(S)]`
- an official InSAR-guided wrapper that builds `M = |M| + 0j`, `S = |S| * exp(-1j * phase)` before calling the same core filter
- shared nonlocal patch weights across the registered master/slave pair
- same-channel weighted reconstruction for master and slave
- standard masked coherence metrics
- wrapped-phase residue metrics

## What is included

- clean Python package under `jnlm/`
- frozen official parameter sets in `configs/`
- single-tile and dataset runners in `scripts/`
- MATLAB-alignment helper script
- unit tests for the core algorithm and metrics
- a self-contained synthetic example under `examples/`

## What is intentionally not included

This official repository does **not** include:

- image registration or preprocessing
- private datasets or generated experiment outputs
- research-only IFG-guided variants beyond the official MATLAB-equivalent InSAR wrapper
- two-stage refinements
- BM3D-inspired research branches
- parameter sweeps or paper-specific local workflows

Those are intentionally kept separate from the official JNLM definition.

## Installation

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Quick start

For reproducible InSAR filtering, the recommended public entry point is the
MATLAB-equivalent amplitude/phase wrapper:

```python
import numpy as np
from jnlm import JNLMConfig, jnlm_filter_insar

rng = np.random.default_rng(0)
phase = np.linspace(-0.5, 0.5, 64)[None, :] + np.linspace(-0.25, 0.25, 64)[:, None]
master = np.exp(1j * (phase + 0.15 * rng.standard_normal((64, 64))))
slave = np.exp(1j * (0.15 * rng.standard_normal((64, 64))))

result = jnlm_filter_insar(
    amplitude_master=np.abs(master),
    amplitude_slave=np.abs(slave),
    phase=np.angle(master * np.conj(slave)),
    config=JNLMConfig(),
)

print(result.phase_after.shape)
```

If you specifically want direct filtering of the original registered complex
SLC pair, use `jnlm_filter_slc_pair(...)` instead. Both entry points share the
same JNLM core; the difference is the input representation.

You can also run the bundled synthetic demo:

```bash
python examples/synthetic_demo.py
```


### Real MAT demo

A real-format post-registration / pre-interferogram example is included at
`examples/data/demo_slc_pair.mat`. It follows the recommended public input
contract:

- `master_cpx`
- `slave_cpx`
- `ifg_cpx`
- `phase_raw`
- `valid_mask`

`coh_map_win11` may be present in convenience demo files, but it is not required
by the filter API.

Run it with:

```bash
python examples/mat_demo.py
```

Or through the single-tile runner:

```bash
python scripts/run_jnlm_single_tile.py \
  --tile examples/data/demo_slc_pair.mat \
  --config configs/jnlm_debug.yaml \
  --mode insar \
  --output_dir outputs/example_mat_demo
```

## Command-line usage

Single tile:

```bash
python scripts/run_jnlm_single_tile.py \
  --tile /path/to/registered_slc_pair.mat \
  --config configs/jnlm_debug.yaml \
  --mode insar \
  --output_dir outputs/single_tile_debug
```

Dataset run:

```bash
python scripts/run_jnlm_dataset.py \
  --dataset_dir /path/to/dataset_root \
  --split samples \
  --config configs/jnlm_paper_v1.yaml \
  --mode insar \
  --output_dir outputs/dataset_run
```

Use `--mode slc_pair` only when you intentionally want the direct complex-pair
formulation rather than the MATLAB-equivalent InSAR wrapper.

## Configurations

- `configs/jnlm_default.yaml` — balanced default
- `configs/jnlm_debug.yaml` — small windows for fast workflow checks
- `configs/jnlm_paper_v1.yaml` — frozen official-run parameters

## Reproducibility workflow

For results intended to match the historical MATLAB InSAR pipeline, export the
**post-registration, pre-interferogram** MAT artifact and run the `insar` mode on
that file. The recommended variables are:

- `master_cpx`
- `slave_cpx`
- `valid_mask`
- `ifg_cpx`
- `phase_raw`

This keeps registration separate from filtering and matches the MATLAB
`jnlm_insar_matlab` reference path used by the HH_HH benchmark. In our internal
HH_HH reproduction, the official Python InSAR wrapper matched the MATLAB summary
at the reported precision: residue density `0.1310202 -> 0.00023609` and mean
coherence `0.64638 -> 0.98496`.

## MATLAB relationship

See:

- `docs/MATLAB_to_Python_mapping.md`
- `scripts/compare_jnlm_matlab_python.py`

The implementation is designed to be numerically close to the MATLAB reference while using SciPy filtering operators. Border handling is documented in the repository notes and should not be described as bit-identical without an explicit exported MATLAB reference comparison.

## Repository layout

```text
jnlm/      core package
configs/   frozen YAML configs
scripts/   runnable utilities
examples/  self-contained demos
tests/     unit tests
docs/      method and mapping notes
```

## Citation

If you use this implementation in research, cite the associated JNLM method paper and this repository. A formal citation file can be added once you decide the exact author list and bibliographic metadata for the public release.
