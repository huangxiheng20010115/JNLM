# JNLM Official Python Implementation

This repository provides the official Python implementation of **Joint Nonlocal Means (JNLM)** for registered complex SLC-pair filtering.

The implementation follows the MATLAB reference formulation used for JNLM:

- four real channels: `[Re(M), Im(M), Re(S), Im(S)]`
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
- IFG-guided experimental variants
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

```python
import numpy as np
from jnlm import JNLMConfig, jnlm_filter_slc_pair

rng = np.random.default_rng(0)
phase = np.linspace(-0.5, 0.5, 64)[None, :] + np.linspace(-0.25, 0.25, 64)[:, None]
master = np.exp(1j * (phase + 0.15 * rng.standard_normal((64, 64))))
slave = np.exp(1j * (0.15 * rng.standard_normal((64, 64))))

result = jnlm_filter_slc_pair(
    master.astype(np.complex64),
    slave.astype(np.complex64),
    config=JNLMConfig(),
)

print(result.phase_after.shape)
```

You can also run the bundled synthetic demo:

```bash
python examples/synthetic_demo.py
```


### Real MAT demo

A small real-format SLC-pair example is included at `examples/data/demo_slc_pair.mat`.
It contains the same core variables expected by the command-line tools:

- `master_cpx`
- `slave_cpx`
- `ifg_cpx`
- `phase_raw`
- `valid_mask`
- `coh_map_win11`

Run it with:

```bash
python examples/mat_demo.py
```

Or through the single-tile runner:

```bash
python scripts/run_jnlm_single_tile.py   --tile examples/data/demo_slc_pair.mat   --config configs/jnlm_debug.yaml   --output_dir outputs/example_mat_demo
```

## Command-line usage

Single tile:

```bash
python scripts/run_jnlm_single_tile.py   --tile /path/to/slc_pair_tile.mat   --config configs/jnlm_debug.yaml   --output_dir outputs/single_tile_debug
```

Dataset run:

```bash
python scripts/run_jnlm_dataset.py   --dataset_dir /path/to/dataset_root   --split samples   --config configs/jnlm_paper_v1.yaml   --output_dir outputs/dataset_run
```

## Configurations

- `configs/jnlm_default.yaml` — balanced default
- `configs/jnlm_debug.yaml` — small windows for fast workflow checks
- `configs/jnlm_paper_v1.yaml` — frozen official-run parameters

## MATLAB relationship

See:

- `docs/MATLAB_to_Python_mapping.md`
- `scripts/compare_jnlm_matlab_python.py`

The implementation is designed to be numerically close to the MATLAB reference while using SciPy filtering operators. Border handling is documented in the repository notes and should not be described as bit-identical without an explicit exported MATLAB reference comparison.

## Repository layout

```text
jnlm/       core package
configs/    frozen YAML configs
scripts/    runnable utilities
examples/   self-contained demo
 tests/     unit tests
 docs/      method and mapping notes
```

## Citation

If you use this implementation in research, cite the associated JNLM method paper and this repository. A formal citation file can be added once you decide the exact author list and bibliographic metadata for the public release.
