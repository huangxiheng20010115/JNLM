from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jnlm.config import load_config
from jnlm.core import jnlm_filter_slc_pair
from jnlm.io import load_mat_any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Python JNLM v1 output against a MATLAB reference MAT.")
    parser.add_argument("--matlab_reference", required=True, type=Path)
    parser.add_argument("--master_var", default="M")
    parser.add_argument("--slave_var", default="S")
    parser.add_argument("--matlab_master_after_var", default="M_dn")
    parser.add_argument("--matlab_slave_after_var", default="S_dn")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    den = np.maximum(np.mean(np.abs(b)), 1.0e-12)
    return float(np.mean(np.abs(a - b)) / den)


def main() -> int:
    args = parse_args()
    data = load_mat_any(args.matlab_reference)
    for name in [args.master_var, args.slave_var, args.matlab_master_after_var, args.matlab_slave_after_var]:
        if name not in data:
            raise KeyError(f"Missing variable {name} in {args.matlab_reference}")

    cfg = load_config(args.config)
    result = jnlm_filter_slc_pair(data[args.master_var], data[args.slave_var], config=cfg)
    report = {
        "master_after_mean_relative_error": rel_err(result.master_after, data[args.matlab_master_after_var]),
        "slave_after_mean_relative_error": rel_err(result.slave_after, data[args.matlab_slave_after_var]),
        "note": "This script reports numerical error only; use it after exporting a MATLAB reference case.",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
