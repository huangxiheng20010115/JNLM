from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jnlm.coherence import coherence_stats, local_coherence
from jnlm.config import load_config
from jnlm.core import jnlm_filter_insar, jnlm_filter_slc_pair
from jnlm.io import load_slc_pair_tile, save_filter_result
from jnlm.residue import residue_density, wrapped_phase_difference
from jnlm.visualize import save_comparison_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official JNLM v1 on one SLC-pair tile.")
    parser.add_argument("--tile", required=True, type=Path)
    parser.add_argument("--output_dir", default=Path("outputs/jnlm_single_tile"), type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--search_window_size", type=int, default=None)
    parser.add_argument("--h", type=float, default=None)
    parser.add_argument("--gauss_ps", type=float, default=None)
    parser.add_argument("--coh_win", type=int, default=None)
    parser.add_argument("--mode", choices=["insar", "slc_pair"], default="insar", help="Use the official InSAR wrapper or direct SLC-pair filtering.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(
        args.config,
        patch_size=args.patch_size,
        search_window_size=args.search_window_size,
        h=args.h,
        gauss_ps=args.gauss_ps,
        coh_win=args.coh_win,
    )
    tile = load_slc_pair_tile(args.tile)
    logging.info("Loaded tile %s shape=%s", args.tile, tile["master_cpx"].shape)
    mask = tile["valid_mask"].astype(bool)
    master = tile["master_cpx"]
    slave = tile["slave_cpx"]
    finite_pair = np.isfinite(master) & np.isfinite(slave)
    nonfinite_count = int(np.count_nonzero(~finite_pair))
    if nonfinite_count:
        logging.warning("Found %d nonfinite SLC pixels; setting them invalid and zero-filling for filtering", nonfinite_count)
        mask = mask & finite_pair
        master = np.where(finite_pair, master, 0).astype(master.dtype, copy=False)
        slave = np.where(finite_pair, slave, 0).astype(slave.dtype, copy=False)
    ifg_before = master * np.conj(slave)

    t0 = time.perf_counter()
    if args.mode == "insar":
        result = jnlm_filter_insar(np.abs(master), np.abs(slave), np.angle(ifg_before), mask, cfg)
        variant_name = "official_insar_v1"
    else:
        result = jnlm_filter_slc_pair(master, slave, mask, cfg)
        variant_name = "official_slc_pair_v1"
    master_after_for_metrics = result.master_after
    slave_after_for_metrics = result.slave_after
    runtime = time.perf_counter() - t0

    coh_before = local_coherence(master, slave, mask, cfg.coh_win)
    coh_after = local_coherence(master_after_for_metrics, slave_after_for_metrics, mask, cfg.coh_win)
    phase_before = np.angle(ifg_before)
    delta = wrapped_phase_difference(result.phase_after, phase_before)

    metrics = {
        "variant": variant_name,
        "tile": str(args.tile),
        "runtime_sec": runtime,
        "nonfinite_input_count": nonfinite_count,
        "valid_frac": float(np.mean(mask)),
        "residue_density_before": residue_density(phase_before, mask),
        "residue_density_after": residue_density(result.phase_after, mask),
        "phase_change_mean": float(np.mean(np.abs(delta[mask]))),
        "phase_change_p95": float(np.percentile(np.abs(delta[mask]), 95)),
        **{f"{k}_before_win{cfg.coh_win}": v for k, v in coherence_stats(coh_before, mask).items()},
        **{f"{k}_after_win{cfg.coh_win}": v for k, v in coherence_stats(coh_after, mask).items()},
    }
    metrics[f"coh_gain_win{cfg.coh_win}"] = (
        metrics[f"coh_mean_after_win{cfg.coh_win}"] - metrics[f"coh_mean_before_win{cfg.coh_win}"]
    )

    out_mat = args.output_dir / f"{args.tile.stem}__{variant_name}.mat"
    payload = {
        "ifg_after": result.ifg_after,
        "phase_after": result.phase_after,
        "valid_mask": mask,
        "coh_before": coh_before,
        "coh_after": coh_after,
    }
    payload.update({"master_after": result.master_after, "slave_after": result.slave_after})
    save_filter_result(out_mat, payload)
    save_comparison_png(
        args.output_dir / f"{args.tile.stem}__comparison.png",
        phase_before=phase_before,
        phase_after=result.phase_after,
        coh_before=coh_before,
        coh_after=coh_after,
        valid_mask=mask,
        title=args.tile.stem,
    )
    (args.output_dir / f"{args.tile.stem}__metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logging.info("Saved %s", out_mat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
