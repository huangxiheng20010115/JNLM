from __future__ import annotations

import argparse
import csv
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
from jnlm.io import list_tile_files, load_slc_pair_tile, save_filter_result
from jnlm.residue import residue_density, wrapped_phase_difference
from jnlm.visualize import save_comparison_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch official JNLM v1 over SLC-pair tiles.")
    parser.add_argument("--dataset_dir", required=True, type=Path)
    parser.add_argument("--split", default="samples", choices=["samples", "rejected_samples"])
    parser.add_argument("--output_dir", default=Path("outputs/jnlm_dataset"), type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max_tiles", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--search_window_size", type=int, default=None)
    parser.add_argument("--h", type=float, default=None)
    parser.add_argument("--gauss_ps", type=float, default=None)
    parser.add_argument("--coh_win", type=int, default=None)
    parser.add_argument("--mode", choices=["insar", "slc_pair"], default="insar", help="Use the official InSAR wrapper or direct SLC-pair filtering.")
    parser.add_argument("--no_figures", action="store_true")
    parser.add_argument("--no_mat", action="store_true")
    return parser.parse_args()


def run_one(
    tile_path: Path,
    out_dir: Path,
    cfg,
    *,
    mode: str = "insar",
    save_mat: bool = True,
    save_figures: bool = True,
) -> dict[str, object]:
    tile = load_slc_pair_tile(tile_path)
    mask = tile["valid_mask"].astype(bool)
    master = tile["master_cpx"]
    slave = tile["slave_cpx"]
    finite_pair = np.isfinite(master) & np.isfinite(slave)
    nonfinite_count = int(np.count_nonzero(~finite_pair))
    nonfinite_in_valid_count = int(np.count_nonzero((~finite_pair) & mask))
    if nonfinite_count:
        mask = mask & finite_pair
        master = np.where(finite_pair, master, 0).astype(master.dtype, copy=False)
        slave = np.where(finite_pair, slave, 0).astype(slave.dtype, copy=False)
    ifg_before = master * np.conj(slave)
    phase_before = np.angle(ifg_before)
    t0 = time.perf_counter()
    if mode == "insar":
        result = jnlm_filter_insar(np.abs(master), np.abs(slave), phase_before, mask, cfg)
        variant_name = "official_insar_v1"
    else:
        result = jnlm_filter_slc_pair(master, slave, mask, cfg)
        variant_name = "official_slc_pair_v1"
    master_after_for_metrics = result.master_after
    slave_after_for_metrics = result.slave_after
    runtime = time.perf_counter() - t0

    coh_before = local_coherence(master, slave, mask, cfg.coh_win)
    coh_after = local_coherence(master_after_for_metrics, slave_after_for_metrics, mask, cfg.coh_win)
    delta = wrapped_phase_difference(result.phase_after, phase_before)
    vals = np.abs(delta[mask])

    mat_path = ""
    png_path = ""
    if save_mat:
        result_dir = out_dir / "mat"
        mat_path = str(result_dir / f"{tile_path.stem}__{variant_name}.mat")
        payload = {
            "ifg_after": result.ifg_after,
            "phase_after": result.phase_after,
            "valid_mask": mask,
            "coh_before": coh_before,
            "coh_after": coh_after,
        }
        payload.update({"master_after": result.master_after, "slave_after": result.slave_after})
        save_filter_result(mat_path, payload)
    if save_figures:
        png_path = str(out_dir / "figures" / f"{tile_path.stem}__comparison.png")
        save_comparison_png(
            png_path,
            phase_before=phase_before,
            phase_after=result.phase_after,
            coh_before=coh_before,
            coh_after=coh_after,
            valid_mask=mask,
            title=tile_path.stem,
        )

    before_stats = coherence_stats(coh_before, mask)
    after_stats = coherence_stats(coh_after, mask)
    row = {
        "variant": variant_name,
        "tile_path": str(tile_path),
        "tile_id": tile_path.stem,
        "runtime_sec": runtime,
        "mat_path": mat_path,
        "png_path": png_path,
        "nonfinite_input_count": nonfinite_count,
        "nonfinite_input_in_valid_count": nonfinite_in_valid_count,
        "valid_frac": float(np.mean(mask)),
        f"coh_mean_before_win{cfg.coh_win}": before_stats["coh_mean"],
        f"coh_mean_after_win{cfg.coh_win}": after_stats["coh_mean"],
        f"coh_gain_win{cfg.coh_win}": after_stats["coh_mean"] - before_stats["coh_mean"],
        f"coh_p25_before_win{cfg.coh_win}": before_stats["coh_p25"],
        f"coh_p25_after_win{cfg.coh_win}": after_stats["coh_p25"],
        "residue_density_before": residue_density(phase_before, mask),
        "residue_density_after": residue_density(result.phase_after, mask),
        "phase_change_mean": float(np.mean(vals)) if vals.size else float("nan"),
        "phase_change_p95": float(np.percentile(vals, 95)) if vals.size else float("nan"),
    }
    row["residue_reduction"] = row["residue_density_before"] - row["residue_density_after"]
    if result.timings:
        for key, value in result.timings.items():
            row[f"profile_{key}"] = value
    return row


def _to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _normalize_metric(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    vals = values.astype(float)
    finite = np.isfinite(vals)
    out = np.zeros_like(vals, dtype=float)
    if np.count_nonzero(finite) == 0:
        return out
    vmin = np.nanmin(vals[finite])
    vmax = np.nanmax(vals[finite])
    if abs(vmax - vmin) < 1e-12:
        out[finite] = 0.0
    else:
        out[finite] = (vals[finite] - vmin) / (vmax - vmin)
    if not higher_is_better:
        out = -out
    return out


def summarize_rows(rows: list[dict[str, object]], out_dir: Path, cfg) -> None:
    ok_rows = [r for r in rows if not r.get("error")]
    if not ok_rows:
        (out_dir / "summary.md").write_text("# JNLM Dataset Summary\n\nNo successful tiles.\n", encoding="utf-8")
        return

    coh_gain_key = f"coh_gain_win{cfg.coh_win}"
    coh_before_key = f"coh_mean_before_win{cfg.coh_win}"
    coh_after_key = f"coh_mean_after_win{cfg.coh_win}"
    coh_p25_before_key = f"coh_p25_before_win{cfg.coh_win}"
    coh_p25_after_key = f"coh_p25_after_win{cfg.coh_win}"

    def col(key: str) -> np.ndarray:
        return np.array([_to_float(r.get(key)) for r in ok_rows], dtype=float)

    score = (
        _normalize_metric(col(coh_gain_key), True)
        + _normalize_metric(col("residue_reduction"), True)
        + _normalize_metric(col("phase_change_mean"), False)
        + _normalize_metric(col("phase_change_p95"), False)
    )
    for row, value in zip(ok_rows, score):
        row["score"] = float(value)

    ranked = sorted(ok_rows, key=lambda r: _to_float(r.get("score")), reverse=True)
    suspicious = [
        r
        for r in ok_rows
        if _to_float(r.get(coh_gain_key)) < -0.02
        or _to_float(r.get("residue_reduction")) < -0.01
        or _to_float(r.get("phase_change_p95")) > 1.5
    ]
    paper_candidates = [
        r
        for r in ranked
        if _to_float(r.get(coh_gain_key)) >= 0
        and _to_float(r.get("residue_reduction")) >= 0
        and _to_float(r.get("phase_change_p95")) <= 1.5
    ][:5]

    for filename, subset in [
        ("ranked_tiles_by_score.csv", ranked),
        ("paper_candidate_tiles.csv", paper_candidates),
        ("suspicious_tiles.csv", suspicious),
    ]:
        path = out_dir / filename
        fieldnames = sorted({k for row in subset for k in row.keys()} | set(ranked[0].keys()))
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subset)

    summary = {
        "num_tiles": len(rows),
        "num_success": len(ok_rows),
        "num_failed": len(rows) - len(ok_rows),
        "config": cfg.to_dict(),
        "means": {
            "runtime_sec": float(np.nanmean(col("runtime_sec"))),
            coh_before_key: float(np.nanmean(col(coh_before_key))),
            coh_after_key: float(np.nanmean(col(coh_after_key))),
            coh_gain_key: float(np.nanmean(col(coh_gain_key))),
            coh_p25_before_key: float(np.nanmean(col(coh_p25_before_key))),
            coh_p25_after_key: float(np.nanmean(col(coh_p25_after_key))),
            "residue_density_before": float(np.nanmean(col("residue_density_before"))),
            "residue_density_after": float(np.nanmean(col("residue_density_after"))),
            "residue_reduction": float(np.nanmean(col("residue_reduction"))),
            "phase_change_mean": float(np.nanmean(col("phase_change_mean"))),
            "phase_change_p95": float(np.nanmean(col("phase_change_p95"))),
        },
        "top_5_tiles": [
            {
                "tile_id": str(r.get("tile_id")),
                "score": _to_float(r.get("score")),
                "png_path": str(r.get("png_path", "")),
            }
            for r in ranked[:5]
        ],
        "suspicious_count": len(suspicious),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# JNLM Dataset Summary",
        "",
        f"- Tiles: {summary['num_tiles']}",
        f"- Success: {summary['num_success']}",
        f"- Failed: {summary['num_failed']}",
        f"- Mean runtime_sec: {summary['means']['runtime_sec']:.3f}",
        f"- Mean coherence: {summary['means'][coh_before_key]:.6f} -> {summary['means'][coh_after_key]:.6f}",
        f"- Mean coherence gain: {summary['means'][coh_gain_key]:.6f}",
        f"- Mean residue density: {summary['means']['residue_density_before']:.6f} -> {summary['means']['residue_density_after']:.6f}",
        f"- Mean residue reduction: {summary['means']['residue_reduction']:.6f}",
        f"- Mean phase_change: {summary['means']['phase_change_mean']:.6f}",
        f"- Mean phase_change_p95: {summary['means']['phase_change_p95']:.6f}",
        f"- Suspicious tiles: {len(suspicious)}",
        "",
        "## Top 5",
    ]
    for r in ranked[:5]:
        lines.append(f"- `{r.get('tile_id')}` score={_to_float(r.get('score')):.4f} png=`{r.get('png_path')}`")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.output_dir / "run_jnlm_dataset.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    cfg = load_config(
        args.config,
        patch_size=args.patch_size,
        search_window_size=args.search_window_size,
        h=args.h,
        gauss_ps=args.gauss_ps,
        coh_win=args.coh_win,
    )
    tiles = list_tile_files(args.dataset_dir, args.split)
    if args.max_tiles and args.max_tiles > 0:
        tiles = tiles[: args.max_tiles]
    logging.info("Running %d tiles from %s split=%s variant=%s", len(tiles), args.dataset_dir, args.split, args.variant)

    rows: list[dict[str, object]] = []
    for tile_path in tiles:
        try:
            rows.append(
                run_one(
                    tile_path,
                    args.output_dir,
                    cfg,
                        save_mat=not args.no_mat,
                    save_figures=not args.no_figures,
                )
            )
            logging.info("OK %s", tile_path.name)
        except Exception as exc:
            logging.exception("FAILED %s", tile_path)
            rows.append({"tile_path": str(tile_path), "tile_id": tile_path.stem, "error": str(exc)})

    metrics_path = args.output_dir / "jnlm_dataset_metrics.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summarize_rows(rows, args.output_dir, cfg)
    logging.info("Saved metrics %s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
