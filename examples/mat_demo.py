from __future__ import annotations

from pathlib import Path

import numpy as np

from jnlm.config import load_config
from jnlm.core import jnlm_filter_insar, jnlm_filter_slc_pair
from jnlm.io import load_slc_pair_tile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tile = load_slc_pair_tile(root / "examples" / "data" / "demo_slc_pair.mat")
    cfg = load_config(root / "configs" / "jnlm_debug.yaml")

    raw_pair = jnlm_filter_slc_pair(
        tile["master_cpx"],
        tile["slave_cpx"],
        tile["valid_mask"],
        cfg,
    )
    insar = jnlm_filter_insar(
        amplitude_master=np.abs(tile["master_cpx"]),
        amplitude_slave=np.abs(tile["slave_cpx"]),
        phase=tile["phase_raw"],
        valid_mask=tile["valid_mask"],
        config=cfg,
    )

    print("input_shape:", tile["master_cpx"].shape)
    print("raw_pair_phase_after_shape:", raw_pair.phase_after.shape)
    print("insar_phase_after_shape:", insar.phase_after.shape)


if __name__ == "__main__":
    main()
