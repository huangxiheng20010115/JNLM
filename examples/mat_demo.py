from __future__ import annotations

from pathlib import Path

from jnlm.config import load_config
from jnlm.core import jnlm_filter_slc_pair
from jnlm.io import load_slc_pair_tile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tile = load_slc_pair_tile(root / "examples" / "data" / "demo_slc_pair.mat")
    cfg = load_config(root / "configs" / "jnlm_debug.yaml")
    result = jnlm_filter_slc_pair(
        tile["master_cpx"],
        tile["slave_cpx"],
        tile["valid_mask"],
        cfg,
    )
    print("input_shape:", tile["master_cpx"].shape)
    print("phase_after_shape:", result.phase_after.shape)


if __name__ == "__main__":
    main()
