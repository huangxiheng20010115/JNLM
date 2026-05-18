from __future__ import annotations

import numpy as np

from jnlm import JNLMConfig, jnlm_filter_insar, jnlm_filter_slc_pair


def main() -> None:
    rng = np.random.default_rng(0)
    h, w = 64, 64
    yy = np.linspace(-0.25, 0.25, h)[:, None]
    xx = np.linspace(-0.5, 0.5, w)[None, :]
    clean_phase = yy + xx
    master = np.exp(1j * (clean_phase + 0.15 * rng.standard_normal((h, w))))
    slave = np.exp(1j * (0.15 * rng.standard_normal((h, w))))
    cfg = JNLMConfig(patch_size=3, search_window_size=5)

    raw_pair = jnlm_filter_slc_pair(
        master.astype(np.complex64),
        slave.astype(np.complex64),
        config=cfg,
    )
    insar = jnlm_filter_insar(
        amplitude_master=np.abs(master),
        amplitude_slave=np.abs(slave),
        phase=np.angle(master * np.conj(slave)),
        config=cfg,
    )

    print("raw_pair_phase_after:", raw_pair.phase_after.shape)
    print("insar_phase_after:", insar.phase_after.shape)


if __name__ == "__main__":
    main()
