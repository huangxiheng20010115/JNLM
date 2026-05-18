from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat


def _decode_mat_value(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == (1, 1):
        item = arr.item()
        return item
    return arr


def load_mat_any(path: str | Path) -> dict[str, Any]:
    """Load MAT v5 or v7.3/HDF5 variables into a simple dictionary."""

    p = Path(path)
    try:
        data = loadmat(p, squeeze_me=False, struct_as_record=False)
        return {k: _decode_mat_value(v) for k, v in data.items() if not k.startswith("__")}
    except NotImplementedError:
        pass
    except ValueError:
        pass

    try:
        import h5py
    except ImportError as exc:
        raise ImportError(f"{p} appears to require h5py for MAT v7.3/HDF5 loading") from exc

    out: dict[str, Any] = {}
    with h5py.File(p, "r") as f:
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                arr = np.array(obj)
                if arr.dtype.names and {"real", "imag"}.issubset(arr.dtype.names):
                    arr = arr["real"] + 1j * arr["imag"]
                out[key] = arr
            elif isinstance(obj, h5py.Group) and {"real", "imag"}.issubset(obj.keys()):
                out[key] = np.array(obj["real"]) + 1j * np.array(obj["imag"])
    return out


def load_slc_pair_tile(path: str | Path) -> dict[str, Any]:
    data = load_mat_any(path)
    required = ["master_cpx", "slave_cpx"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{path} missing required variables: {missing}")

    master = np.asarray(data["master_cpx"])
    slave = np.asarray(data["slave_cpx"])
    if not np.iscomplexobj(master) or not np.iscomplexobj(slave):
        raise ValueError(f"{path} master_cpx/slave_cpx must be complex")
    if master.shape != slave.shape:
        raise ValueError(f"{path} master/slave shape mismatch: {master.shape} vs {slave.shape}")

    valid = np.asarray(data.get("valid_mask", np.ones(master.shape, dtype=bool))).astype(bool)
    if valid.shape != master.shape:
        raise ValueError(f"{path} valid_mask shape {valid.shape} does not match {master.shape}")

    ifg = np.asarray(data.get("ifg_cpx", master * np.conj(slave)))
    phase = np.asarray(data.get("phase_raw", np.angle(ifg)))
    return {
        **data,
        "master_cpx": master,
        "slave_cpx": slave,
        "ifg_cpx": ifg,
        "phase_raw": phase,
        "valid_mask": valid,
        "path": str(path),
    }


def save_filter_result(path: str | Path, result: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    savemat(p, result, do_compression=True)


def list_tile_files(dataset_dir: str | Path, split: str = "samples") -> list[Path]:
    root = Path(dataset_dir) / split
    if not root.exists():
        raise FileNotFoundError(f"Tile split directory not found: {root}")
    return sorted(root.glob("*.mat"))
