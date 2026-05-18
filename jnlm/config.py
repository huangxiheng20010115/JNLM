from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JNLMConfig:
    """Configuration for MATLAB-style complex SLC-pair JNLM."""

    patch_size: int = 7
    search_window_size: int = 21
    h: float = 0.15
    gauss_ps: float = 1.0
    do_norm: bool = True
    use_single: bool = True
    coh_win: int = 11
    eps: float = 1.0e-12

    def validate(self) -> None:
        if self.patch_size < 3 or self.patch_size % 2 != 1:
            raise ValueError("patch_size must be odd and >= 3")
        if self.search_window_size < 5 or self.search_window_size % 2 != 1:
            raise ValueError("search_window_size must be odd and >= 5")
        if self.h <= 0:
            raise ValueError("h must be positive")
        if self.coh_win < 1 or self.coh_win % 2 != 1:
            raise ValueError("coh_win must be odd and >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None, **overrides: Any) -> JNLMConfig:
    """Load a YAML config if provided and apply keyword overrides.

    PyYAML is optional for callers that use the dataclass directly, but it is
    available in the current project environment.
    """

    values: dict[str, Any] = {}
    if path is not None:
        import yaml

        with Path(path).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must contain a mapping: {path}")
        values.update(loaded)

    values.update({k: v for k, v in overrides.items() if v is not None})
    cfg = JNLMConfig(**values)
    cfg.validate()
    return cfg
