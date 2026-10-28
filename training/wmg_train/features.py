"""Thin wrapper over the Rust `wmg_features` module (features/ crate, built with maturin).

All detector input features must come from that crate so training, the backend and on-device
code compute identical tensors. This module never re-implements the mel pipeline in Python.
"""

from __future__ import annotations

import json
from types import ModuleType
from typing import Any

import numpy as np

_INSTALL_HINT = (
    "The `wmg_features` Python module is not installed. Build it from the Rust crate:\n"
    "    pip install maturin && maturin develop --release -m ../features/pyproject.toml\n"
    "(or `pip install` the wheel produced by `maturin build` in features/)."
)


class FeaturesUnavailable(ImportError):
    pass


def _module() -> ModuleType:
    try:
        import wmg_features  # type: ignore[import-not-found]
    except ImportError as e:
        raise FeaturesUnavailable(_INSTALL_HINT) from e
    return wmg_features


def available() -> bool:
    try:
        _module()
    except FeaturesUnavailable:
        return False
    return True


def crate_commit() -> str:
    """Git commit the installed feature crate was built from (recorded per manifest row)."""
    return str(_module().COMMIT)


def spec() -> dict[str, Any]:
    return json.loads(_module().SPEC_JSON)


def spec_hash() -> str:
    return str(_module().SPEC_HASH)


def check_spec(data_cfg: dict[str, Any]) -> None:
    """Fail if configs/data.yaml disagrees with the parameters baked into the crate."""
    s = spec()
    a = data_cfg["audio"]
    expected = {
        "sample_rate": a["sample_rate"],
        "n_mels": a["n_mels"],
        "n_fft": a["n_fft"],
        "hop": round(a["hop_ms"] * a["sample_rate"] / 1000),
        "win_length": round(a["window_ms"] * a["sample_rate"] / 1000),
        "fmin": a["fmin"],
        "fmax": a["fmax"],
        "window_seconds": data_cfg["windows"]["seconds"],
        "window_hop_seconds": data_cfg["windows"]["hop_seconds"],
    }
    bad = {k: (v, s.get(k)) for k, v in expected.items() if s.get(k) != v}
    if bad:
        raise ValueError(f"configs/data.yaml disagrees with wmg_features spec (config, crate): {bad}")


def extract_windows(samples: np.ndarray, sr: int) -> list[tuple[float, float, np.ndarray]]:
    """Mono audio at `sr` Hz -> list of (start_s, rms_dbfs, logmel[128, 1000] float32)."""
    mod = _module()
    mel = mod.DetectorMel()
    return mel.extract_windows(np.ascontiguousarray(samples, dtype=np.float64), int(sr))
