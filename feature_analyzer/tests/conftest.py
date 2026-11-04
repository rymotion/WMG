"""Shared fixtures: an isolated data dir and synthetic WAVs written with soundfile."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Point the analyzer at a temp data dir before `analyzer.config` is imported anywhere.
_TMP_DATA = Path(os.environ.get("PYTEST_ANALYZER_DATA", "")) if os.environ.get("PYTEST_ANALYZER_DATA") else None


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = _TMP_DATA or tmp_path_factory.mktemp("analyzer-data")
    from analyzer import config
    config.DATA = d
    config.UPLOADS = d / "uploads"
    config.ANALYSES = d / "analyses"
    config.SOURCES_JSON = d / "sources.json"
    return d


def _noise(sr: int, seconds: float, wall_hz: float | None, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(sr * seconds))
    if wall_hz is not None:
        spec = np.fft.rfft(x)
        f = np.fft.rfftfreq(x.size, 1 / sr)
        spec[f > wall_hz] = 0
        x = np.fft.irfft(spec, x.size)
    x = x / (np.abs(x).max() + 1e-9) * 0.5
    return np.stack([x, x * 0.9], axis=1).astype(np.float32)


@pytest.fixture(scope="session")
def wav_wall(tmp_path_factory) -> Path:
    """White noise with a hard 16 kHz wall (an AI-like top end), 44.1 kHz, 75 s."""
    p = tmp_path_factory.mktemp("audio") / "noise_wall16k.wav"
    sf.write(str(p), _noise(44_100, 75, 16_000, seed=1), 44_100, subtype="PCM_16")
    return p


@pytest.fixture(scope="session")
def wav_full(tmp_path_factory) -> Path:
    """Full-band white noise, 44.1 kHz, 75 s."""
    p = tmp_path_factory.mktemp("audio") / "noise_fullband.wav"
    sf.write(str(p), _noise(44_100, 75, None, seed=2), 44_100, subtype="PCM_16")
    return p


@pytest.fixture(scope="session")
def models_ok() -> bool:
    from analyzer import analysis as A
    ok = A.models_available()
    if not ok and os.environ.get("ANALYZER_REQUIRE_MODELS"):
        pytest.fail("ANALYZER_REQUIRE_MODELS is set but training/runs has no installed model")
    return ok
