"""The Rust side features (what the device appends to the embedding) match the Python
reference in wmg_train.handcrafted on the same 32 kHz window."""

import numpy as np
import pytest

from wmg_train.handcrafted import extract

wmg_features = pytest.importorskip("wmg_features")

SR = 32_000


def _signal(seed: int, framed: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 10 * SR
    x = rng.standard_normal(n) * 0.1
    if framed:
        t = np.arange(n) / SR
        x = x * (1 + 0.6 * np.sign(np.sin(2 * np.pi * 75.0 * t)))
    return x


@pytest.mark.parametrize("framed", [False, True])
def test_rust_matches_python_reference(framed):
    x = _signal(3, framed)
    ref = extract(x, SR)
    raw = wmg_features.side_features_raw(np.ascontiguousarray(x, dtype=np.float64))
    expect = [ref["flux_p2m"], ref["mod_peak_db"], ref["sfm_mean"]]
    assert list(wmg_features.SIDE_FEATURES) == ["flux_p2m", "mod_peak_db", "sfm_mean"]
    np.testing.assert_allclose(raw, expect, rtol=0.02, atol=0.05)


def test_standardised_values_and_silence():
    x = _signal(4, False)
    raw = wmg_features.side_features_raw(x)
    std = wmg_features.side_features(x)
    for i, (off, scale) in enumerate(wmg_features.SIDE_SCALING):
        assert std[i] == pytest.approx((raw[i] - off) / scale, rel=1e-5)
    assert list(wmg_features.side_features(np.zeros(10 * SR))) == [0.0, 0.0, 0.0]


def test_extract_windows_side_returns_one_vector_per_window():
    x = _signal(5, False)
    mel = wmg_features.DetectorMel()
    wins = mel.extract_windows_side(np.concatenate([x, x]), SR)
    assert len(wins) == 3  # 20 s at 10 s / 50% overlap
    for _start, _rms, logmel, side in wins:
        assert logmel.shape == (128, 1000)
        assert side.shape == (3,)
