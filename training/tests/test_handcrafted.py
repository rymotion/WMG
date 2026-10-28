"""Each hand-crafted feature moves the right way on a synthetic signal built to trigger it."""

import numpy as np
from scipy import signal

from wmg_train.handcrafted import FEATURES, extract

SR = 44_100
RNG = np.random.default_rng(0)


def _pink(n: int) -> np.ndarray:
    white = RNG.standard_normal(n)
    spec = np.fft.rfft(white) / np.sqrt(np.maximum(np.fft.rfftfreq(n, 1 / SR), 1.0))
    x = np.fft.irfft(spec, n)
    return x / np.abs(x).max() * 0.5


def _lowpass(x: np.ndarray, hz: float) -> np.ndarray:
    sos = signal.butter(12, hz, fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x)


def test_all_features_present_and_finite_at_44k():
    f = extract(_pink(SR * 8), SR)
    assert set(FEATURES) <= set(f)
    assert all(np.isfinite(f[k]) for k in FEATURES), f


def test_bandwidth_features_see_a_16k_wall():
    full = extract(_pink(SR * 8), SR)
    wall = _lowpass(_pink(SR * 8), 16_000)
    walled = extract(wall, SR)
    assert walled["fc_hz"] < 17_000 < full["fc_hz"]
    assert walled["slope_break"] < full["slope_break"]
    # Faint noise above the wall, as an upsampler leaves behind, raises the residue.
    air = _pink(SR * 8) - _lowpass(_pink(SR * 8), 17_000)
    assert extract(wall + air * 10 ** (-40 / 20), SR)["residue_db"] > walled["residue_db"] + 20


def test_notch_is_undefined_at_32k_and_measured_at_44k():
    assert np.isnan(extract(_pink(32_000 * 8), 32_000)["notch_db"])
    assert np.isfinite(extract(_pink(SR * 8), SR)["notch_db"])


def test_smear_features_separate_hats_from_texture():
    n = SR * 8
    t = np.arange(n) / SR
    texture = _pink(n)  # continuous noise-like top end: flat all the time
    # Between hits a real mix's top end is tonal (harmonics, ring-outs), so flatness drops.
    tonal = sum(np.sin(2 * np.pi * f * t) for f in (9_000, 10_500, 12_000, 13_500, 15_000)) * 0.002
    hats = np.zeros(n)
    for start in np.arange(0.0, 8.0, 0.25):  # a hit every 250 ms with a 20 ms decay
        i = int(start * SR)
        k = int(0.02 * SR)
        hats[i : i + k] += RNG.standard_normal(k) * np.exp(-np.arange(k) / (0.004 * SR))
    hats = hats / np.abs(hats).max() * 0.5 + tonal
    a, b = extract(hats, SR), extract(texture, SR)
    assert a["sfm_var"] > b["sfm_var"] * 3
    assert a["flux_p2m"] > b["flux_p2m"] * 2
    assert a["crest_db"] > b["crest_db"] + 3


def test_codec_frame_rate_shows_as_a_modulation_peak():
    n = SR * 8
    t = np.arange(n) / SR
    smooth = _pink(n)
    framed = smooth * (1 + 0.6 * signal.square(2 * np.pi * 75.0 * t))  # re-synthesised every 1/75 s
    a, b = extract(framed, SR), extract(smooth, SR)
    assert a["mod_codec_db"] > b["mod_codec_db"] + 6
    assert abs(a["mod_peak_hz"] - 75.0) < 1.5


def test_ripple_sees_a_comb_not_a_cutoff():
    x = _pink(SR * 8)
    # A periodic ripple every 689 Hz (MP3's 32 subbands at 44.1 kHz) along the spectrum.
    spec = np.fft.rfft(x)
    fr = np.fft.rfftfreq(x.size, 1 / SR)
    combed = np.fft.irfft(spec * (1 + 0.5 * np.cos(2 * np.pi * fr / 689.0)), x.size)
    assert extract(combed, SR)["ripple_db"] > extract(x, SR)["ripple_db"] + 6
    # A brick wall alone must not look like a ripple.
    assert abs(extract(_lowpass(x, 16_000), SR)["ripple_db"] - extract(x, SR)["ripple_db"]) < 4
