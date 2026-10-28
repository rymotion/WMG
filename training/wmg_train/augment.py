"""Recording-chain augmentation, applied with the same random policy to both classes.

The raw data is confounded: AI files are ~128 kbps with a 16.5 kHz low-pass, human files are
256 kbps+ with content to 22 kHz, and levels differ by codec. Every training window therefore
goes through a fresh lossy encode drawn from one shared distribution, then level
normalisation, so codec and loudness stop predicting the label.

Encoding uses macOS `afconvert` (AAC, and decoding of everything) and `lameenc` (MP3).
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import lameenc
import numpy as np
import soundfile as sf

# Every source is brought to this rate before encoding, so the source sample rate (Suno exports
# 48 kHz, most other audio here is 44.1 kHz) can't become a label shortcut.
WORK_SR = 44_100
CODECS: tuple[tuple[str, int], ...] = (("aac", 96), ("aac", 128), ("aac", 160), ("mp3", 128), ("mp3", 192))
TARGET_RMS_DBFS = -14.0
GAIN_JITTER_DB = 6.0
NOISE_PROB, NOISE_SNR_DB = 0.3, (20.0, 40.0)
LOWPASS_PROB, LOWPASS_HZ = 0.3, (14_000.0, 16_000.0)
# Phone-speaker -> room -> microphone playback, as in the app's live mode. Small speakers cut the
# bass, add resonant peaks and distort on loud passages (bright 6-12 kHz harmonics) -- the same
# direction the AI class leans spectrally -- so both classes must see it equally often.
SPEAKER_PROB = 0.4


@dataclass(frozen=True)
class AugMeta:
    codec: str
    bitrate_kbps: int
    gain_db: float  # RMS level after normalisation + jitter, in dBFS
    noise_snr_db: float | None
    lowpass_hz: float | None
    speaker: dict | None = None  # speaker/room/mic parameters when the playback chain was applied


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr.decode(errors='replace').strip()[:300]}")


def decode(path: Path, tmp: Path) -> tuple[np.ndarray, int]:
    """Any file afconvert can read -> float32 [frames, channels] at its native rate."""
    wav = tmp / "decoded.wav"
    _run(["afconvert", "-f", "WAVE", "-d", "LEF32", str(path), str(wav)])
    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    wav.unlink()
    return x, sr


def encode_decode(x: np.ndarray, sr: int, codec: str, kbps: int, tmp: Path) -> np.ndarray:
    """Round-trip `x` [frames, channels] through a lossy codec; returns float32 at `sr`."""
    if codec == "aac":
        src, enc = tmp / "src.wav", tmp / "enc.m4a"
        sf.write(str(src), x, sr, subtype="FLOAT")
        _run(["afconvert", "-f", "m4af", "-d", "aac", "-b", str(kbps * 1000), str(src), str(enc)])
        src.unlink()
    elif codec == "mp3":
        enc = tmp / "enc.mp3"
        e = lameenc.Encoder()
        e.set_bit_rate(kbps)
        e.set_in_sample_rate(sr)
        e.set_channels(x.shape[1])
        e.set_quality(2)
        pcm = (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2")
        enc.write_bytes(e.encode(pcm.tobytes()) + e.flush())
    else:
        raise ValueError(codec)
    y, sr2 = decode(enc, tmp)
    enc.unlink()
    if sr2 != sr:
        raise RuntimeError(f"codec changed sample rate {sr} -> {sr2}")
    return y


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(spec.size)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    y = np.fft.irfft(spec, n)
    return y / (np.sqrt(np.mean(y**2)) + 1e-12)


def _lowpass(x: np.ndarray, sr: int, cutoff: float, width: float = 500.0) -> np.ndarray:
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1 / sr)
    gain = np.clip((cutoff + width / 2 - f) / width, 0.0, 1.0)  # linear taper over `width` Hz
    return np.fft.irfft(spec * gain, x.size)


def _eq(x: np.ndarray, sr: int, fn) -> np.ndarray:
    """Zero-phase FFT-domain EQ: `fn(freqs_hz) -> gain in dB`."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1 / sr)
    return np.fft.irfft(spec * 10 ** (fn(f) / 20), x.size)


def _highpass_db(f: np.ndarray, fc: float, order: int) -> np.ndarray:
    return -10 * np.log10(1 + (fc / np.maximum(f, 1.0)) ** (2 * order))


def _lowpass_db(f: np.ndarray, fc: float, order: int) -> np.ndarray:
    return -10 * np.log10(1 + (f / fc) ** (2 * order))


def _room_ir(sr: int, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    """Synthetic small-room impulse response: direct path + exponentially decaying noise tail."""
    rt60 = float(rng.uniform(0.15, 0.6))
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.91 * t / rt60)  # -60 dB at rt60
    tail[: int(0.002 * sr)] = 0  # first reflections arrive after ~2 ms
    drr_db = float(rng.uniform(0.0, 12.0))  # direct-to-reverberant ratio
    tail *= 10 ** (-drr_db / 20) / (np.sqrt(np.sum(tail**2)) + 1e-12)
    tail[0] = 1.0
    return tail, rt60, drr_db


def speaker_mic(x: np.ndarray, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Simulate playing mono `x` on a phone speaker and recording it with a phone mic."""
    # Ranges checked against a real phone speaker -> mic recording from the app, which kept far
    # more 10-16 kHz energy than a band-limited speaker model predicts.
    hp = float(rng.uniform(150, 500))
    peaks = [(float(rng.uniform(800, 5000)), float(rng.uniform(3, 9)), float(rng.uniform(0.15, 0.4)))
             for _ in range(rng.integers(1, 4))]  # (centre Hz, gain dB, width in octaves)
    lp = float(rng.uniform(11000, 20000))
    shelf = float(rng.uniform(-3, 6))  # high-shelf above ~5 kHz

    def speaker_db(f):
        g = _highpass_db(f, hp, 3) + _lowpass_db(f, lp, 2)
        g = g + shelf / (1 + (5000 / np.maximum(f, 1.0)) ** 2)
        lf = np.log2(np.maximum(f, 1.0))
        for fc, gain, width in peaks:
            g = g + gain * np.exp(-0.5 * ((lf - np.log2(fc)) / width) ** 2)
        return g

    y = _eq(x, sr, fn=speaker_db)
    # Driver distortion: soft clip, driven harder on loud passages.
    drive = float(rng.uniform(1.0, 6.0))
    peak = np.percentile(np.abs(y), 99.9) + 1e-12
    y = np.tanh(drive * y / peak) / np.tanh(drive) * peak
    ir, rt60, drr = _room_ir(sr, rng)
    y = np.fft.irfft(np.fft.rfft(y, y.size + ir.size) * np.fft.rfft(ir, y.size + ir.size))[: y.size]
    mic_hp = float(rng.uniform(80, 150))
    y = _eq(y, sr, fn=lambda f: _highpass_db(f, mic_hp, 2))
    return y, {"hp_hz": round(hp), "lp_hz": round(lp), "shelf_db": round(shelf, 1), "peaks": [[round(a), round(b, 1)] for a, b, _ in peaks],
               "drive": round(drive, 2), "rt60_s": round(rt60, 2), "drr_db": round(drr, 1)}


def augment(path: Path, rng: np.random.Generator) -> tuple[np.ndarray, int, AugMeta]:
    """Decode `path`, apply one random recording chain, return mono float64 + its rate."""
    with tempfile.TemporaryDirectory(prefix="wmg_aug_") as d:
        tmp = Path(d)
        x, sr = decode(path, tmp)
        if sr != WORK_SR:
            import wmg_features
            x = np.stack([wmg_features.resample(np.ascontiguousarray(x[:, c], dtype=np.float64), sr, WORK_SR)
                          for c in range(x.shape[1])], axis=1).astype(np.float32)
            sr = WORK_SR
        codec, kbps = CODECS[rng.integers(len(CODECS))]
        y = encode_decode(x, sr, codec, kbps, tmp)
    mono = y.astype(np.float64).mean(axis=1)

    spk = None
    if rng.random() < SPEAKER_PROB:
        mono, spk = speaker_mic(mono, sr, rng)

    lp = None
    if rng.random() < LOWPASS_PROB:
        lp = float(rng.uniform(*LOWPASS_HZ))
        mono = _lowpass(mono, sr, lp)

    rms = np.sqrt(np.mean(mono**2)) + 1e-12
    snr = None
    if rng.random() < NOISE_PROB:
        snr = float(rng.uniform(*NOISE_SNR_DB))
        mono = mono + _pink_noise(mono.size, rng) * rms * 10 ** (-snr / 20)
        rms = np.sqrt(np.mean(mono**2)) + 1e-12

    level = TARGET_RMS_DBFS + float(rng.uniform(-GAIN_JITTER_DB, GAIN_JITTER_DB))
    mono = mono * (10 ** (level / 20) / rms)
    return mono, sr, AugMeta(codec, int(kbps), round(level, 2), snr, lp, spk)
