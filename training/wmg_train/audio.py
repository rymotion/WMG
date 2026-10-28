"""Audio file metadata needed by the manifest: original sample rate, duration, LUFS, windows."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    duration_s: float
    lufs: float  # NaN when loudness is undefined (silence / too short)
    window_starts_s: list[float]


def rms_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -math.inf
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * math.log10(rms) if rms > 0 else -math.inf


def window_starts(mono: np.ndarray, sr: int, windows_cfg: dict) -> list[float]:
    """Mirror of wmg_features::window_starts + RMS gate, at the file's native rate.

    The gate is evaluated before resampling, so a window right at -45 dBFS may differ from
    the crate's decision; extraction is the source of truth and re-checks it.
    """
    win = round(windows_cfg["seconds"] * sr)
    hop = round(windows_cfg["hop_seconds"] * sr)
    n = mono.shape[0]
    if n < win:
        starts = [0] if n >= int(windows_cfg["min_partial_seconds"] * sr) else []
    else:
        starts = list(range(0, n - win + 1, hop))
    gate = windows_cfg["rms_gate_dbfs"]
    return [s / sr for s in starts if rms_dbfs(mono[s : s + win]) >= gate]


def _ffmpeg_load(path: Path) -> tuple[np.ndarray, int]:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise RuntimeError(f"{path.suffix} needs ffmpeg/ffprobe on PATH (libsndfile can't decode it)")
    def run(cmd: list[str]) -> bytes:
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            msg = r.stderr.decode(errors="replace").strip().splitlines()
            raise RuntimeError(f"{cmd[0]} failed (exit {r.returncode}): {msg[0] if msg else ''}")
        return r.stdout

    meta = json.loads(run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels", "-of", "json", str(path)]))["streams"][0]
    sr, ch = int(meta["sample_rate"]), int(meta["channels"])
    raw = run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "f32le",
               "-acodec", "pcm_f32le", "-"])
    return np.frombuffer(raw, dtype="<f4").reshape(-1, ch).astype(np.float64), sr


def load(path: Path) -> tuple[np.ndarray, int]:
    """Decode to float64 [frames, channels] at the file's native rate (ffmpeg for m4a/aac)."""
    try:
        return sf.read(str(path), dtype="float64", always_2d=True)
    except sf.LibsndfileError:
        return _ffmpeg_load(path)


def probe(path: Path, windows_cfg: dict) -> AudioInfo:
    data, sr = load(path)
    mono = data.mean(axis=1)
    try:
        lufs = float(pyloudnorm.Meter(sr).integrated_loudness(data))
    except ValueError:  # shorter than one 400 ms gating block
        lufs = math.nan
    if not math.isfinite(lufs):
        lufs = math.nan
    return AudioInfo(
        sample_rate=int(sr),
        duration_s=data.shape[0] / sr,
        lufs=lufs,
        window_starts_s=window_starts(mono, sr, windows_cfg),
    )
