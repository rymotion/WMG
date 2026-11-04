"""One audio file -> an `Analysis` (JSON-serialisable dataclass + PNGs on disk).

Everything numeric comes from `wmg_train` / `wmg_features`:
  decode            wmg_train.augment.decode (afconvert)
  hand-crafted      wmg_train.handcrafted.extract / file_features (11 top-end features)
  detector scores   wmg_train.score.load_model / score_file / plot (BandNet, both installed runs)
  side features     wmg_features.side_features_raw on each 10 s window at 32 kHz
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from . import config

FEATURE_SECONDS = 60.0
CONDITIONS = ("raw", "aac128")
MODEL_NAMES = ("10s", "5s")
SIDE_SR = 32_000
WINDOW_S, HOP_S = 10.0, 5.0

FEATURE_HELP = {
    "fc_hz": "effective bandwidth: highest frequency within 30 dB of the 2-5 kHz peak",
    "notch_db": "dip at 15-16 kHz vs 13-14 and 17-18 kHz (NaN below 36 kHz)",
    "slope_break": "log-log slope 16-20 kHz minus slope 10-14 kHz, dB/octave",
    "residue_db": "energy above fc+1 kHz over energy below fc, dB",
    "sfm_mean": "spectral flatness 8-16 kHz, mean over frames",
    "sfm_var": "spectral flatness 8-16 kHz, variance over frames",
    "flux_p2m": "spectral flux 8-16 kHz, 99th percentile over median",
    "crest_db": "8-16 kHz band-passed crest factor, median over 1 s",
    "mod_peak_db": "sharpest narrow peak of the 8-16 kHz envelope modulation spectrum, 30-120 Hz",
    "mod_codec_db": "the same at neural-codec frame rates (50, 75, 86 Hz)",
    "ripple_db": "periodic ripple along frequency of the long-term spectrum",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Analysis:
    id: str                                  # sha1 of the file bytes
    name: str
    source_path: str
    created_at: str
    category: str | None = None              # human | ai_assisted | ai | None
    generator: str = ""
    artist: str = ""
    notes: str = ""
    profile: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)        # {cond: {feature: value|None}}
    scores: dict = field(default_factory=dict)          # {model: {...}}
    side_features: list = field(default_factory=list)   # [{start_s, flux_p2m, mod_peak_db, sfm_mean}]
    images: dict = field(default_factory=dict)          # {key: file name under the analysis dir}
    reference_folder: str | None = None
    models_skipped: bool = False
    errors: list = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return config.ANALYSES / self.id

    def to_dict(self) -> dict:
        return _clean(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "Analysis":
        known = {k: d.get(k) for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    def save(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / "analysis.json"
        p.write_text(json.dumps(self.to_dict(), indent=1))
        return p

    # Summary numbers the list / compare pages show.
    def p_ai(self, model: str) -> float | None:
        return (self.scores.get(model) or {}).get("p_ai")

    def headline(self) -> float | None:
        ps = [p for p in (self.p_ai(m) for m in MODEL_NAMES) if p is not None]
        return round(sum(ps) / len(ps), 4) if ps else None

    def verdict(self) -> dict:
        return verdict([w["p_ai"] for m in MODEL_NAMES for w in (self.scores.get(m) or {}).get("windows") or []
                        if w.get("p_ai") is not None])


# Window colours on the pages: P(AI) below P_HUMAN is green, P_AI and above red, between them
# yellow. The verdict reads the mix of colours over every window of both models.
P_HUMAN, P_AI = 0.4, 0.7
VERDICTS = {
    "ai": {"label": "AI-generated", "confidence": "high", "estimate": "90%+", "css": "p-ai"},
    "likely_ai": {"label": "likely AI-generated", "confidence": "medium", "estimate": "75–80%", "css": "p-mid"},
    "human": {"label": "likely human", "confidence": "medium", "estimate": "20% or less", "css": "p-human"},
    "undetermined": {"label": "undetermined", "confidence": "low", "estimate": "–", "css": "p-none"},
    "none": {"label": "not scored", "confidence": "", "estimate": "–", "css": "p-none"},
}


def verdict(window_p: list[float]) -> dict:
    """Call from the colours of the windows: mostly red -> AI; a red+yellow majority (typically
    mostly yellow) -> likely AI at roughly 75-80%; mostly green -> human; anything else is left
    undetermined rather than guessed."""
    n = len(window_p)
    red = sum(p >= P_AI for p in window_p)
    green = sum(p < P_HUMAN for p in window_p)
    yellow = n - red - green
    if n == 0:
        key = "none"
    elif red / n > 0.5:
        key = "ai"
    elif (red + yellow) / n > 0.5:
        key = "likely_ai"
    elif green / n >= 0.6:
        key = "human"
    else:
        key = "undetermined"
    return {"key": key, **VERDICTS[key], "n": n, "red": red, "yellow": yellow, "green": green,
            "reason": f"{red} red, {yellow} yellow, {green} green of {n} windows (both models)" if n else "no windows"}


def _clean(v: Any) -> Any:
    """NaN/inf -> None, numpy scalars -> Python, so the JSON is valid and stable."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    return v


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(analysis_id: str) -> Analysis | None:
    p = config.ANALYSES / analysis_id / "analysis.json"
    if not p.is_file():
        return None
    return Analysis.from_dict(json.loads(p.read_text()))


def list_analyses() -> list[Analysis]:
    out = []
    if config.ANALYSES.is_dir():
        for d in config.ANALYSES.iterdir():
            a = load(d.name)
            if a:
                out.append(a)
    return sorted(out, key=lambda a: a.created_at, reverse=True)


def delete(analysis_id: str) -> None:
    d = config.ANALYSES / analysis_id
    if d.is_dir() and d.parent == config.ANALYSES:
        shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Reference distributions (reports/handcrafted/<newest>/training_{raw,aac128}.csv)
# ---------------------------------------------------------------------------

def reference_folder() -> Path | None:
    if not config.HANDCRAFTED_REPORTS.is_dir():
        return None
    cands = [d for d in config.HANDCRAFTED_REPORTS.iterdir() if (d / "training_raw.csv").is_file()]
    return max(cands, key=lambda d: d.name) if cands else None


@lru_cache(maxsize=4)
def _reference(folder: str | None) -> dict:
    """{"folder", "counts": {cond: {human, ai}}, "bands": {cond: {feature: {human: [p10,p50,p90], ai: [...]}}}}"""
    from wmg_train.handcrafted import FEATURES
    import pandas as pd

    out: dict = {"folder": folder, "counts": {}, "bands": {}}
    if folder is None:
        return out
    for cond in CONDITIONS:
        csv = Path(folder) / f"training_{cond}.csv"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv)
        df = df[df["error"].fillna("").astype(str) == ""]
        out["counts"][cond] = {"human": int((df.label == 0).sum()), "ai": int((df.label == 1).sum())}
        bands = {}
        for k in FEATURES:
            if k not in df:
                continue
            bands[k] = {}
            for name, lab in (("human", 0), ("ai", 1)):
                v = df.loc[df.label == lab, k].astype(float).dropna().to_numpy()
                bands[k][name] = [float(x) for x in np.percentile(v, [10, 50, 90])] if v.size else None
        out["bands"][cond] = bands
    return out


def reference() -> dict:
    f = reference_folder()
    return _reference(str(f) if f else None)


def where(value: float | None, band_h: list | None, band_a: list | None) -> str:
    """Which p10-p90 band(s) the value sits in: 'human', 'ai', 'both', 'neither' or ''."""
    if value is None or not (isinstance(value, (int, float)) and math.isfinite(value)):
        return ""
    in_h = bool(band_h) and band_h[0] <= value <= band_h[2]
    in_a = bool(band_a) and band_a[0] <= value <= band_a[2]
    return {(True, True): "both", (True, False): "human", (False, True): "ai", (False, False): "neither"}[(in_h, in_a)]


def feature_rows(a: Analysis, ref: dict | None = None) -> list[dict]:
    """Table rows for the detail page / summary.md: value per condition next to the bands."""
    from wmg_train.handcrafted import FEATURES
    ref = ref if ref is not None else reference()
    rows = []
    for k in FEATURES:
        row = {"name": k, "help": FEATURE_HELP.get(k, ""), "conds": {}}
        for cond in CONDITIONS:
            b = (ref.get("bands", {}).get(cond, {}) or {}).get(k, {}) or {}
            v = (a.features.get(cond) or {}).get(k)
            row["conds"][cond] = {"value": v, "human": b.get("human"), "ai": b.get("ai"),
                                  "where": where(v, b.get("human"), b.get("ai"))}
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

_AFINFO_FMT = re.compile(r"Data format:\s+(\d+) ch,\s+(\d+) Hz,\s+(\S+)")
_AFINFO_BR = re.compile(r"bit rate:\s+(\d+) bits per second")
_AFINFO_DUR = re.compile(r"estimated duration:\s+([\d.]+) sec")
_AFINFO_DEPTH = re.compile(r"source bit depth:\s+(\S+)")
CODEC_NAMES = {"aac": "aac", ".mp3": "mp3", "lpcm": "pcm", "alac": "alac", "flac": "flac", "ulaw": "ulaw"}


def afinfo(path: Path) -> dict:
    """Container/codec facts from macOS `afinfo` (no ffprobe needed). Empty dict if it fails."""
    try:
        r = subprocess.run(["afinfo", str(path)], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if r.returncode != 0:
        return {}
    out: dict = {}
    if m := _AFINFO_FMT.search(r.stdout):
        out["channels"], out["sample_rate"] = int(m.group(1)), int(m.group(2))
        raw = m.group(3).strip()
        out["codec"] = CODEC_NAMES.get(raw.lower(), raw.strip("'. "))
    if m := _AFINFO_BR.search(r.stdout):
        out["bitrate_kbps"] = round(int(m.group(1)) / 1000)
    if m := _AFINFO_DUR.search(r.stdout):
        out["duration_s"] = float(m.group(1))
    if m := _AFINFO_DEPTH.search(r.stdout):
        out["bit_depth"] = m.group(1)
    if m := re.search(r"File type ID:\s+(\S+)", r.stdout):
        out["file_type"] = m.group(1)
    return out


def profile_file(path: Path, x: np.ndarray | None = None, sr: int | None = None) -> dict:
    """duration, rate, channels, codec/bitrate (afinfo), RMS and peak dBFS of the decoded audio."""
    path = Path(path)
    prof: dict = {"file": path.name, "size_bytes": path.stat().st_size, "extension": path.suffix.lower()}
    prof.update(afinfo(path))
    if x is None:
        from wmg_train.augment import decode
        with tempfile.TemporaryDirectory(prefix="wmg_an_") as d:
            x, sr = decode(path, Path(d))
    mono = x.astype(np.float64).mean(axis=1)
    prof["sample_rate"] = int(sr)
    prof["channels"] = int(x.shape[1])
    prof["duration_s"] = round(x.shape[0] / sr, 3)
    prof["rms_dbfs"] = round(float(20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-12)), 2)
    prof["peak_dbfs"] = round(float(20 * np.log10(np.abs(x).max() + 1e-12)), 2)
    if prof.get("codec") in (None, "pcm") and "bitrate_kbps" not in prof and prof.get("duration_s"):
        prof["bitrate_kbps"] = round(prof["size_bytes"] * 8 / prof["duration_s"] / 1000)
    return prof


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

_MODELS: dict[str, Any] = {}


def model_path(name: str) -> Path | None:
    """`training/runs/<installed run>/model.pt` for '10s' or '5s', if the run exists."""
    try:
        from wmg_train import retrain
        run = retrain.installed_run(name)
    except Exception:  # noqa: BLE001
        run = None
    if run:
        p = config.RUNS / run / "model.pt"
        if p.is_file():
            return p
    return None


def models_available() -> bool:
    return all(model_path(n) is not None for n in MODEL_NAMES)


def get_model(name: str):
    if name not in _MODELS:
        from wmg_train.score import load_model
        p = model_path(name)
        _MODELS[name] = load_model(p) if p else None
    return _MODELS[name]


def _sigmoid(v: float) -> float:
    return float(1 / (1 + math.exp(-v)))


def _score_5s(model, res10: dict, run: str) -> dict:
    """The 5 s model on both halves of every 10 s window (as the app and session_report do).
    Windows overlap 50 %, so the second half of window i is the first half of window i+1: kept once."""
    import torch
    from wmg_train.models.student import BAND_NAMES
    mel = res10["_mel"]
    half = mel.shape[2] // 2
    starts = [w["start_s"] for w in res10["windows"]]
    seen: dict[float, dict] = {}
    with torch.no_grad():
        for part, sl in ((0, slice(0, half)), (1, slice(half, 2 * half))):
            out = model(torch.from_numpy(np.ascontiguousarray(mel[:, :, sl]))[:, None])
            logits = out["ai_logit"].numpy()
            bands = out["band_scores"].numpy()
            for s, lg, bs in zip(starts, logits, bands):
                t = round(s + part * WINDOW_S / 2, 3)
                if t not in seen:
                    seen[t] = {"start_s": t, "logit": float(lg), "p_ai": round(_sigmoid(float(lg)), 4),
                               "band_scores": {b: round(float(v), 4) for b, v in zip(BAND_NAMES, bs)}}
    wins = [seen[t] for t in sorted(seen)]
    mean_logit = float(np.mean([w["logit"] for w in wins]))
    band_mean = {b: round(float(np.mean([w["band_scores"][b] for w in wins])), 4) for b in BAND_NAMES}
    return {"run": run, "window_s": WINDOW_S / 2, "p_ai": round(_sigmoid(mean_logit), 4),
            "band_scores": band_mean, "windows": [{k: v for k, v in w.items() if k != "logit"} for w in wins]}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_spectrum(mono: np.ndarray, sr: int, fc_hz: float | None, png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal
    n = min(4096, mono.size)
    f, P = signal.welch(mono, fs=sr, nperseg=n)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(f / 1000, 10 * np.log10(np.maximum(P, 1e-20)), lw=0.8)
    if fc_hz:
        ax.axvline(fc_hz / 1000, color="C3", ls="--", lw=0.8, label=f"fc = {fc_hz / 1000:.1f} kHz")
        ax.legend(loc="upper right")
    for khz in (16, 20):
        if khz * 1000 < sr / 2:
            ax.axvline(khz, color="gray", lw=0.4, ls=":")
    ax.set_xlim(0, sr / 2000)
    ax.set_xlabel("kHz")
    ax.set_ylabel("dB")
    ax.set_title(f"{title}: long-term spectrum (middle {FEATURE_SECONDS:.0f} s, {sr} Hz)")
    ax.grid(alpha=0.3)
    fig.savefig(png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_timeline(scores: dict, side: list[dict], png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["flux_p2m", "mod_peak_db", "sfm_mean"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1, 1]})
    ax = axes[0]
    for name, style in (("10s", "o-"), ("5s", "s--")):
        s = scores.get(name) or {}
        if s.get("windows"):
            w = s["window_s"] if "window_s" in s else WINDOW_S
            ax.plot([x["start_s"] + w / 2 for x in s["windows"]], [x["p_ai"] for x in s["windows"]], style, ms=3,
                    label=f"{name} model ({s.get('run')}): P(AI) = {s['p_ai']:.2f}")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("window P(AI)")
    ax.set_title(f"{title}: detector windows and side features")
    if ax.lines:
        ax.legend(loc="upper right", fontsize=8)
    for axk, k in zip(axes[1:], keys):
        if side:
            axk.plot([s["start_s"] + WINDOW_S / 2 for s in side], [s.get(k) if s.get(k) is not None else np.nan for s in side],
                     ".-", ms=3, color="C2")
        axk.set_ylabel(k, fontsize=8)
        axk.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.savefig(png, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------

def _middle(x: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    n = int(seconds * sr)
    if x.shape[0] <= n:
        return x
    s = (x.shape[0] - n) // 2
    return x[s : s + n]


def _window_starts(n_samples: int, sr: int) -> list[float]:
    dur = n_samples / sr
    starts, t = [], 0.0
    while t + 3.0 <= dur:  # same minimum partial length as the crate (3 s)
        starts.append(round(t, 3))
        t += HOP_S
    return starts or [0.0]


def side_features(mono_norm: np.ndarray, sr: int, starts: list[float]) -> list[dict]:
    import wmg_features
    y = np.ascontiguousarray(mono_norm, dtype=np.float64)
    if sr != SIDE_SR:
        y = np.asarray(wmg_features.resample(y, int(sr), SIDE_SR), dtype=np.float64)
    n = int(WINDOW_S * SIDE_SR)
    out = []
    for s in starts:
        i = int(round(s * SIDE_SR))
        seg = y[i : i + n]
        if seg.size < n:
            seg = np.pad(seg, (0, n - seg.size))
        vals = wmg_features.side_features_raw(np.ascontiguousarray(seg))
        out.append({"start_s": s, **{k: float(v) for k, v in zip(wmg_features.SIDE_FEATURES, vals)}})
    return out


def analyze(path: Path, *, meta: dict | None = None, models: bool = True, force: bool = False,
            log=None) -> Analysis:
    """Analyse `path`; cached under data/analyses/<sha1>/ unless `force`."""
    from wmg_train import handcrafted
    from wmg_train.augment import decode
    from wmg_train.score import normalize_level

    path = Path(path).resolve()
    say = log or (lambda s: None)
    aid = sha1_of(path)
    meta = {k: v for k, v in (meta or {}).items() if v not in (None, "")}
    cached = None if force else load(aid)
    if cached is not None:
        for k, v in meta.items():
            setattr(cached, k, v)
        if meta:
            cached.save()
        say(f"cached: {path.name}")
        return cached

    a = Analysis(id=aid, name=path.name, source_path=str(path),
                 created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), **meta)
    a.dir.mkdir(parents=True, exist_ok=True)
    ref = reference()
    a.reference_folder = Path(ref["folder"]).name if ref.get("folder") else None

    say(f"decoding {path.name}")
    with tempfile.TemporaryDirectory(prefix="wmg_an_") as d:
        x, sr = decode(path, Path(d))
    mono = x.astype(np.float64).mean(axis=1)
    a.profile = profile_file(path, x, sr)
    mid = _middle(mono, sr, FEATURE_SECONDS)

    say("hand-crafted features (raw)")
    try:
        a.features["raw"] = handcrafted.extract(mid, sr)
    except Exception as e:  # noqa: BLE001
        a.errors.append(f"features raw: {type(e).__name__}: {e}")
        a.features["raw"] = {}
    say("hand-crafted features (aac128 re-encode)")
    row = handcrafted.file_features(path, FEATURE_SECONDS, reencode=("aac", 128))
    if row.get("error"):
        a.errors.append(f"features aac128: {row['error']}")
        a.features["aac128"] = {}
    else:
        a.features["aac128"] = {k: row[k] for k in handcrafted.FEATURES if k in row}
        if "mod_peak_hz" in row:
            a.features["aac128"]["mod_peak_hz"] = row["mod_peak_hz"]

    try:
        plot_spectrum(mid, sr, a.features.get("raw", {}).get("fc_hz"), a.dir / "spectrum.png", a.name)
        a.images["spectrum"] = "spectrum.png"
    except Exception as e:  # noqa: BLE001
        a.errors.append(f"spectrum plot: {type(e).__name__}: {e}")

    starts: list[float] | None = None
    if models and models_available():
        from wmg_train import score as wscore
        m10, m5 = get_model("10s"), get_model("5s")
        run10, run5 = model_path("10s").parent.name, model_path("5s").parent.name
        say(f"scoring with the 10 s model ({run10})")
        try:
            res10 = wscore.score_file(m10, path)
        except Exception as e:  # noqa: BLE001
            res10 = {"error": f"{type(e).__name__}: {e}"}
        if "error" in res10:
            a.scores["10s"] = {"run": run10, "error": res10["error"]}
            a.scores["5s"] = {"run": run5, "error": res10["error"]}
            a.errors.append(f"detector: {res10['error']}")
        else:
            a.scores["10s"] = {"run": run10, "window_s": WINDOW_S,
                               **{k: v for k, v in res10.items() if not k.startswith("_") and k != "path"}}
            starts = [w["start_s"] for w in res10["windows"]]
            try:
                wscore.plot(res10, a.dir / "heatmap_10s.png")
                a.images["heatmap_10s"] = "heatmap_10s.png"
            except Exception as e:  # noqa: BLE001
                a.errors.append(f"heatmap: {type(e).__name__}: {e}")
            say(f"scoring with the 5 s model ({run5})")
            try:
                a.scores["5s"] = _score_5s(m5, res10, run5)
            except Exception as e:  # noqa: BLE001
                a.scores["5s"] = {"run": run5, "error": f"{type(e).__name__}: {e}"}
                a.errors.append(f"5s model: {type(e).__name__}: {e}")
    else:
        a.models_skipped = True
        if models:
            a.errors.append("detector models not found (training/runs + installed run metadata)")

    say("side features")
    try:
        mono_norm = normalize_level(mono)
        a.side_features = side_features(mono_norm, sr, starts if starts is not None else _window_starts(mono.size, sr))
    except Exception as e:  # noqa: BLE001
        a.errors.append(f"side features: {type(e).__name__}: {e}")
    try:
        plot_timeline(a.scores, a.side_features, a.dir / "timeline.png", a.name)
        a.images["timeline"] = "timeline.png"
    except Exception as e:  # noqa: BLE001
        a.errors.append(f"timeline plot: {type(e).__name__}: {e}")

    a.save()
    say(f"done: {path.name}")
    return a


# ---------------------------------------------------------------------------
# Markdown summary (CLI reports)
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def _band(b) -> str:
    return "–" if not b else f"{_fmt(b[0])} / {_fmt(b[1])} / {_fmt(b[2])}"


def summary_markdown(analyses: list[Analysis], ref: dict | None = None) -> str:
    ref = ref if ref is not None else reference()
    lines = [f"# Analyzer report, {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    if ref.get("folder"):
        c = ref.get("counts", {}).get("raw", {})
        lines.append(f"Reference bands (p10 / p50 / p90) from `{Path(ref['folder']).name}`: "
                     f"{c.get('human', '?')} human and {c.get('ai', '?')} AI training files.")
    lines += ["", "Verdict from the window colours of both models (P(AI) < 0.4 green, 0.4–0.7 yellow, ≥ 0.7 red): "
              "mostly red → AI-generated; red+yellow majority → likely AI (~75–80%); mostly green → likely human; "
              "otherwise undetermined.",
              "", "| file | category | verdict | P(AI) 10 s | P(AI) 5 s | rate | codec | kbps | duration |",
              "|---|---|---|---|---|---|---|---|---|"]
    for a in analyses:
        p, v = a.profile, a.verdict()
        lines.append(f"| {a.name} | {a.category or '–'} | {v['label']} ({v['estimate']}) | {_fmt(a.p_ai('10s'))} | {_fmt(a.p_ai('5s'))} | "
                     f"{p.get('sample_rate', '–')} | {p.get('codec', '–')} | {p.get('bitrate_kbps', '–')} | {_fmt(p.get('duration_s'))} s |")
    for a in analyses:
        lines += ["", f"## {a.name}", ""]
        lines.append(f"- source: `{a.source_path}`")
        lines.append(f"- sha1: `{a.id}`; analysed {a.created_at}")
        if a.category or a.generator or a.artist or a.notes:
            lines.append(f"- category {a.category or '–'}, generator {a.generator or '–'}, artist {a.artist or '–'}; {a.notes}")
        v = a.verdict()
        lines.append(f"- verdict: {v['label']}" + (f", {v['confidence']} confidence, about {v['estimate']} chance of AI; {v['reason']}"
                                                   if v["n"] else ""))
        for m in MODEL_NAMES:
            s = a.scores.get(m) or {}
            if s.get("error"):
                lines.append(f"- {m} model ({s.get('run')}): {s['error']}")
            elif s:
                bands = " ".join(f"{k}={v:.2f}" for k, v in s.get("band_scores", {}).items())
                lines.append(f"- {m} model ({s.get('run')}): P(AI) {s['p_ai']:.3f}, bands {bands}, {len(s.get('windows', []))} windows")
        if a.models_skipped:
            lines.append("- detector models skipped")
        if a.side_features:
            for k in ("flux_p2m", "mod_peak_db", "sfm_mean"):
                vals = [s[k] for s in a.side_features if s.get(k) is not None]
                if vals:
                    lines.append(f"- side feature {k}: median {np.median(vals):.3g} over {len(vals)} windows")
        lines += ["", "| feature | raw | vs human (p10/p50/p90) | vs AI | sits in | aac128 | vs human | vs AI | sits in |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in feature_rows(a, ref):
            c1, c2 = r["conds"]["raw"], r["conds"]["aac128"]
            lines.append(f"| {r['name']} | {_fmt(c1['value'])} | {_band(c1['human'])} | {_band(c1['ai'])} | {c1['where']} | "
                         f"{_fmt(c2['value'])} | {_band(c2['human'])} | {_band(c2['ai'])} | {c2['where']} |")
        if a.errors:
            lines += ["", "Problems:"] + [f"- {e}" for e in a.errors]
    return "\n".join(lines) + "\n"
