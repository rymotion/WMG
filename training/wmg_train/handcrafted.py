"""Hand-crafted top-end features for AI-music detection, and an evaluation of them.

Seven numbers per clip, all computed from a linear-frequency STFT at the clip's native rate
(nothing is resampled, so anything above 16 kHz is kept):

  group 1, bandwidth (needs the delivery file; a codec or a speaker hides it)
    fc_hz        effective bandwidth: highest frequency still within 30 dB of the 2-5 kHz peak
    notch_db     dip at 15-16 kHz relative to 13-14 kHz and 17-18 kHz (NaN below 36 kHz)
    slope_break  log-log spectral slope 16-20 kHz minus slope 10-14 kHz (dB/octave; NaN < 40 kHz)
    residue_db   energy above fc+1 kHz over energy below fc, in dB
  group 2, smeared top end (8-16 kHz)
    sfm_mean / sfm_var   spectral flatness per frame in 8-16 kHz: mean and variance
    flux_p2m     spectral flux in 8-16 kHz, 99th percentile over median
    crest_db     8-16 kHz band-passed waveform, median over 1 s of peak / RMS
  group 3, codec frame rate
    mod_peak_db  sharpest narrow peak of the 8-16 kHz envelope's modulation spectrum, 30-120 Hz,
                 over its local median; mod_peak_hz says where
    mod_codec_db the same at the usual neural-codec frame rates (50, 75, 86 Hz)
    ripple_db    periodic ripple along the frequency axis of the long-term spectrum (checkerboard)

Evaluate on everything we have:

    python -m wmg_train.handcrafted eval [--seconds 60] [--workers 4]

Conditions: training_data at native rate; the same after a 128 kbps AAC round trip; the
playlist excerpts cut from the playlist file; the phone captures. Logistic regression with
leave-one-AI-artist-out folds on the clean files, then that model applied to the others.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from . import labels
from .augment import decode, encode_decode
from .groups import inventory
from .train_baseline import make_folds

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "training_data"
PLAYLISTS = REPO / "reports" / "playlists"
CAPTURES = REPO / "reports" / "phone_captures"
OUT = REPO / "reports" / "handcrafted"

HI = (8_000.0, 16_000.0)
CODEC_RATES_HZ = (50.0, 75.0, 86.0)
FEATURES = ["fc_hz", "notch_db", "slope_break", "residue_db", "sfm_mean", "sfm_var", "flux_p2m",
            "crest_db", "mod_peak_db", "mod_codec_db", "ripple_db"]
GROUPS = {"bandwidth": FEATURES[:4], "smear": FEATURES[4:8], "codec": FEATURES[8:]}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _stft_power(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """|STFT|^2 as [bins, frames], Hann window, no padding."""
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    w = signal.windows.hann(n_fft, sym=False).astype(np.float32)
    spec = np.fft.rfft(frames.astype(np.float32) * w, axis=1)
    return (spec.real ** 2 + spec.imag ** 2).T


def _db(p: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(p, 1e-20))


def _band_mean_db(ldb: np.ndarray, f: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f < hi)
    return float(ldb[m].mean()) if m.any() else math.nan


def _loglog_slope(ldb: np.ndarray, f: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f < hi)
    if m.sum() < 8:
        return math.nan
    return float(np.polyfit(np.log2(f[m]), ldb[m], 1)[0])  # dB per octave


def _peakiness_db(spec: np.ndarray, f: np.ndarray, centre: float, half_width: float, local: float) -> float:
    """Peak within +-half_width of `centre` over the median of +-local around it (excluding the peak)."""
    near = np.abs(f - centre) <= half_width
    around = (np.abs(f - centre) <= local) & ~near
    if not near.any() or around.sum() < 4:
        return math.nan
    return float(_db(spec[near].max()) - _db(np.median(spec[around])))


def extract(x: np.ndarray, sr: int) -> dict[str, float]:
    """The feature vector for mono float `x` at `sr`. NaN where the rate can't support a feature."""
    x = np.asarray(x, dtype=np.float32)
    nyq = sr / 2.0
    out: dict[str, float] = {}

    # Long-term spectrum, fine frequency resolution.
    n_fft = 4096 if sr > 36_000 else 2048
    P = _stft_power(x, n_fft, n_fft // 2).mean(axis=1)
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    ldb = _db(P)
    # Smooth by 5 bins so one noisy bin can't set the cutoff.
    sm = np.convolve(ldb, np.ones(5) / 5, mode="same")
    mid = (f >= 2_000) & (f < 5_000)
    ref = sm[mid].max()
    above = np.nonzero(sm > ref - 30.0)[0]
    fc = float(f[above.max()]) if above.size else 0.0
    out["fc_hz"] = fc
    out["notch_db"] = ((_band_mean_db(ldb, f, 13_000, 14_000) + _band_mean_db(ldb, f, 17_000, 18_000)) / 2
                       - _band_mean_db(ldb, f, 15_000, 16_000)) if nyq >= 18_000 else math.nan
    out["slope_break"] = (_loglog_slope(ldb, f, 16_000, 20_000) - _loglog_slope(ldb, f, 10_000, 14_000)) if nyq >= 20_000 else math.nan
    hi_m, lo_m = f > fc + 1_000, f < fc
    # Nothing above the cutoff (the band reaches Nyquist) is "no residue": the floor, not NaN.
    out["residue_db"] = max(-100.0, float(_db(P[hi_m].sum()) - _db(P[lo_m].sum()))) if hi_m.any() and lo_m.any() else -100.0
    # Checkerboard: periodic ripple along frequency of the long-term spectrum, between 2 kHz
    # and the cutoff (so the cutoff cliff itself can't register), ripple periods 100 Hz - 2 kHz.
    m = (f >= 2_000) & (f < fc - 500)
    if m.sum() > 64:
        r = signal.detrend(ldb[m])
        rs = np.abs(np.fft.rfft(r * signal.windows.hann(r.size)))
        q = np.fft.rfftfreq(r.size, d=float(f[1] - f[0]))  # cycles per Hz
        keep = (q > 0) & (1.0 / np.maximum(q, 1e-12) >= 100) & (1.0 / np.maximum(q, 1e-12) <= 2_000)
        out["ripple_db"] = float(_db(rs[keep].max()) - _db(np.median(rs[keep]))) if keep.sum() > 8 else math.nan
    else:
        out["ripple_db"] = math.nan

    # Frame-level analysis of the 8-16 kHz band.
    n_fft2, hop2 = 1024, 128
    S = _stft_power(x, n_fft2, hop2)
    f2 = np.fft.rfftfreq(n_fft2, 1.0 / sr)
    hi = (f2 >= HI[0]) & (f2 < min(HI[1], nyq))
    if hi.sum() < 4 or S.shape[1] < 16:
        for k in FEATURES[4:10]:
            out[k] = math.nan
        return out
    Sh = S[hi]
    fr = sr / hop2  # frames per second
    # Only frames with some signal in the band; a silent gap would pin flatness to 1.
    active = Sh.sum(axis=0) > 10 ** ((_db(Sh.sum(axis=0)).max() - 60) / 10)
    Sa = np.maximum(Sh[:, active], 1e-20)
    sfm = np.exp(np.log(Sa).mean(axis=0)) / Sa.mean(axis=0)
    out["sfm_mean"], out["sfm_var"] = float(sfm.mean()), float(sfm.var())
    L = np.log(np.maximum(Sh, 1e-20))
    flux = np.sum(np.maximum(0.0, np.diff(L, axis=1)) ** 2, axis=0)
    med = float(np.median(flux))
    out["flux_p2m"] = float(np.percentile(flux, 99) / med) if med > 0 else math.nan
    # Crest factor of the band-passed waveform, per second, median.
    sos = signal.butter(4, [HI[0], min(HI[1], nyq * 0.98)], btype="band", fs=sr, output="sos")
    xb = signal.sosfiltfilt(sos, x.astype(np.float64))
    n = int(sr)
    crests = []
    for i in range(0, xb.size - n + 1, n):
        seg = xb[i : i + n]
        rms = np.sqrt(np.mean(seg ** 2))
        if rms > 1e-6:
            crests.append(20 * math.log10(np.abs(seg).max() / rms))
    out["crest_db"] = float(np.median(crests)) if crests else math.nan
    # Modulation spectrum of the band envelope (Welch over 4 s pieces).
    env = Sh.sum(axis=0)
    env = env / (env.mean() + 1e-20)
    seg_len = int(4 * fr)
    if env.size >= seg_len:
        fm, M = signal.welch(env - env.mean(), fs=fr, nperseg=seg_len, noverlap=seg_len // 2)
        band = (fm >= 30) & (fm <= 120)
        peaks = [_peakiness_db(M, fm, c, 1.0, 8.0) for c in fm[band]]
        best = int(np.nanargmax(peaks)) if peaks and not all(math.isnan(p) for p in peaks) else -1
        out["mod_peak_db"] = float(peaks[best]) if best >= 0 else math.nan
        out["mod_peak_hz"] = float(fm[band][best]) if best >= 0 else math.nan
        out["mod_codec_db"] = float(np.nanmax([_peakiness_db(M, fm, c, 1.0, 8.0) for c in CODEC_RATES_HZ]))
    else:
        out["mod_peak_db"] = out["mod_peak_hz"] = out["mod_codec_db"] = math.nan
    return out


def _middle(x: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    n = int(seconds * sr)
    if x.shape[0] <= n:
        return x
    s = (x.shape[0] - n) // 2
    return x[s : s + n]


def file_features(path: Path, seconds: float, reencode: tuple[str, int] | None = None,
                  start_s: float | None = None, duration_s: float | None = None) -> dict:
    """Decode `path` (an excerpt if start/duration are given), optionally round-trip it through
    a codec, and extract. Returns the features plus `sample_rate` and `error`."""
    row: dict = {"path": str(path), "error": ""}
    try:
        with tempfile.TemporaryDirectory(prefix="wmg_hc_") as d:
            tmp = Path(d)
            x, sr = decode(path, tmp)
            if start_s is not None:
                x = x[int(start_s * sr) : int((start_s + (duration_s or seconds)) * sr)]
            else:
                x = _middle(x, sr, seconds)
            if reencode is not None:
                x = encode_decode(x, sr, reencode[0], reencode[1], tmp)
        mono = x.astype(np.float64).mean(axis=1)
        row["sample_rate"] = sr
        row.update(extract(mono, sr))
    except Exception as e:  # noqa: BLE001 - one bad file must not stop the run
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def _job(args):
    meta, kw = args
    return {**meta, **file_features(**kw)}


def run_jobs(jobs: list[tuple[dict, dict]], workers: int) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = []
        for i, r in enumerate(ex.map(_job, jobs, chunksize=2), 1):
            rows.append(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"    {i}/{len(jobs)}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _fit(df: pd.DataFrame, feats: list[str]):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    m.fit(df[feats].to_numpy(dtype=float), df["label"].to_numpy())
    return m


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    if ok.sum() < 2 or len(set(y[ok])) < 2:
        return math.nan
    return float(roc_auc_score(y[ok], s[ok]))


def univariate(df: pd.DataFrame) -> dict[str, dict]:
    """Per-feature AUC (oriented so > 0.5 means AI scores higher on the raw value, reported as
    is), plus class medians and the share of NaNs."""
    out = {}
    for k in FEATURES:
        v = df[k].astype(float)
        out[k] = {"auc": _auc(df["label"], v),
                  "human_median": float(v[df.label == 0].median()),
                  "ai_median": float(v[df.label == 1].median()),
                  "nan": float(v.isna().mean())}
    return out


def cv_auc(df: pd.DataFrame, feats: list[str]) -> tuple[float, pd.Series]:
    """Leave-one-AI-artist-out folds over `group`; mean held-out AUC and out-of-fold scores."""
    idx = df.rename(columns={"group": "group"})[["song_id", "group", "label"]].copy()
    idx["source"] = "clean"
    folds = make_folds(idx)
    scores = pd.Series(np.nan, index=df.index)
    aucs = []
    for _ai_group, test_groups in folds:
        te = df["group"].isin(test_groups)
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        m = _fit(df[~te], feats)
        s = m.predict_proba(df.loc[te, feats].to_numpy(dtype=float))[:, 1]
        scores[te] = s
        aucs.append(_auc(df.loc[te, "label"], s))
    return float(np.nanmean(aucs)), scores


def apply_model(m, df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return m.predict_proba(df[feats].to_numpy(dtype=float))[:, 1]


def summarise(name: str, df: pd.DataFrame, lines: list[str], models: dict | None = None) -> None:
    df = df[df.error == ""]
    lines.append(f"\n## {name}: {len(df)} clips, {int((df.label == 1).sum())} AI / {int((df.label == 0).sum())} human, "
                 f"rates {sorted(df.sample_rate.unique().tolist())}")
    uni = univariate(df)
    lines.append("\n| feature | AUC alone | human median | AI median | NaN |")
    lines.append("|---|---|---|---|---|")
    for k, u in uni.items():
        lines.append(f"| {k} | {u['auc']:.3f} | {u['human_median']:.3g} | {u['ai_median']:.3g} | {u['nan']:.0%} |")
    if models is not None:
        lines.append("\n| feature set | AUC, model fitted on clean files | AUC, fitted on re-encoded files |")
        lines.append("|---|---|---|")
        for gname, feats in [("all", FEATURES)] + list(GROUPS.items()):
            a = _auc(df.label, apply_model(models["raw"][gname], df, feats))
            b = _auc(df.label, apply_model(models["aac128"][gname], df, feats))
            lines.append(f"| {gname} | {a:.3f} | {b:.3f} |")


def cmd_eval(args) -> None:
    out = OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    songs = inventory(DATA)
    print(f"{len(songs)} songs in training_data")
    lines = [f"# Hand-crafted features, {out.name}", f"\n{args.seconds:.0f} s from the middle of each song; "
             "features from a linear STFT at the file's own rate.", f"\nFeatures: {', '.join(FEATURES)}"]

    conds = {}
    for cond, enc in [("raw", None), ("aac128", ("aac", 128))]:
        print(f"== training_data, {cond}")
        if args.reuse and (args.reuse / f"training_{cond}.csv").is_file():
            df = pd.read_csv(args.reuse / f"training_{cond}.csv")
            df["error"] = df["error"].fillna("").astype(str)
            print(f"    reusing {args.reuse / f'training_{cond}.csv'}")
        else:
            jobs = [({"song_id": s.song_id, "group": s.group, "label": s.label, "cond": cond},
                     {"path": s.path, "seconds": args.seconds, "reencode": enc}) for s in songs]
            df = run_jobs(jobs, args.workers)
        df.to_csv(out / f"training_{cond}.csv", index=False)
        bad = df[df.error != ""]
        if len(bad):
            print(f"    {len(bad)} failed, e.g. {bad.iloc[0].error}")
        conds[cond] = df[df.error == ""].copy()

    models = {}
    for cond, df in conds.items():
        lines.append(f"\n# training_data, {cond}")
        summarise(f"training_data {cond}", df, lines)
        lines.append("\n| feature set | leave-one-AI-artist-out AUC |")
        lines.append("|---|---|")
        models[cond] = {}
        for gname, feats in [("all", FEATURES)] + list(GROUPS.items()):
            a, _ = cv_auc(df, feats)
            lines.append(f"| {gname} | {a:.3f} |")
            models[cond][gname] = _fit(df, feats)
        print("\n".join(lines[-6:]))
    # Cross: clean-fitted folds scored on the re-encoded copies of the held-out songs.
    lines.append("\n## Clean-fitted model on the re-encoded files (held-out artists)")
    raw, aac = conds["raw"], conds["aac128"]
    idx = raw[["song_id", "group", "label"]].copy(); idx["source"] = "clean"
    for gname, feats in [("all", FEATURES)] + list(GROUPS.items()):
        aucs = []
        for _g, test_groups in make_folds(idx):
            tr = raw[~raw.group.isin(test_groups)]
            te = aac[aac.group.isin(test_groups)]
            if len(tr) and len(te):
                aucs.append(_auc(te.label, apply_model(_fit(tr, feats), te, feats)))
        lines.append(f"- {gname}: {np.nanmean(aucs):.3f}")
    print("\n".join(lines[-5:]))

    # Playlist excerpts, cut from the playlist file itself (44.1 kHz AAC).
    for pl in sorted(PLAYLISTS.glob("*/playlist.json")):
        meta = json.loads(pl.read_text())
        print(f"== playlist {meta['playlist_id']}")
        jobs = [({"song_id": e["song_id"], "group": e["group"], "label": int(e["label"]), "cond": "playlist"},
                 {"path": pl.with_name("playlist.m4a"), "seconds": args.seconds,
                  "start_s": e["playlist_start_s"], "duration_s": e["duration_s"]}) for e in meta["entries"]]
        df = run_jobs(jobs, args.workers)
        df.to_csv(out / f"playlist_{meta['playlist_id']}.csv", index=False)
        summarise(f"playlist {meta['playlist_id']} (excerpts of training songs)", df, lines, models)
        print("\n".join(lines[-7:]))

    # Phone captures: every binary-labelled segment.
    print("== phone captures")
    jobs = []
    for csvp in sorted(CAPTURES.glob("*/segments.csv")):
        with open(csvp) as fh:
            for r in csv.DictReader(fh):
                cat = labels.category_of(r)
                if cat not in labels.BINARY:
                    continue
                jobs.append(({"song_id": r["song_id"], "group": r["group"], "label": labels.label_of(cat),
                              "session": r["session_id"][:8], "source": r["source_path"][:30], "cond": "phone"},
                             {"path": csvp.parent / r["segment"], "seconds": args.seconds}))
    df = run_jobs(jobs, args.workers)
    df.to_csv(out / "phone.csv", index=False)
    ok = df[df.error == ""]
    summarise("phone captures (all)", ok, lines, models)
    # Within-phone CV by session, so the phone data can also speak for itself.
    lines.append("\n| feature set | leave-sessions-out AUC within phone captures |")
    lines.append("|---|---|")
    for gname, feats in [("all", FEATURES)] + list(GROUPS.items()):
        a, _ = cv_auc(ok.assign(group=ok.session), feats)
        lines.append(f"| {gname} | {a:.3f} |")
    lines.append("\n### Per session, clean-fitted 'all' model: mean P(AI)")
    s = apply_model(models["raw"]["all"], ok, FEATURES)
    per = ok.assign(p=s).groupby(["session", "source", "label"]).p.agg(["mean", "count"]).reset_index()
    for r in per.itertuples():
        lines.append(f"- {r.session} {r.source!r} label {r.label}: P(AI) {r.mean:.2f} over {r.count}")
    print("\n".join(lines[-12:]))

    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out / 'summary.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="extract on all data and report")
    e.add_argument("--seconds", type=float, default=60.0)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--reuse", type=Path, help="earlier report folder whose training_*.csv are reused")
    e.set_defaults(fn=cmd_eval)
    f = sub.add_parser("features", help="print the features of one file")
    f.add_argument("path", type=Path)
    f.add_argument("--seconds", type=float, default=60.0)
    f.set_defaults(fn=lambda a: print(json.dumps(file_features(a.path, a.seconds), indent=2)))
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
