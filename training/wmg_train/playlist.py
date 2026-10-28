"""Real phone-chain training data: play a labelled playlist into the app, then auto-label clips.

1. Build a playlist of short excerpts from labelled songs (one audio file + a manifest):

    python -m wmg_train.playlist make --n-human 30 --n-ai 30 --excerpt 60

   -> reports/playlists/<playlist_id>/playlist.m4a, playlist.json

2. Play playlist.m4a through a phone speaker while the app records it in live mode.

3. Match the recorded session's 30 s clips back to the playlist and cut labelled segments:

    python -m wmg_train.playlist match <playlist_id> [--session <uuid>] [--api http://localhost:9000]

   -> reports/phone_captures/<session_id>/segments/*.wav, segments.csv

Alignment cross-correlates spectral-flux onset envelopes (150 Hz - 6 kHz, 100 frames/s), which
survive a phone speaker's bass cut, colouration and room echo. Each clip is located on its own,
so pauses or skips in playback only lose the affected clips.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from . import labels
from .augment import decode
from .groups import inventory

REPO = Path(__file__).resolve().parents[2]
PLAYLISTS = REPO / "reports" / "playlists"
CAPTURES = REPO / "reports" / "phone_captures"
DATA = REPO / "training_data"
PLAYLIST_SR = 44_100
GAP_S, FADE_S = 2.0, 0.5
ENV_RATE = 100  # envelope frames per second
MIN_SEGMENT_S = 10.0  # one detector window
EDGE_MARGIN_S = 0.75  # trimmed from each side of a segment (fades, alignment error)
MIN_CORR = 0.25  # normalised cross-correlation needed to accept a clip's alignment


# ---------------------------------------------------------------------------
# Onset envelope + alignment
# ---------------------------------------------------------------------------

def onset_envelope(x: np.ndarray, sr: int) -> np.ndarray:
    """Spectral-flux envelope at ENV_RATE frames/s over 8 log bands in 150 Hz - 6 kHz."""
    hop = sr // ENV_RATE
    n_fft = 2048 if sr > 24_000 else 1024
    n = max(0, (len(x) - n_fft) // hop + 1)
    if n < 2:
        return np.zeros(max(n, 1))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = np.geomspace(150, 6000, 9)
    band_of = np.digitize(freqs, edges) - 1  # -1 / 8 = outside
    win = np.hanning(n_fft)
    bands = np.zeros((n, 8))
    for start in range(0, n, 4096):  # chunked to bound memory on hour-long inputs
        idx = np.arange(start, min(n, start + 4096))
        frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[idx * hop] * win
        p = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        for b in range(8):
            bands[idx, b] = p[:, band_of == b].sum(axis=1)
    logb = np.log(bands + 1e-10)
    flux = np.maximum(np.diff(logb, axis=0), 0).sum(axis=1)
    flux = np.r_[0.0, flux]
    # Remove slow loudness trends so only rhythmic/onset structure is compared.
    k = ENV_RATE * 3
    trend = np.convolve(flux, np.ones(k) / k, mode="same")
    return flux - trend


def normalized_xcorr(clip: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Pearson correlation of `clip` against every placement inside `ref` (valid lags only)."""
    m = len(clip)
    if len(ref) < m:
        return np.array([])
    c = (clip - clip.mean()) / (clip.std() * m + 1e-12)
    size = 1 << int(np.ceil(np.log2(len(ref) + m)))
    raw = np.fft.irfft(np.fft.rfft(ref, size) * np.conj(np.fft.rfft(c, size)), size)[: len(ref) - m + 1]
    cs, cs2 = np.r_[0, np.cumsum(ref)], np.r_[0, np.cumsum(ref**2)]
    s, s2 = cs[m:] - cs[:-m], cs2[m:] - cs2[:-m]
    std = np.sqrt(np.maximum(s2 / m - (s / m) ** 2, 1e-12))
    return raw / std


# ---------------------------------------------------------------------------
# make
# ---------------------------------------------------------------------------

def _pick(songs, n, rng):
    """Round-robin over artist groups so one prolific artist can't dominate the playlist."""
    by_group: dict[str, list] = {}
    for s in songs:
        by_group.setdefault(s.group, []).append(s)
    for v in by_group.values():
        rng.shuffle(v)
    groups = list(by_group)
    rng.shuffle(groups)
    out = []
    while len(out) < n and any(by_group.values()):
        for g in groups:
            if by_group[g] and len(out) < n:
                out.append(by_group[g].pop())
    return out


def cmd_make(args) -> None:
    rng = np.random.default_rng(args.seed)
    songs = inventory(DATA)
    picked = _pick([s for s in songs if s.label == 0], args.n_human, rng) + \
        _pick([s for s in songs if s.label == 1], args.n_ai, rng)
    rng.shuffle(picked)
    pid = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-h{args.n_human}-a{args.n_ai}-s{args.seed}"
    out = PLAYLISTS / pid
    out.mkdir(parents=True)

    gap = np.zeros(int(GAP_S * PLAYLIST_SR), dtype=np.float32)
    fade = np.linspace(0, 1, int(FADE_S * PLAYLIST_SR), dtype=np.float32)[:, None]
    parts, entries, t = [gap], [], GAP_S
    with tempfile.TemporaryDirectory() as d:
        for i, s in enumerate(picked):
            x, sr = decode(s.path, Path(d))
            if sr != PLAYLIST_SR:
                import wmg_features
                x = np.stack([wmg_features.resample(np.ascontiguousarray(x[:, c], dtype=np.float64), sr, PLAYLIST_SR)
                              for c in range(x.shape[1])], axis=1).astype(np.float32)
                sr = PLAYLIST_SR
            if x.shape[1] == 1:
                x = np.repeat(x, 2, axis=1)
            dur = len(x) / sr
            ex = min(args.excerpt, dur)
            # Start somewhere in the middle 80% so intros/outros don't dominate.
            lo, hi = 0.1 * dur, max(0.1 * dur, 0.9 * dur - ex)
            start = float(rng.uniform(lo, hi)) if hi > lo else 0.0
            seg = x[int(start * sr): int(start * sr) + int(ex * sr), :2].copy()
            seg[: len(fade)] *= fade
            seg[-len(fade):] *= fade[::-1]
            parts += [seg, np.zeros((len(gap), 2), np.float32)]
            entries.append({
                "index": i, "song_id": s.song_id, "label": s.label, "group": s.group,
                "path": str(s.path.relative_to(REPO)), "source_start_s": round(start, 3),
                "playlist_start_s": round(t, 3), "duration_s": round(len(seg) / sr, 3),
            })
            t += len(seg) / sr + GAP_S
            print(f"{i + 1:3d}/{len(picked)} {'AI   ' if s.label else 'human'} {s.path.name}", flush=True)
        parts[0] = np.zeros((len(gap), 2), np.float32)
        audio = np.concatenate(parts)
        wav = Path(d) / "playlist.wav"
        sf.write(str(wav), audio, PLAYLIST_SR, subtype="FLOAT")
        # 256 kbps AAC: transparent enough that the phone chain, not this encode, dominates.
        subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "256000", str(wav),
                        str(out / "playlist.m4a")], check=True, capture_output=True)
    env = onset_envelope(audio.mean(axis=1).astype(np.float64), PLAYLIST_SR)
    np.save(out / "envelope.npy", env.astype(np.float32))
    meta = {"playlist_id": pid, "sample_rate": PLAYLIST_SR, "gap_s": GAP_S, "env_rate": ENV_RATE,
            "duration_s": round(len(audio) / PLAYLIST_SR, 1), "n_human": sum(e["label"] == 0 for e in entries),
            "n_ai": sum(e["label"] == 1 for e in entries), "seed": args.seed, "entries": entries}
    (out / "playlist.json").write_text(json.dumps(meta, indent=1))
    print(f"\nplaylist {pid}: {meta['duration_s'] / 60:.1f} min, {meta['n_human']} human + {meta['n_ai']} AI\n"
          f"  play this file: {out / 'playlist.m4a'}")


# ---------------------------------------------------------------------------
# match
# ---------------------------------------------------------------------------

def _session_clips(api: str, session: str | None):
    from .score_live import download, get_json
    clips = [m for m in get_json(f"{api}/audio") if m.get("session_id")]
    if not clips:
        raise SystemExit(f"no live clips on {api}")
    if session is None:
        session = max(clips, key=lambda m: m["created_at"])["session_id"]
    clips = sorted((m for m in clips if m["session_id"] == session), key=lambda m: m["clip_index"])
    out = CAPTURES / session / "audio"
    out.mkdir(parents=True, exist_ok=True)
    files = [download(f"{api}/audio/{m['id']}/stream", out / f"clip_{m['clip_index']:03d}.wav") for m in clips]
    return session, [(m["clip_index"], m["id"], f) for m, f in zip(clips, files)]


def _wav_clips(wav: Path, clip_s: float = 30.0):
    """Treat a local recording as a session (for testing): cut it into 30 s clips."""
    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    sid = "local-" + hashlib.sha1(wav.read_bytes()[:1_000_000]).hexdigest()[:10]
    out = CAPTURES / sid / "audio"
    out.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, a in enumerate(range(0, len(x), int(clip_s * sr))):
        f = out / f"clip_{i:03d}.wav"
        if not f.exists():
            sf.write(str(f), x[a: a + int(clip_s * sr)], sr, subtype="PCM_16")
        clips.append((i, None, f))
    return sid, clips


def cmd_match(args) -> None:
    pdir = PLAYLISTS / args.playlist
    meta = json.loads((pdir / "playlist.json").read_text())
    ref = np.load(pdir / "envelope.npy").astype(np.float64)
    session, clips = _wav_clips(args.wav) if args.wav else _session_clips(args.api.rstrip("/"), args.session)
    sdir = CAPTURES / session
    seg_dir = sdir / "segments"
    if (sdir / "segments.csv").exists() and not args.force:
        raise SystemExit(f"{sdir}/segments.csv exists; pass --force to redo this session")
    seg_dir.mkdir(parents=True, exist_ok=True)

    # 1) Locate each clip independently in the playlist.
    found = []
    for idx, clip_id, f in clips:
        x, sr = sf.read(str(f), dtype="float64", always_2d=True)
        mono = x.mean(axis=1)
        env = onset_envelope(mono, sr)
        if len(env) < ENV_RATE * 5:
            print(f"  clip {idx:3d}: too short to align")
            continue
        xc = normalized_xcorr(env, ref)
        lag = int(np.argmax(xc))
        found.append({"idx": idx, "clip_id": clip_id, "file": f, "sr": sr, "mono": mono,
                      "start": lag / ENV_RATE, "corr": float(xc[lag]), "xc": xc})

    # 2) While playback runs continuously, consecutive clips share one offset (playlist time minus
    #    session time). A pause or skip starts a new offset, so a clip is trusted when its offset
    #    agrees with a neighbouring clip's; a clip containing the pause itself matches neither.
    pos, t = {}, 0.0  # each clip's start time within the session
    for idx, _, f in clips:
        pos[idx] = t
        t += sf.info(str(f)).duration
    for c in found:
        c["offset"] = c["start"] - pos[c["idx"]]
    for i, c in enumerate(found):
        nbrs = [found[j] for j in (i - 1, i + 1) if 0 <= j < len(found) and abs(found[j]["idx"] - c["idx"]) == 1]
        agree = any(abs(n["offset"] - c["offset"]) < 1.0 and n["corr"] >= MIN_CORR for n in nbrs)
        c["status"] = ("low_corr" if c["corr"] < MIN_CORR else
                       "ok" if agree or (not nbrs and c["corr"] >= 2 * MIN_CORR) else "no_neighbour_agreement")
    # 3) Second pass: a rejected clip may still sit at the offset of the nearest accepted clip
    #    (a repetitive beat can make a wrong peak win narrowly). Re-check it there, +-0.2 s.
    ok_idx = [c for c in found if c["status"] == "ok"]
    for c in found:
        if c["status"] == "ok" or not ok_idx:
            continue
        ref_clip = min(ok_idx, key=lambda o: abs(o["idx"] - c["idx"]))
        lag0 = round((ref_clip["offset"] + pos[c["idx"]]) * ENV_RATE)
        lo, hi = max(0, lag0 - 20), min(len(c["xc"]), lag0 + 21)
        if lo >= hi:
            continue  # clip extends past the playlist start/end at that offset
        lag = lo + int(np.argmax(c["xc"][lo:hi]))
        if c["xc"][lag] >= MIN_CORR:
            c.update(start=lag / ENV_RATE, corr=float(c["xc"][lag]), status="ok")
            c["offset"] = c["start"] - pos[c["idx"]]
            c["refined"] = True
    if not any(c["status"] == "ok" for c in found):
        raise SystemExit("no clip aligned with the playlist (wrong playlist, or playback inaudible)")

    entries = meta["entries"]
    rows = []
    for c in found:
        status = c["status"]
        dur = len(c["mono"]) / c["sr"]
        print(f"  clip {c['idx']:3d}: playlist {c['start']:7.1f}s  offset {c['offset']:+7.1f}s  "
              f"corr {c['corr']:.2f}  {status}{' (re-checked at neighbour offset)' if c.get('refined') else ''}")
        if status != "ok":
            continue
        for e in entries:
            a = max(c["start"], e["playlist_start_s"]) + EDGE_MARGIN_S
            b = min(c["start"] + dur, e["playlist_start_s"] + e["duration_s"]) - EDGE_MARGIN_S
            if b - a < MIN_SEGMENT_S:
                continue
            i0, i1 = int((a - c["start"]) * c["sr"]), int((b - c["start"]) * c["sr"])
            name = f"clip{c['idx']:03d}_{'ai' if e['label'] else 'human'}_{e['song_id']}.wav"
            sf.write(str(seg_dir / name), c["mono"][i0:i1], c["sr"], subtype="PCM_16")
            rows.append({
                "session_id": session, "playlist_id": meta["playlist_id"], "clip_index": c["idx"],
                "clip_id": c["clip_id"] or "", "segment": f"segments/{name}",
                **labels.columns("ai" if e["label"] == 1 else "human"),
                "song_id": e["song_id"], "group": e["group"], "source_path": e["path"],
                "source_start_s": round(e["source_start_s"] + (a - e["playlist_start_s"]), 2),
                "duration_s": round(b - a, 2), "align_corr": round(c["corr"], 3), "sample_rate": c["sr"],
            })
    with (sdir / "segments.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["session_id"])
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(1 for c in found if c["status"] == "ok")
    print(f"\n{session}: {n_ok}/{len(clips)} clips aligned, "
          f"{len(rows)} labelled segments ({sum(r['label'] for r in rows)} AI, "
          f"{sum(1 - r['label'] for r in rows)} human), "
          f"{sum(r['duration_s'] for r in rows) / 60:.1f} min -> {sdir}")


def cmd_label_session(args) -> None:
    """Label the clips of a live session: one label for all of them (e.g. a streaming playlist
    that is all human-made), and/or a label per clip (--clip-label) where they differ. Nothing
    to align against, so each 30 s clip becomes a segment as recorded.
    The session is one CV group (artists are unknown), so it is held out as a whole."""
    per_clip = {}
    for item in args.clip_label:
        idx, _, lab = item.partition("=")
        if not idx.strip().isdigit() or lab not in labels.CATEGORIES:
            raise SystemExit(f"--clip-label {item!r}: expected INDEX=LABEL with LABEL one of {list(labels.CATEGORIES)}")
        per_clip[int(idx)] = lab
    if args.label is None and not per_clip:
        raise SystemExit("pass --label for the whole session and/or --clip-label INDEX=LABEL")
    session, clips = _session_clips(args.api.rstrip("/"), args.session)
    sdir = CAPTURES / session
    if (sdir / "segments.csv").exists() and not args.force:
        raise SystemExit(f"{sdir}/segments.csv exists; pass --force to redo this session")
    (sdir / "segments").mkdir(parents=True, exist_ok=True)
    # A name with a prefix ("ai:The Velvet Sundown", "human:CHVRCHES") joins that existing training
    # group, so CV holds the phone clips out together with the artist's clean files.
    group = args.group if args.group and ":" in args.group else f"phone:{args.group or session[:8]}"
    rows = []
    for idx, clip_id, f in clips:
        info = sf.info(str(f))
        if info.duration < MIN_SEGMENT_S:
            print(f"  clip {idx:3d}: {info.duration:.1f} s, shorter than one window; skipped")
            continue
        lab = per_clip.get(idx, args.label)
        if lab is None:
            print(f"  clip {idx:3d}: no label; left out")
            continue
        name = f"clip{idx:03d}_{lab}.wav"
        dest = sdir / "segments" / name
        if not dest.exists():
            dest.write_bytes(Path(f).read_bytes())
        rows.append({
            "session_id": session, "playlist_id": "", "clip_index": idx, "clip_id": clip_id or "",
            "segment": f"segments/{name}", **labels.columns(lab),
            # Songs are unknown; each clip stands in for a "song" when scores are aggregated.
            "song_id": f"{session[:8]}-{idx:03d}", "group": group, "source_path": args.source,
            "source_start_s": "", "duration_s": round(info.duration, 2), "align_corr": "",
            "sample_rate": info.samplerate,
        })
    if not rows:
        raise SystemExit("no labelled clip long enough to use")
    with (sdir / "segments.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = ", ".join(f"{sum(r['category'] == c for r in rows)} {c}" for c in labels.CATEGORIES
                       if any(r["category"] == c for r in rows))
    print(f"{session}: {len(rows)} clips labelled ({counts}) "
          f"({sum(r['duration_s'] for r in rows) / 60:.1f} min), group {group} -> {sdir}")


def cmd_score(args) -> None:
    """Score a capture's labelled segments with a model; writes scores_<model_run>.csv."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from .score import load_model, score_file

    sdir = CAPTURES / args.session
    segs = pd.read_csv(sdir / "segments.csv")
    segs["category"] = [labels.category_of(r) for r in segs.to_dict("records")]
    n_assisted = int((segs.category == "ai_assisted").sum())
    segs = segs[segs.category.isin(labels.BINARY)]
    if n_assisted:
        print(f"  ({n_assisted} AI-assisted segments kept but not scored against Human-vs-AI)")
    if segs.empty:
        raise SystemExit("no Human/AI segments to score in this capture")
    model_run = args.model.resolve().parent.name
    norm = not args.no_normalize
    out = sdir / f"scores_{model_run}{'' if norm else '_unnormalized'}.csv"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to rescore")
    model = load_model(args.model)
    rows = []
    for r in segs.itertuples():
        res = score_file(model, sdir / r.segment, normalize=norm)
        if "error" in res:
            continue
        logit = float(np.log(res["p_ai"] + 1e-9) - np.log(1 - res["p_ai"] + 1e-9))
        rows.append({"segment": r.segment, "label": r.label, "song_id": r.song_id, "group": r.group,
                     "duration_s": r.duration_s, "p_ai": res["p_ai"], "logit": logit,
                     **{f"band_{k}": v for k, v in res["band_scores"].items()}})
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    songs = df.groupby(["song_id", "label", "group"], as_index=False).agg(
        logit=("logit", "mean"), n=("logit", "size"),
        treble=("band_treble", "mean"))
    songs["p_ai"] = 1 / (1 + np.exp(-songs.logit))
    print(f"{model_run}: {len(df)} segments, {len(songs)} songs")
    for r in songs.sort_values(["label", "p_ai"]).itertuples():
        print(f"  {'AI   ' if r.label else 'human'} P(AI)={r.p_ai:.3f}  treble={r.treble:.2f}  "
              f"({r.n} seg)  {r.group}")
    if songs.label.nunique() == 2:
        ai, hu = songs[songs.label == 1], songs[songs.label == 0]
        print(f"  song AUC {roc_auc_score(songs.label, songs.logit):.3f} | "
              f"AI caught@0.5 {(ai.p_ai > 0.5).sum()}/{len(ai)} | human flagged@0.5 {(hu.p_ai > 0.5).sum()}/{len(hu)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="build a labelled excerpt playlist")
    mk.add_argument("--n-human", type=int, default=30)
    mk.add_argument("--n-ai", type=int, default=30)
    mk.add_argument("--excerpt", type=float, default=60.0, help="seconds per song")
    mk.add_argument("--seed", type=int, default=0)
    mt = sub.add_parser("match", help="align a recorded session to a playlist and cut labelled segments")
    mt.add_argument("playlist", help="playlist id (folder name under reports/playlists)")
    mt.add_argument("--session", help="live session id (default: most recent)")
    mt.add_argument("--api", default="http://localhost:9000")
    mt.add_argument("--wav", type=Path, help="use a local recording instead of a backend session")
    mt.add_argument("--force", action="store_true")
    ls = sub.add_parser("label-session", help="label every clip of a live session (e.g. an all-human streaming playlist)")
    ls.add_argument("--session", help="live session id (default: most recent)")
    ls.add_argument("--label", choices=list(labels.CATEGORIES),
                    help="label for every clip: human, ai, or ai_assisted (human-made with AI parts, "
                         "~10-50%% AI; kept but not used for Human-vs-AI training yet)")
    ls.add_argument("--clip-label", action="append", default=[], metavar="INDEX=LABEL",
                    help="label one clip differently, e.g. --clip-label 3=ai (repeatable). Clips with "
                         "neither this nor --label are left out")
    ls.add_argument("--source", default="", help="what was played, e.g. 'Spotify: Indie Mix'")
    ls.add_argument("--group", help="CV group name (default: the session id); reuse a name to keep "
                                    "sessions of the same playlist/artists in one group. Use an existing "
                                    "group such as 'ai:The Velvet Sundown' if every song is by an artist "
                                    "already in training_data")
    ls.add_argument("--api", default="http://localhost:9000")
    ls.add_argument("--force", action="store_true")
    sc = sub.add_parser("score", help="score a capture's labelled segments with a model")
    sc.add_argument("session", help="session id (folder under reports/phone_captures)")
    sc.add_argument("model", type=Path)
    sc.add_argument("--force", action="store_true")
    sc.add_argument("--no-normalize", action="store_true", help="score at the recording's own level")
    args = ap.parse_args()
    {"make": cmd_make, "match": cmd_match, "label-session": cmd_label_session,
     "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
