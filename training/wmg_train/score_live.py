"""Score clips recorded through the app's live mode (`/live`) against a trained BandNet.

    python -m wmg_train.score_live runs/<ts>/model.pt [--api http://localhost:9000]
        [--session <uuid> | --all-new] [--label human|ai] [--note "phone speaker, chorus loud"]

Nothing is ever overwritten. Layout under `--out` (default `<repo>/reports/scores/live`):

    index.csv                          one appended row per scoring run (the results log)
    <session_id>/audio/clip_NNN.wav    clips as downloaded, fetched once and reused
    <session_id>/audio/session.wav     clips stitched in order
    <session_id>/runs/<run_id>/        session.png, session.json, clips.json for that run

`run_id` is `<UTC time>-<model run>` (plus `-2`, `-3`, ... if it already exists), so re-scoring a
session, or scoring it with a newer model, adds a run next to the old ones.
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from . import features
from .augment import TARGET_RMS_DBFS
from .score import load_model, plot, score_file

DETECTOR_SR = 32_000
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "reports" / "scores" / "live"
INDEX_FIELDS = [
    "run_id", "scored_at", "session_id", "session_started_at", "duration_s", "sample_rate",
    "n_clips", "model_run", "p_ai", "band_bass", "band_mid", "band_treble", "max_window_p_ai",
    "label", "note", "run_dir", "input_rms_dbfs", "level_normalized_to_dbfs",
]


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def download(url: str, dest: Path) -> Path:
    if not dest.exists():  # clips never change once recorded; keep the first copy
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())
    return dest


def read_index(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def append_index(path: Path, row: dict) -> None:
    new = not path.exists()
    if not new:
        with path.open(newline="") as f:
            header = next(csv.reader(f), [])
        if header != INDEX_FIELDS:
            # Columns were added since this log started: rewrite it with the new header, keeping
            # every existing row and value (new columns are blank for old rows).
            old = read_index(path)
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
                w.writeheader()
                w.writerows(old)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def new_run_dir(session_dir: Path, model_run: str) -> tuple[str, Path]:
    base = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{model_run}"
    run_id, n = base, 1
    while (session_dir / "runs" / run_id).exists():
        n += 1
        run_id = f"{base}-{n}"
    d = session_dir / "runs" / run_id
    d.mkdir(parents=True)
    return run_id, d


def score_session(model, model_run: str, api: str, session: str, clips: list[dict], out: Path,
                  label: str, note: str, normalize: bool = True) -> dict | None:
    sdir = out / session
    (sdir / "audio").mkdir(parents=True, exist_ok=True)
    sr = clips[0]["sample_rate"]
    started = min(m["created_at"] for m in clips)
    print(f"\nsession {session} (started {started}): {len(clips)} clips, "
          f"{sum(m['duration_s'] for m in clips):.0f} s at {sr} Hz")
    if sr < DETECTOR_SR:
        print(f"WARNING: recorded at {sr} Hz. The detector looks up to 16 kHz, but this audio stops at "
              f"{sr // 2} Hz, so the treble band is empty and scores are out of distribution. "
              f"Record at 32 kHz (the app default now) for a meaningful result.")

    audio, clip_results = [], []
    for m in clips:
        path = download(f"{api}/audio/{m['id']}/stream", sdir / "audio" / f"clip_{m['clip_index']:03d}.wav")
        x, _ = sf.read(str(path), dtype="float64", always_2d=True)
        audio.append(x.mean(axis=1))
        res = score_file(model, path, normalize)
        entry = {"clip_index": m["clip_index"], "clip_id": m["id"], "duration_s": m["duration_s"]}
        if "error" in res:
            print(f"  clip {m['clip_index']:3d}: {res['error']}")
            clip_results.append(entry | {"error": res["error"]})
            continue
        bands = " ".join(f"{k}={v:.2f}" for k, v in res["band_scores"].items())
        print(f"  clip {m['clip_index']:3d} ({m['duration_s']:.0f} s): P(AI)={res['p_ai']:.3f}  [{bands}]")
        clip_results.append(entry | {"p_ai": res["p_ai"], "band_scores": res["band_scores"]})

    whole = sdir / "audio" / "session.wav"
    if not whole.exists():
        sf.write(str(whole), np.concatenate(audio), sr, subtype="PCM_16")
    res = score_file(model, whole, normalize)
    if "error" in res:
        print(f"  session: {res['error']} (not logged)")
        return None

    run_id, rdir = new_run_dir(sdir, model_run)
    res |= {"run_id": run_id, "session_id": session, "api": api, "model_run": model_run,
            "feature_spec_hash": features.spec_hash(), "label": label, "note": note}
    plot(res, rdir / "session.png")
    (rdir / "session.json").write_text(json.dumps({k: v for k, v in res.items() if not k.startswith("_")}, indent=1))
    (rdir / "clips.json").write_text(json.dumps(clip_results, indent=1))
    row = {
        "run_id": run_id, "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_id": session, "session_started_at": started,
        "duration_s": round(sum(m["duration_s"] for m in clips), 1), "sample_rate": sr,
        "n_clips": len(clips), "model_run": model_run, "p_ai": res["p_ai"],
        **{f"band_{k}": v for k, v in res["band_scores"].items()},
        "max_window_p_ai": max(w["p_ai"] for w in res["windows"]),
        "label": label, "note": note, "run_dir": str(rdir.relative_to(out)),
        "input_rms_dbfs": res["input_rms_dbfs"], "level_normalized_to_dbfs": res["level_normalized_to_dbfs"],
    }
    append_index(out / "index.csv", row)
    bands = " ".join(f"{k}={v:.2f}" for k, v in res["band_scores"].items())
    print(f"  session P(AI)={res['p_ai']:.3f}  [{bands}]  run {run_id}\n  -> {rdir}/session.png")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--api", default="http://localhost:9000", help="backend base URL")
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("--session", help="live session id (default: most recent)")
    pick.add_argument("--all-new", action="store_true",
                      help="score every session this model hasn't scored yet (per index.csv)")
    ap.add_argument("--label", default="", help="ground truth for this recording: human | ai | ai_assisted")
    ap.add_argument("--note", default="", help="free text, e.g. 'phone speaker -> mic, song title'")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-normalize", action="store_true", help="score at the recording's own level")
    args = ap.parse_args()
    if args.label and args.label not in ("human", "ai", "ai_assisted"):
        raise SystemExit("--label must be 'human', 'ai' or 'ai_assisted'")
    if args.all_new and (args.label or args.note):
        raise SystemExit("--label/--note describe one recording; use them with --session or the default latest")
    api = args.api.rstrip("/")
    args.out.mkdir(parents=True, exist_ok=True)
    model_run = args.model.resolve().parent.name

    clips = [m for m in get_json(f"{api}/audio") if m.get("session_id")]
    if not clips:
        raise SystemExit(f"no live clips on {api}; record something in the app first")
    by_session: dict[str, list[dict]] = {}
    for m in clips:
        by_session.setdefault(m["session_id"], []).append(m)
    for v in by_session.values():
        v.sort(key=lambda m: m["clip_index"])
    started = lambda s: min(m["created_at"] for m in by_session[s])  # noqa: E731

    if args.all_new:
        norm = "" if args.no_normalize else str(TARGET_RMS_DBFS)
        done = {(r["session_id"], r["model_run"]) for r in read_index(args.out / "index.csv")
                if (r.get("level_normalized_to_dbfs") or "") == norm}
        sessions = sorted((s for s in by_session if (s, model_run) not in done), key=started)
        if not sessions:
            print(f"all {len(by_session)} sessions already scored with {model_run}")
            return
    else:
        sessions = [args.session or max(by_session, key=started)]
        if sessions[0] not in by_session:
            raise SystemExit(f"session {sessions[0]} has no clips on {api}")

    model = load_model(args.model)
    for s in sessions:
        score_session(model, model_run, api, s, by_session[s], args.out, args.label, args.note,
                      normalize=not args.no_normalize)
    print(f"\nresults log: {args.out / 'index.csv'}")


if __name__ == "__main__":
    main()
