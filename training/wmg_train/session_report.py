"""Score the recordings the macOS app saved with the installed models and compare with their labels.

    python -m wmg_train.session_report [--history DIR] [--out reports/sessions/<ts>]
        [--model-10s runs/<ts>/model.pt] [--model-5s runs/<ts>/model.pt]

Every saved session (session.json + audio.wav under the app's history folder) is scored the way
the app scores it: level-normalised, 10 s windows at a 5 s hop, the 5 s model on each half of a
10 s window, each model's score = sigmoid of its mean window logit, headline = the average of
the two probabilities. Labels come from the app: one for the whole recording and/or one per
30 s clip. The report says, per session and per labelled clip, what the models answered, and
sums it up over all human and AI clips (AUC, AI caught, human flagged at 0.5). Sessions whose
audio was in the installed models' training data are marked, since their numbers are not a
test. The app's own on-device scores saved in session.json are compared with this re-scoring
where the same model produced them.

Writes sessions.csv, clips.csv, windows.csv and summary.md under --out. Audio is never copied.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from . import features, labels
from .retrain import APP_HISTORY, RUNS, installed_run
from .score import MIN_INPUT_DBFS, load_model, normalize_level

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "sessions"
CLIP_S = 30.0
BATCH = 64


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=np.float64)))


def logit_of(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def training_sessions(run: str) -> set[str]:
    """Phone capture sessions whose segments were in a model run's training data."""
    p = RUNS / run / "oof_songs.csv"
    if not p.exists():
        return set()
    return set(re.findall(r"phone_captures/([0-9a-f-]{36})/", p.read_text()))


def window_logits(model, mel: np.ndarray, halves: bool) -> np.ndarray:
    """Mean-free per-window logits; with `halves`, each 10 s window gives two 5 s logits."""
    if halves:
        mel = np.concatenate([mel[:, :, : mel.shape[2] // 2], mel[:, :, mel.shape[2] // 2 :]])
    out = []
    with torch.no_grad():
        for i in range(0, len(mel), BATCH):
            out.append(model(torch.from_numpy(mel[i : i + BATCH])[:, None])["ai_logit"].numpy())
    return np.concatenate(out)


def score_audio(path: Path, m10, m5) -> tuple[pd.DataFrame, dict]:
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    mono = x.mean(axis=1)
    in_db = 20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-12)
    info = {"input_rms_dbfs": round(float(in_db), 1), "audio_s": round(len(mono) / sr, 1)}
    if in_db < MIN_INPUT_DBFS:
        return pd.DataFrame(), info
    wins = features.extract_windows(normalize_level(mono), sr)
    if not wins:
        return pd.DataFrame(), info
    mel = np.stack([w[2] for w in wins]).astype(np.float32)
    l10 = window_logits(m10, mel, halves=False)
    l5 = window_logits(m5, mel, halves=True)
    n = len(wins)
    rows = [{"start_s": w[0], "window_rms_dbfs": w[1], "logit_10s": float(a), "logit_5s_a": float(b),
             "logit_5s_b": float(c)} for w, a, b, c in zip(wins, l10, l5[:n], l5[n:])]
    return pd.DataFrame(rows), info


def session_labels(s: dict) -> tuple[str | None, dict[int, str]]:
    whole = (s.get("label") or {}).get("category")
    clips = {c["index"]: (c.get("label") or {}).get("category") for c in s.get("clips") or []}
    return whole, {i: c for i, c in clips.items() if c in labels.CATEGORIES}


def agg(w: pd.DataFrame) -> dict:
    """The app's rule: sigmoid of the mean window logit per model, headline = mean of the two."""
    if w.empty:
        return {"n_windows": 0, "p_10s": np.nan, "p_5s": np.nan, "p": np.nan, "ai_windows_10s": np.nan}
    l10 = w.logit_10s.mean()
    l5 = pd.concat([w.logit_5s_a, w.logit_5s_b]).mean()
    p10, p5 = float(sigmoid(l10)), float(sigmoid(l5))
    return {"n_windows": len(w), "p_10s": round(p10, 4), "p_5s": round(p5, 4), "p": round((p10 + p5) / 2, 4),
            "ai_windows_10s": round(float((w.logit_10s > 0).mean()), 3)}


def on_device(s: dict, w: pd.DataFrame, run10: str, run5: str) -> dict:
    """What the app itself computed while recording, and how far it is from this re-scoring."""
    od = s.get("on_device") or {}
    out = {"od_model_10s": od.get("model_run"), "od_p_10s": np.nan, "od_p_5s": np.nan, "od_logit_diff_10s": np.nan}
    w10, w5 = od.get("windows") or [], od.get("windows_5s") or []
    if w10:
        out["od_p_10s"] = round(float(sigmoid(np.mean([x["logit"] for x in w10]))), 4)
    if w5:
        out["od_p_5s"] = round(float(sigmoid(np.mean([x["logit"] for x in w5]))), 4)
    if w10 and od.get("model_run") == run10 and not w.empty:
        mine = dict(zip(w.start_s.round(1), w.logit_10s))
        d = [abs(x["logit"] - mine[round(x["start_s"], 1)]) for x in w10 if round(x["start_s"], 1) in mine]
        if d:
            out["od_logit_diff_10s"] = round(float(np.mean(d)), 3)
    return out


def verdict(p: float) -> str:
    return "ai" if p > 0.5 else "human"


def correct(label: str | None, p: float) -> bool | None:
    if label not in labels.BINARY or not np.isfinite(p):
        return None
    return verdict(p) == label


def binary_metrics(df: pd.DataFrame, col: str) -> dict:
    from sklearn.metrics import roc_auc_score

    d = df[df.label.isin(labels.BINARY) & df[col].notna()]
    ai, hu = d[d.label == "ai"], d[d.label == "human"]
    out = {"n_ai": len(ai), "n_human": len(hu),
           "ai_caught": f"{int((ai[col] > 0.5).sum())}/{len(ai)}" if len(ai) else "-",
           "human_flagged": f"{int((hu[col] > 0.5).sum())}/{len(hu)}" if len(hu) else "-"}
    out["auc"] = round(roc_auc_score((d.label == "ai").astype(int), d[col]), 3) if len(ai) and len(hu) else None
    return out


def fmt(p) -> str:
    return "-" if p is None or (isinstance(p, float) and not np.isfinite(p)) else f"{p:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=Path, default=APP_HISTORY, help="the macOS app's saved sessions")
    ap.add_argument("--model-10s", type=Path, help="default: the installed 10 s model")
    ap.add_argument("--model-5s", type=Path, help="default: the installed 5 s model")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    run10 = args.model_10s.resolve().parent.name if args.model_10s else installed_run("10s")
    run5 = args.model_5s.resolve().parent.name if args.model_5s else installed_run("5s")
    m10 = load_model(args.model_10s or RUNS / run10 / "model.pt")
    m5 = load_model(args.model_5s or RUNS / run5 / "model.pt")
    trained_on = training_sessions(run10) | training_sessions(run5)
    out = args.out or OUT / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    sessions, clips, windows = [], [], []
    for f in sorted(args.history.glob("*/session.json")):
        s = json.loads(f.read_text())
        wav = f.parent / "audio.wav"
        if not wav.exists():
            continue
        sid = s.get("server_session_id")
        whole, clip_labels = session_labels(s)
        print(f"{s['id']}  {s['duration_s']:.0f} s  label={whole}  clip labels={len(clip_labels)}", flush=True)
        w, info = score_audio(wav, m10, m5)
        # Effective label per window: the clip's own label, else the whole recording's.
        clip_of = {}
        for c in s.get("clips") or []:
            for st in w.start_s if not w.empty else []:
                if c["start_s"] <= st < c["start_s"] + c["duration_s"]:
                    clip_of[st] = c["index"]
        if not w.empty:
            w["clip_index"] = [clip_of.get(st) for st in w.start_s]
            w["label"] = [clip_labels.get(ci, whole) if ci is not None else whole for ci in w.clip_index]
            w.insert(0, "session", s["id"])
            windows.append(w)
        a = agg(w)
        row = {"session": s["id"], "server_session_id": sid, "started_at": s.get("started_at"),
               "duration_s": round(s["duration_s"], 1), **info, "label": whole,
               "clip_labels": json.dumps(dict(sorted(pd.Series(list(clip_labels.values())).value_counts().items())))
               if clip_labels else "", "in_training": sid in trained_on, **a,
               "verdict": verdict(a["p"]) if a["n_windows"] else "no windows",
               "correct": correct(whole, a["p"]), **on_device(s, w, run10, run5)}
        sessions.append(row)
        for c in s.get("clips") or []:
            lab = clip_labels.get(c["index"], whole)
            if lab is None:
                continue
            cw = w[w.clip_index == c["index"]] if not w.empty else w
            ca = agg(cw)
            clips.append({"session": s["id"], "server_session_id": sid, "clip_index": c["index"],
                          "start_s": c["start_s"], "duration_s": round(c["duration_s"], 1), "label": lab,
                          "label_from": "clip" if c["index"] in clip_labels else "recording",
                          "in_training": sid in trained_on, **ca, "correct": correct(lab, ca["p"])})

    S, C = pd.DataFrame(sessions), pd.DataFrame(clips)
    W = pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()
    S.to_csv(out / "sessions.csv", index=False)
    C.to_csv(out / "clips.csv", index=False)
    W.to_csv(out / "windows.csv", index=False)
    (out / "summary.md").write_text(summary_md(S, C, W, run10, run5))
    print(f"\nwrote {out / 'summary.md'}")
    print((out / "summary.md").read_text())


def summary_md(S: pd.DataFrame, C: pd.DataFrame, W: pd.DataFrame, run10: str, run5: str) -> str:
    L = [f"# App recordings scored with the installed models", "",
         f"Models: 10 s `{run10}`, 5 s `{run5}`. {len(S)} saved sessions, "
         f"{int(S.label.notna().sum())} labelled as a whole, {int((S.clip_labels != '').sum())} with clip labels; "
         f"{int(S.in_training.sum())} of them were in the models' training data.", "",
         "Scores follow the app: P = mean of the two models' probabilities, each the sigmoid of its mean "
         "window logit over the whole recording. `AI win` is the share of 10 s windows the 10 s model "
         "scores above 0.5. `on-device` is what the app computed while recording (blank where an older "
         "model did), and `Δlogit` the mean |difference| between the app's 10 s window logits and this "
         "re-scoring when the same model made both.", ""]

    def tbl(df: pd.DataFrame, cols: list[tuple[str, str]]) -> list[str]:
        head = "| " + " | ".join(h for h, _ in cols) + " |"
        sep = "|" + "---|" * len(cols)
        rows = ["| " + " | ".join(str(fn(r)) for _, fn in cols) + " |" for r in df.itertuples()]
        return [head, sep, *rows, ""]

    def ok(r):
        return "" if r.correct is None or pd.isna(r.correct) else ("yes" if r.correct else "**no**")

    L += ["## Sessions", ""]
    L += tbl(S.sort_values("started_at"), [
        ("session", lambda r: r.session), ("dur", lambda r: f"{r.duration_s:.0f} s"),
        ("label", lambda r: r.label or ""), ("clip labels", lambda r: r.clip_labels),
        ("win", lambda r: r.n_windows), ("P10", lambda r: fmt(r.p_10s)), ("P5", lambda r: fmt(r.p_5s)),
        ("P", lambda r: fmt(r.p)), ("AI win", lambda r: fmt(r.ai_windows_10s)),
        ("verdict", lambda r: r.verdict), ("correct", ok),
        ("on-device P10/P5", lambda r: f"{fmt(r.od_p_10s)}/{fmt(r.od_p_5s)}" if r.od_model_10s == run10 else ""),
        ("Δlogit", lambda r: fmt(r.od_logit_diff_10s)), ("in training", lambda r: "yes" if r.in_training else ""),
    ])

    if not C.empty:
        L += ["## Labelled clips (30 s)", "",
              "Clips take their own label where the app has one, else the recording's. Metrics use the "
              "headline P per clip; `window AUC` uses single 10 s windows of the 10 s model.", ""]
        test = C[~C.in_training]
        for name, d in (("all labelled clips", C), ("clips not in training", test)):
            if d.empty or d.label.isin(labels.BINARY).sum() == 0:
                continue
            m = {k: binary_metrics(d, k) for k in ("p", "p_10s", "p_5s")}
            L += [f"### {name}: {len(d)} clips ({', '.join(f'{v} {k}' for k, v in d.label.value_counts().items())})", "",
                  "| score | AUC | AI caught @0.5 | human flagged @0.5 |", "|---|---|---|---|"]
            for k, lab in (("p", "headline"), ("p_10s", "10 s model"), ("p_5s", "5 s model")):
                L.append(f"| {lab} | {m[k]['auc']} | {m[k]['ai_caught']} | {m[k]['human_flagged']} |")
            if not W.empty:
                wd = W[W.session.isin(d.session.unique()) & W.label.isin(labels.BINARY)]
                if wd.label.nunique() == 2:
                    wm = binary_metrics(wd.assign(p=sigmoid(wd.logit_10s)), "p")
                    L.append(f"| window (10 s) | {wm['auc']} | {wm['ai_caught']} | {wm['human_flagged']} |")
            L.append("")
            asst = d[d.label == "ai_assisted"]
            if len(asst):
                L += [f"AI-assisted clips ({len(asst)}): median P {asst.p.median():.2f}, "
                      f"{int((asst.p > 0.5).sum())} of {len(asst)} called AI.", ""]
        per = C[C.label.isin(labels.BINARY)].groupby(["session", "label"], as_index=False).agg(
            n=("p", "size"), right=("correct", "sum"), p_med=("p", "median"))
        if len(per):
            L += ["### Per session and label", "", "| session | label | clips | correct | median P |", "|---|---|---|---|---|"]
            L += [f"| {r.session} | {r.label} | {r.n} | {int(r.right)} | {r.p_med:.2f} |" for r in per.itertuples()]
            L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
