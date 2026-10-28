"""Test bench: compare detector variants (e.g. 10 s vs 5 s windows) on identical held-out data.

    python -m wmg_train.bench --run 10s=runs/<cv run> --run 5s=runs/<cv run> \\
        [--coreml 10s=runs/<model run> --coreml 5s=runs/<model run>]

Each --run is a cross-validation run made with --save-oof-windows: every window was scored by
a model that never trained on that artist or recording, and both runs use the same folds and
seeds. --coreml points at a run with a Core ML export, for on-device speed.

Writes reports/bench/<timestamp>/report.md, results.json and listening_time.png:
  * accuracy: window- and item-level AUC, AI caught / human flagged at 0.5, for clean files and
    speaker->mic captures separately, and per long recording;
  * listening time: on 30 s speaker->mic clips, accuracy after the first 5/10/15/20/30 s of audio
    (the 10 s detector can't answer before 10 s);
  * speed: Core ML milliseconds per window on this Mac, feature extraction per window, and time
    to first result (window length + compute).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "reports" / "bench"
LISTEN_S = (5, 10, 15, 20, 30)


def _kv(items: list[str]) -> dict[str, Path]:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = Path(v) if Path(v).is_absolute() else ROOT / v
    return out


def _auc(y, s) -> float | None:
    return round(float(roc_auc_score(y, s)), 4) if len(set(y)) == 2 else None


def _rates(items: pd.DataFrame) -> dict:
    ai, hu = items[items.label == 1], items[items.label == 0]
    p = 1 / (1 + np.exp(-items.logit))
    return {
        "auc": _auc(items.label, items.logit),
        "ai_caught": round(float((p[items.label == 1] > 0.5).mean()), 4) if len(ai) else None,
        "human_flagged": round(float((p[items.label == 0] > 0.5).mean()), 4) if len(hu) else None,
        "n_ai": int(len(ai)), "n_human": int(len(hu)),
    }


def accuracy(w: pd.DataFrame) -> dict:
    """Window and item (song / 30 s clip) metrics, split clean vs speaker->mic."""
    w = w.assign(phone=w.path.str.contains("phone_captures"))
    items = w.groupby(["song_id", "path", "label", "group", "phone"], as_index=False).logit.mean()
    out = {}
    for name, m_w, m_i in (("all", slice(None), slice(None)),
                           ("clean", ~w.phone, ~items.phone),
                           ("speaker_mic", w.phone, items.phone)):
        ww, ii = w[m_w], items[m_i]
        out[name] = {"window_auc": _auc(ww.label, ww.logit), "windows": int(len(ww)), "items": _rates(ii)}
    rec = items[items.phone].groupby("group").filter(lambda x: len(x) >= 20)
    out["recordings"] = {
        g: {"label": "ai" if x.label.iloc[0] else "human", "clips": int(len(x)),
            "flagged_as_ai": round(float((1 / (1 + np.exp(-x.logit)) > 0.5).mean()), 4)}
        for g, x in rec.groupby("group")
    }
    return out


def listening_time(w: pd.DataFrame, window_s: float) -> dict:
    """Accuracy on speaker->mic clips using only windows that end within the first t seconds."""
    ph = w[w.path.str.contains("phone_captures") & (w.variant == 0)]
    out = {}
    for t in LISTEN_S:
        sub = ph[ph.start_s + window_s <= t + 1e-6]
        if sub.empty:
            out[t] = None
            continue
        items = sub.groupby(["song_id", "label"], as_index=False).logit.mean()
        r = _rates(items)
        p = 1 / (1 + np.exp(-items.logit))
        r["accuracy"] = round(float(((p > 0.5).astype(int) == items.label).mean()), 4)
        out[t] = r
    return out


def speed(run: Path, n: int = 50) -> dict:
    """Core ML latency per window (batch 1, all compute units) + Rust feature time per window."""
    import coremltools as ct
    import wmg_features as wf

    meta = json.loads((run / "coreml" / "model_meta.json").read_text())
    frames = int(meta.get("frames", 1000))
    model = ct.models.MLModel(str(run / "coreml" / "WmgDetector.mlpackage"), compute_units=ct.ComputeUnit.ALL)
    x = (np.random.default_rng(0).standard_normal((1, 1, 128, frames)) * 10 - 50).astype(np.float32)
    for _ in range(5):
        model.predict({"logmel": x})
    ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict({"logmel": x})
        ms.append((time.perf_counter() - t0) * 1000)
    mel = wf.DetectorMel()
    audio = np.random.default_rng(1).standard_normal(32000 * frames // 100) * 0.1
    fm = []
    for _ in range(20):
        t0 = time.perf_counter()
        mel.logmel(audio)
        fm.append((time.perf_counter() - t0) * 1000)
    infer, feat = statistics.median(ms), statistics.median(fm)
    window_s = frames / 100
    return {
        "window_seconds": window_s, "coreml_ms_per_window": round(infer, 2), "features_ms_per_window": round(feat, 2),
        "time_to_first_result_s": round(window_s + (infer + feat) / 1000, 3),
        "realtime_factor": round(window_s * 1000 / (infer + feat), 1),
    }


def plot(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, r in results["models"].items():
        lt = r["listening_time"]
        ts = [t for t in LISTEN_S if lt.get(t) or lt.get(str(t))]
        get = lambda t, k: (lt.get(t) or lt.get(str(t)))[k]  # noqa: E731
        ax[0].plot(ts, [get(t, "auc") for t in ts], "o-", label=name)
        ax[1].plot(ts, [get(t, "accuracy") for t in ts], "o-", label=name)
    for a, title in zip(ax, ("AUC", "accuracy at 0.5")):
        a.set_xlabel("seconds of audio heard")
        a.set_title(f"Speaker->mic clips: {title}")
        a.set_ylim(0.4, 1.0)
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def report_md(results: dict) -> str:
    names = list(results["models"])
    pct = lambda v: "n/a" if v is None else f"{v * 100:.0f}%"  # noqa: E731
    num = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    L = [f"# Detector test bench ({results['created_at'][:16]} UTC)", "",
         "All numbers are held-out: each window was scored by a model that never trained on that artist "
         "or recording. Both models use identical folds, seeds and training settings.", ""]
    L += ["## Accuracy", "", "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for split, label in (("clean", "Clean files"), ("speaker_mic", "Speaker → mic")):
        for k, lab in (("auc", "item AUC"), ("ai_caught", "AI caught @0.5"), ("human_flagged", "human flagged @0.5")):
            f = num if k == "auc" else pct
            L.append(f"| {label}: {lab} | " + " | ".join(f(results['models'][n]['accuracy'][split]['items'][k]) for n in names) + " |")
        L.append(f"| {label}: window AUC | " + " | ".join(num(results['models'][n]['accuracy'][split]['window_auc']) for n in names) + " |")
    L += ["", "### Long speaker → mic recordings (share flagged as AI)", "",
          "| recording | truth | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    recs = sorted(set().union(*[results["models"][n]["accuracy"]["recordings"] for n in names]))
    for g in recs:
        rr = [results["models"][n]["accuracy"]["recordings"].get(g) for n in names]
        truth = next(r["label"] for r in rr if r)
        L.append(f"| {g} | {truth} | " + " | ".join(pct(r['flagged_as_ai']) if r else "n/a" for r in rr) + " |")
    L += ["", "## Listening time (30 s speaker → mic clips)", "",
          "| seconds heard | " + " | ".join(f"{n} AUC | {n} accuracy" for n in names) + " |",
          "|---|" + "---|---|" * len(names)]
    for t in LISTEN_S:
        cells = []
        for n in names:
            r = results["models"][n]["listening_time"].get(t)
            cells += [num(r["auc"]) if r else "—", pct(r["accuracy"]) if r else "—"]
        L.append(f"| {t} s | " + " | ".join(cells) + " |")
    if any("speed" in results["models"][n] for n in names):
        L += ["", "## Speed (this Mac, Core ML, batch 1)", "",
              "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for k, lab in (("coreml_ms_per_window", "Core ML ms / window"), ("features_ms_per_window", "features ms / window"),
                       ("time_to_first_result_s", "time to first result (s)"), ("realtime_factor", "× faster than real time")):
            L.append(f"| {lab} | " + " | ".join(str(results['models'][n].get('speed', {}).get(k, 'n/a')) for n in names) + " |")
    L += ["", "![listening time](listening_time.png)", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=cv run dir with oof_windows.csv")
    ap.add_argument("--coreml", action="append", help="name=run dir with coreml/ export (for speed)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    runs, coreml = _kv(args.run), _kv(args.coreml)

    results = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "models": {}}
    for name, run in runs.items():
        w = pd.read_csv(run / "oof_windows.csv")
        m = json.loads((run / "metrics.json").read_text())
        window_s = float(m.get("window_seconds", 10))
        r = {"run": run.name, "window_seconds": window_s, "accuracy": accuracy(w), "listening_time": listening_time(w, window_s)}
        if name in coreml:
            r["speed"] = speed(coreml[name])
        results["models"][name] = r
        print(f"{name}: clean AUC {r['accuracy']['clean']['items']['auc']}, speaker->mic AUC {r['accuracy']['speaker_mic']['items']['auc']}")

    out = args.out / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    plot(results, out / "listening_time.png")
    (out / "report.md").write_text(report_md(results))
    print(f"-> {out}/report.md")


if __name__ == "__main__":
    main()
