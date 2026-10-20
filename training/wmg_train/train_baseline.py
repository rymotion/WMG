"""Baseline BandNet training on the augmented cache, with leave-one-AI-artist-out CV.

    python -m wmg_train.train_baseline --cache cache --out runs

Each fold holds out one AI artist group plus a share of the human artist groups, so the score
measures detection of an unseen AI artist rather than recognition of known ones. After CV, one
model is trained on everything and exported (model.pt, model.onnx, metrics.json).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .device import pick_device
from .models.student import BandNet, count_params, export_onnx

CFG = Path(__file__).resolve().parents[1] / "configs" / "student.yaml"
BAND_AUX_WEIGHT = 0.3


def load_cache(cache: Path) -> tuple[np.memmap, pd.DataFrame]:
    index = pd.read_parquet(cache / "index.parquet")
    x = np.memmap(cache / "windows.f16", dtype=np.float16, mode="r", shape=(len(index), 128, 1000))
    return x, index


def make_folds(index: pd.DataFrame, seed: int = 0) -> list[tuple[str, set[str]]]:
    """One fold per AI group; human groups spread greedily so folds get similar song counts."""
    songs = index.drop_duplicates("song_id")
    ai_groups = sorted(songs.loc[songs.label == 1, "group"].unique())
    sizes = songs.loc[songs.label == 0].groupby("group").size()
    rng = np.random.default_rng(seed)
    order = sorted(sizes.index, key=lambda g: (-sizes[g], rng.random()))
    buckets: list[set[str]] = [set() for _ in ai_groups]
    load = np.zeros(len(ai_groups))
    for g in order:
        k = int(load.argmin())
        buckets[k].add(g)
        load[k] += sizes[g]
    return [(ai, {ai} | hum) for ai, hum in zip(ai_groups, buckets)]


def spec_augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Two frequency masks (≤12 bins) and two time masks (≤100 frames), filled with the floor."""
    b, _, f, t = x.shape
    x = x.clone()
    floor = x.amin(dim=(2, 3), keepdim=True)
    for size, axis in ((12, 2), (12, 2), (100, 3), (100, 3)):
        n = f if axis == 2 else t
        w = torch.randint(0, size + 1, (b,), generator=gen)
        s = (torch.rand(b, generator=gen) * (n - w)).long()
        ar = torch.arange(n)
        mask = (ar[None] >= s[:, None]) & (ar[None] < (s + w)[:, None])  # [b, n]
        mask = mask.to(x.device)
        mask = mask[:, None, :, None] if axis == 2 else mask[:, None, None, :]
        x = torch.where(mask, floor, x)
    return x


def batches_balanced(labels: np.ndarray, batch: int, steps: int, rng: np.random.Generator):
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    for _ in range(steps):
        idx = np.concatenate([rng.choice(pos, batch // 2), rng.choice(neg, batch - batch // 2)])
        yield np.sort(idx)


def train_model(x: np.memmap, index: pd.DataFrame, rows: np.ndarray, args, device, seed: int) -> BandNet:
    torch.manual_seed(seed)
    cfg = yaml.safe_load(CFG.read_text())
    model = BandNet.from_config(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    labels = index.label.to_numpy()[rows]
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    t0 = time.time()
    for ep in range(args.epochs):
        losses = []
        for idx in batches_balanced(labels, args.batch, args.steps, rng):
            xb = torch.from_numpy(np.asarray(x[rows[idx]], dtype=np.float32))[:, None]
            yb = torch.from_numpy(labels[idx].astype(np.float32))
            xb = spec_augment(xb, gen).to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = F.binary_cross_entropy_with_logits(out["ai_logit"], yb)
            band = out["band_scores"].clamp(1e-6, 1 - 1e-6)
            loss = loss + BAND_AUX_WEIGHT * F.binary_cross_entropy(band, yb[:, None].expand_as(band))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        print(f"    epoch {ep + 1}/{args.epochs} loss {np.mean(losses):.4f} ({time.time() - t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def predict(model: BandNet, x: np.memmap, rows: np.ndarray, device, batch: int = 64) -> dict[str, np.ndarray]:
    model.eval()
    logits, bands = [], []
    for i in range(0, len(rows), batch):
        xb = torch.from_numpy(np.asarray(x[rows[i : i + batch]], dtype=np.float32))[:, None].to(device)
        out = model(xb)
        logits.append(out["ai_logit"].float().cpu().numpy())
        bands.append(out["band_scores"].float().cpu().numpy())
    return {"logit": np.concatenate(logits), "bands": np.concatenate(bands)}


def song_table(index: pd.DataFrame, rows: np.ndarray, pred: dict[str, np.ndarray]) -> pd.DataFrame:
    df = index.iloc[rows][["song_id", "path", "label", "group"]].copy()
    df["logit"] = pred["logit"]
    for i, b in enumerate(("bass", "mid", "treble")):
        df[f"band_{b}"] = pred["bands"][:, i]
    agg = df.groupby(["song_id", "path", "label", "group"], as_index=False).agg(
        logit=("logit", "mean"), n_windows=("logit", "size"),
        band_bass=("band_bass", "mean"), band_mid=("band_mid", "mean"), band_treble=("band_treble", "mean"))
    agg["p_ai"] = 1 / (1 + np.exp(-agg.logit))
    return agg


def confound_gate(index: pd.DataFrame) -> float:
    """Spec gate on the *augmented* windows: recording-chain metadata alone vs label."""
    songs = index.drop_duplicates(["song_id", "variant"])
    cols = [songs.bitrate_kbps, songs.codec.eq("mp3"), songs.orig_sr, songs.gain_db,
            songs.lowpass_hz.notna(), songs.noise_snr_db.notna()]
    if "speaker" in songs:
        cols.append(songs.speaker.notna())
    X = np.c_[tuple(cols)].astype(float)
    y, groups = songs.label.to_numpy(), songs.group.to_numpy()
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LogisticRegression(class_weight="balanced").fit((X[tr] - mu) / sd, y[tr])
        scores[te] = clf.decision_function((X[te] - mu) / sd)
    return float(roc_auc_score(y, scores))


def calibration(songs: pd.DataFrame, bins: int = 5) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (songs.p_ai >= a) & (songs.p_ai < b if b < 1 else songs.p_ai <= b)
        if m.any():
            out.append({"bin": f"{a:.1f}-{b:.1f}", "n": int(m.sum()),
                        "mean_p": round(float(songs.p_ai[m].mean()), 3),
                        "frac_ai": round(float(songs.label[m].mean()), 3)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=250, help="balanced batches per epoch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--folds", type=str, default="all", help="'all', 'none', or comma list of AI groups")
    args = ap.parse_args()

    device = pick_device()
    x, index = load_cache(args.cache)
    run = args.out / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    cache_meta = json.loads((args.cache / "meta.json").read_text())
    print(f"device {device}; {len(index)} windows, {index.song_id.nunique()} songs; run {run}", flush=True)

    gate = confound_gate(index)
    print(f"confound gate (augmented codec/bitrate/sr/gain metadata) AUC = {gate:.3f} "
          f"({'PASS' if gate <= 0.55 else 'FAIL'})", flush=True)

    metrics: dict = {"args": vars(args) | {"cache": str(args.cache), "out": str(args.out)},
                     "device": str(device), "confound_gate_auc": gate,
                     "feature_spec_hash": cache_meta["feature_spec_hash"],
                     "feature_crate_commit": cache_meta["feature_crate_commit"], "folds": []}
    folds = make_folds(index)
    if args.folds != "all":
        keep = set() if args.folds == "none" else set(args.folds.split(","))
        folds = [f for f in folds if f[0] in keep]

    oof = []
    for k, (ai_group, test_groups) in enumerate(folds):
        test = index.group.isin(test_groups).to_numpy()
        tr_rows, te_rows = np.flatnonzero(~test), np.flatnonzero(test)
        print(f"fold {k + 1}/{len(folds)}: hold out {ai_group} + {len(test_groups) - 1} human groups "
              f"({len(te_rows)} test windows)", flush=True)
        model = train_model(x, index, tr_rows, args, device, seed=k)
        pred = predict(model, x, te_rows, device)
        songs = song_table(index, te_rows, pred)
        songs["fold"] = k
        oof.append(songs)
        win_auc = roc_auc_score(index.label.to_numpy()[te_rows], pred["logit"])
        song_auc = roc_auc_score(songs.label, songs.logit)
        ai = songs[songs.label == 1]
        hum = songs[songs.label == 0]
        fm = {"held_out_ai": ai_group, "window_auc": round(win_auc, 4), "song_auc": round(song_auc, 4),
              "ai_recall@0.5": round(float((ai.p_ai > 0.5).mean()), 3),
              "human_fpr@0.5": round(float((hum.p_ai > 0.5).mean()), 3),
              "n_ai_songs": len(ai), "n_human_songs": len(hum)}
        if "speaker" in index:
            # Window AUC on clean vs simulated phone-speaker -> mic windows.
            spk = index.speaker.notna().to_numpy()[te_rows]
            y_te = index.label.to_numpy()[te_rows]
            for name, m in (("clean", ~spk), ("speaker", spk)):
                if len(np.unique(y_te[m])) == 2:
                    fm[f"window_auc_{name}"] = round(roc_auc_score(y_te[m], pred["logit"][m]), 4)
        metrics["folds"].append(fm)
        print("   ", fm, flush=True)

    if oof:
        allsongs = pd.concat(oof)
        allsongs.to_csv(run / "oof_songs.csv", index=False)
        hum = allsongs[allsongs.label == 0]
        metrics["oof"] = {
            "song_auc": round(roc_auc_score(allsongs.label, allsongs.logit), 4),
            "ai_recall@0.5": round(float((allsongs[allsongs.label == 1].p_ai > 0.5).mean()), 3),
            "human_fpr@0.5": round(float((hum.p_ai > 0.5).mean()), 3),
            "ai_recall_by_group": allsongs[allsongs.label == 1].groupby("group").p_ai
                .apply(lambda p: round(float((p > 0.5).mean()), 3)).to_dict(),
            "human_fpr_by_group": hum.groupby("group").p_ai
                .apply(lambda p: round(float((p > 0.5).mean()), 3)).to_dict(),
            "calibration": calibration(allsongs),
        }
        print("OOF:", json.dumps(metrics["oof"], indent=1), flush=True)

    print("final model on all data", flush=True)
    model = train_model(x, index, np.arange(len(index)), args, device, seed=100)
    model = model.cpu().eval()
    torch.save({"state_dict": model.state_dict(), "config": yaml.safe_load(CFG.read_text()),
                "feature_spec_hash": cache_meta["feature_spec_hash"]}, run / "model.pt")
    export_onnx(model, run / "model.onnx")
    metrics["params"] = count_params(model)
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved {run}/model.pt, model.onnx, metrics.json")


if __name__ == "__main__":
    main()
