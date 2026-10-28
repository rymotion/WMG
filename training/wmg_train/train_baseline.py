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


class Windows:
    """Several window caches (clean + phone) addressed as one array of global row numbers."""

    def __init__(self, parts: list[np.memmap]):
        self.parts = parts
        self.offsets = np.cumsum([0] + [len(p) for p in parts])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, idx) -> np.ndarray:
        idx = np.asarray(idx)
        out = np.empty((len(idx), 128, 1000), dtype=np.float16)
        src = np.searchsorted(self.offsets, idx, side="right") - 1
        for s in np.unique(src):
            m = src == s
            out[m] = self.parts[s][idx[m] - self.offsets[s]]
        return out


def load_all(cache: Path, phone_cache: Path | None) -> tuple[Windows, pd.DataFrame]:
    x, index = load_cache(cache)
    index = index.assign(source="clean")
    parts = [x]
    if phone_cache:
        px, pindex = load_cache(phone_cache)
        index = pd.concat([index, pindex.assign(source="phone")], ignore_index=True)
        parts.append(px)
    index["row"] = np.arange(len(index))
    return Windows(parts), index


class SplitWindows:
    """View 10 s windows as `parts` shorter windows (e.g. 2 x 5 s) without recomputing features.

    The detector's frames are identical whatever the window length (same STFT and hop), so a
    5 s window is a 500-frame slice of a 10 s one; only the last few frames at a cut differ from
    a direct 5 s extraction (real audio instead of zero padding).
    """

    def __init__(self, base, src_rows: np.ndarray, part: np.ndarray, parts: int):
        self.base, self.src, self.part, self.parts = base, src_rows, part, parts
        self.frames = 1000 // parts

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx) -> np.ndarray:
        idx = np.asarray(idx)
        full = self.base[self.src[idx]]
        out = np.empty((len(idx), 128, self.frames), dtype=full.dtype)
        for i, p in enumerate(self.part[idx]):
            out[i] = full[i, :, p * self.frames : (p + 1) * self.frames]
        return out


def split_windows(x, index: pd.DataFrame, parts: int) -> tuple[SplitWindows, pd.DataFrame]:
    """Each window row becomes `parts` rows with start_s shifted. Overlapping 10 s windows (5 s
    hop) yield the same 5 s audio twice; those duplicates are dropped."""
    seg = 10.0 / parts
    rep = index.loc[index.index.repeat(parts)].copy()
    rep["src_row"] = np.repeat(index["row"].to_numpy(), parts)
    rep["part"] = np.tile(np.arange(parts), len(index))
    rep["start_s"] = rep["start_s"] + rep["part"] * seg
    rep = rep.drop_duplicates(["song_id", "variant", "start_s"]).reset_index(drop=True)
    rep["row"] = np.arange(len(rep))
    return SplitWindows(x, rep["src_row"].to_numpy(), rep["part"].to_numpy(), parts), rep


def load_cfg(window_seconds: int) -> dict:
    cfg = yaml.safe_load(CFG.read_text())
    cfg["input"]["frames"] = 1000 * window_seconds // 10
    cfg["input"]["window_seconds"] = window_seconds
    return cfg


def make_folds(index: pd.DataFrame, seed: int = 0) -> list[tuple[str, set[str]]]:
    """One fold per clean AI group; human groups spread greedily so folds get similar song counts.

    Phone segments from a playlist keep their artist group, so they follow that artist's fold.
    Phone-only groups (whole labelled sessions, `phone:*`) are spread over the folds the same way.
    """
    if "source" in index:
        folds = make_folds(index[index.source == "clean"].drop(columns="source"), seed)
        clean_groups = set(index.loc[index.source == "clean", "group"])
        extra = (index[(index.source == "phone") & ~index.group.isin(clean_groups)]
                 .drop_duplicates("song_id").groupby("group").size().sort_values(ascending=False))
        load = np.zeros(len(folds))
        for g, n in extra.items():
            k = int(load.argmin())
            folds[k][1].add(g)
            load[k] += n
        return folds
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


def batches_balanced(labels: np.ndarray, batch: int, steps: int, rng: np.random.Generator,
                     is_phone: np.ndarray | None = None, phone_frac: float = 0.0):
    """Half AI, half human. Within each half, `phone_frac` of the windows come from real phone
    captures of that class when there are any, so scarce phone audio isn't drowned out."""
    is_phone = np.zeros(len(labels), bool) if is_phone is None else is_phone
    halves = ((1, batch // 2), (0, batch - batch // 2))
    pools = {c: (np.flatnonzero((labels == c) & ~is_phone), np.flatnonzero((labels == c) & is_phone))
             for c, _ in halves}
    for _ in range(steps):
        idx = []
        for c, n in halves:
            clean, phone = pools[c]
            k = int(round(phone_frac * n)) if len(phone) else 0
            if not len(clean):
                k = n
            idx += [rng.choice(phone, k), rng.choice(clean, n - k)] if k < n else [rng.choice(phone, k)]
        yield np.sort(np.concatenate(idx))


def train_model(x: np.memmap, index: pd.DataFrame, rows: np.ndarray, args, device, seed: int) -> BandNet:
    torch.manual_seed(seed)
    model = BandNet.from_config(args.cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    labels = index.label.to_numpy()[rows]
    is_phone = (index.source.to_numpy()[rows] == "phone") if "source" in index else None
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    t0 = time.time()
    for ep in range(args.epochs):
        losses = []
        for idx in batches_balanced(labels, args.batch, args.steps, rng, is_phone, args.phone_frac):
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


# Spec: metadata-only AUC <= 0.55. Far *below* 0.5 is as separable as far above, so strong
# inversions fail too; the lower bound is looser because leave-group-out CV on uninformative
# features is itself biased a little below 0.5.
GATE_RANGE = (0.40, 0.55)


def gate_passes(auc: float) -> bool:
    return GATE_RANGE[0] <= auc <= GATE_RANGE[1]


def confound_gate(index: pd.DataFrame, phone_frac: float | None = None) -> float:
    """Spec gate on the *augmented* windows: recording-chain metadata alone vs label.

    With phone audio, rows are weighted the way training samples them (phone = `phone_frac` of
    each class), so the gate measures what the model sees rather than the raw row counts
    (each 30 s phone clip is its own row and the AI class has fewer clean songs).
    """
    songs = index.drop_duplicates(["song_id", "variant"])
    cols = [songs.bitrate_kbps, songs.codec.eq("mp3"), songs.orig_sr, songs.gain_db,
            songs.lowpass_hz.notna(), songs.noise_snr_db.notna()]
    if "speaker" in songs:
        cols.append(songs.speaker.notna())
    if "source" in songs:
        cols.append(songs.source.eq("phone"))  # catches "phone audio = one class"
    X = np.c_[tuple(cols)].astype(float)
    y, groups = songs.label.to_numpy(), songs.group.to_numpy()
    sw = np.ones(len(y))
    if phone_frac is not None and "source" in songs:
        ph = songs.source.eq("phone").to_numpy()
        for c in (0, 1):
            m = y == c
            n_ph, n_cl = (m & ph).sum(), (m & ~ph).sum()
            if n_ph and n_cl:
                sw[m & ph] = phone_frac / n_ph
                sw[m & ~ph] = (1 - phone_frac) / n_cl
            else:
                sw[m] = 1 / m.sum()
    # AUC within each held-out fold, averaged. Pooling scores across folds is biased below 0.5
    # when features are uninformative: each fold's intercept follows its training-set class
    # balance, which is the opposite of its test fold's.
    aucs, sizes = [], []
    for tr, te in GroupKFold(5).split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LogisticRegression().fit((X[tr] - mu) / sd, y[tr], sample_weight=sw[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function((X[te] - mu) / sd), sample_weight=sw[te]))
        sizes.append(len(te))
    return float(np.average(aucs, weights=sizes))


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
    ap.add_argument("--phone-cache", type=Path, help="cache of real phone captures (wmg_train.phone_cache)")
    ap.add_argument("--phone-frac", type=float, default=0.25,
                    help="share of each class's half-batch drawn from phone windows of that class")
    ap.add_argument("--window-seconds", type=int, choices=[10, 5], default=10,
                    help="detector window; 5 s windows are halves of the cached 10 s ones")
    ap.add_argument("--save-oof-windows", action="store_true",
                    help="write per-window held-out scores (oof_windows.csv) for the test bench")
    ap.add_argument("--no-final", action="store_true", help="cross-validation only; skip the final model")
    ap.add_argument("--allow-imbalanced-phone", action="store_true",
                    help="train even if phone audio is >80%% one class (risks learning 'phone = that class')")
    args = ap.parse_args()

    device = pick_device()
    x, index = load_all(args.cache, args.phone_cache)
    args.cfg = load_cfg(args.window_seconds)
    if args.window_seconds != 10:
        x, index = split_windows(x, index, 10 // args.window_seconds)
    phone_stats = None
    if args.phone_cache:
        ph = index[index.source == "phone"]
        share_ai = float(ph.label.mean()) if len(ph) else 0.0
        phone_stats = {"windows": len(ph), "ai_windows": int(ph.label.sum()),
                       "human_windows": int((ph.label == 0).sum()), "ai_share": round(share_ai, 3),
                       "sessions": int(ph.session_id.nunique()), "phone_frac": args.phone_frac}
        print(f"phone audio: {phone_stats}", flush=True)
        if not 0.2 <= share_ai <= 0.8:
            msg = (f"phone windows are {share_ai:.0%} AI / {1 - share_ai:.0%} human. Training on this teaches "
                   f"the model that phone-recorded audio means "
                   f"{'AI' if share_ai > 0.5 else 'human'}; record the other class through the phone too.")
            if not args.allow_imbalanced_phone:
                raise SystemExit("REFUSING: " + msg + " (override: --allow-imbalanced-phone)")
            print("WARNING: " + msg, flush=True)
    run = args.out / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    cache_meta = json.loads((args.cache / "meta.json").read_text())
    print(f"device {device}; {len(index)} windows, {index.song_id.nunique()} songs; run {run}", flush=True)

    # Gate the augmented clean data. Phone audio is checked by the class-balance rule above and
    # sampled at the same share for both classes; its sessions are single-class groups, which
    # makes a grouped-CV gate on it measure fold composition rather than leakage.
    clean = index[index.source == "clean"].drop(columns="source") if "source" in index else index
    gate = confound_gate(clean)
    print(f"confound gate (augmented recording-chain metadata, clean data) AUC = {gate:.3f} "
          f"({'PASS' if gate_passes(gate) else 'FAIL'}: needs {GATE_RANGE[0]}-{GATE_RANGE[1]})", flush=True)

    metrics: dict = {"args": {k: v for k, v in vars(args).items() if k != "cfg"} | {"cache": str(args.cache), "out": str(args.out),
                                           "phone_cache": str(args.phone_cache) if args.phone_cache else None},
                     "phone": phone_stats,
                     "device": str(device), "confound_gate_auc": gate,
                     "feature_spec_hash": cache_meta["feature_spec_hash"],
                     "feature_crate_commit": cache_meta["feature_crate_commit"], "folds": []}
    folds = make_folds(index)
    if args.folds != "all":
        keep = set() if args.folds == "none" else set(args.folds.split(","))
        folds = [f for f in folds if f[0] in keep]

    oof, oof_windows = [], []
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
        if args.save_oof_windows:
            w = index.iloc[te_rows][["song_id", "path", "label", "group", "variant", "start_s"]].copy()
            if "source" in index:
                w["source"] = index.source.to_numpy()[te_rows]
            w["logit"], w["fold"] = pred["logit"], k
            oof_windows.append(w)
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
        if "source" in index and (index.source.to_numpy()[te_rows] == "phone").any():
            ph = index.source.to_numpy()[te_rows] == "phone"
            y_te = index.label.to_numpy()[te_rows]
            ps = song_table(index, te_rows[ph], {k: v[ph] for k, v in pred.items()})
            fm["phone_windows"] = int(ph.sum())
            if len(np.unique(y_te[ph])) == 2:
                fm["phone_window_auc"] = round(roc_auc_score(y_te[ph], pred["logit"][ph]), 4)
            if (ps.label == 1).any():
                fm["phone_ai_recall@0.5"] = round(float((ps[ps.label == 1].p_ai > 0.5).mean()), 3)
            if (ps.label == 0).any():
                fm["phone_human_fpr@0.5"] = round(float((ps[ps.label == 0].p_ai > 0.5).mean()), 3)
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
        phone_paths = set(index.loc[index.source == "phone", "path"]) if "source" in index else set()
        ps = allsongs[allsongs.path.isin(phone_paths)]
        if len(ps):
            # Phone segments/clips scored by models that never saw that artist or session.
            metrics["oof_phone"] = {
                "segments": len(ps), "ai_segments": int(ps.label.sum()), "human_segments": int((ps.label == 0).sum()),
                "segment_auc": round(roc_auc_score(ps.label, ps.logit), 4) if ps.label.nunique() == 2 else None,
                "ai_recall@0.5": round(float((ps[ps.label == 1].p_ai > 0.5).mean()), 3) if (ps.label == 1).any() else None,
                "human_fpr@0.5": round(float((ps[ps.label == 0].p_ai > 0.5).mean()), 3) if (ps.label == 0).any() else None,
            }
        print("OOF:", json.dumps({k: metrics[k] for k in ("oof", "oof_phone") if k in metrics}, indent=1), flush=True)

    if oof_windows:
        pd.concat(oof_windows).to_csv(run / "oof_windows.csv", index=False)
    metrics["window_seconds"] = args.window_seconds
    if args.no_final:
        (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"saved {run}/metrics.json (cross-validation only)")
        return

    print("final model on all data", flush=True)
    model = train_model(x, index, np.arange(len(index)), args, device, seed=100)
    model = model.cpu().eval()
    torch.save({"state_dict": model.state_dict(), "config": args.cfg,
                "feature_spec_hash": cache_meta["feature_spec_hash"]}, run / "model.pt")
    export_onnx(model, run / "model.onnx")
    metrics["params"] = count_params(model)
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved {run}/model.pt, model.onnx, metrics.json")


if __name__ == "__main__":
    main()
