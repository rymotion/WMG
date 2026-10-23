"""Score audio files with a trained BandNet and draw where the evidence came from.

    python -m wmg_train.score runs/<ts>/model.pt song.m4a clip.wav ... [--out scores/]

Per file: song-level P(AI) (mean of window logits), per-window P(AI) and band scores, and a PNG
with the log-mel spectrogram, the band x time AI-evidence overlay, and P(AI) over time.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import features  # noqa: E402
from .augment import TARGET_RMS_DBFS, decode  # noqa: E402
from .models.student import BAND_NAMES, BandNet  # noqa: E402


def load_model(path: Path) -> BandNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("feature_spec_hash") != features.spec_hash():
        raise SystemExit(f"model trained on feature spec {ckpt.get('feature_spec_hash')}, "
                         f"installed wmg_features is {features.spec_hash()}")
    model = BandNet.from_config(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


# Same rule as wmg_features::MIN_INPUT_DBFS (applied before normalisation).
MIN_INPUT_DBFS = -60.0


def normalize_level(mono: np.ndarray, target_dbfs: float = TARGET_RMS_DBFS) -> np.ndarray:
    """Scale to the RMS level training used. Features are absolute dB, so a quiet phone capture
    (-35 to -45 dBFS) would otherwise sit 20-30 dB below anything the model has seen."""
    rms = np.sqrt(np.mean(mono**2))
    return mono if rms <= 0 else mono * (10 ** (target_dbfs / 20) / rms)


@torch.no_grad()
def score_file(model: BandNet, path: Path, normalize: bool = True) -> dict:
    with tempfile.TemporaryDirectory() as d:
        x, sr = decode(path, Path(d))
    mono = x.astype(np.float64).mean(axis=1)
    in_rms = 20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-12)
    if normalize and in_rms < MIN_INPUT_DBFS:
        return {"path": str(path), "error": f"input is {in_rms:.0f} dBFS: silence (below {MIN_INPUT_DBFS:.0f} dBFS)"}
    if normalize:
        mono = normalize_level(mono)
    windows = features.extract_windows(mono, sr)
    if not windows:
        return {"path": str(path), "error": "no window above the RMS gate (too short or silent)"}
    mel = np.stack([w[2] for w in windows]).astype(np.float32)
    out = model(torch.from_numpy(mel)[:, None])
    logits = out["ai_logit"].numpy()
    song_logit = float(logits.mean())
    return {
        "path": str(path),
        "orig_sample_rate": sr,
        "input_rms_dbfs": round(float(in_rms), 1),
        "level_normalized_to_dbfs": TARGET_RMS_DBFS if normalize else None,
        "p_ai": round(1 / (1 + np.exp(-song_logit)), 4),
        "band_scores": {b: round(float(v), 4) for b, v in zip(BAND_NAMES, out["band_scores"].mean(0).numpy())},
        "windows": [
            {"start_s": w[0], "p_ai": round(float(p), 4),
             "band_scores": {b: round(float(v), 4) for b, v in zip(BAND_NAMES, bs)}}
            for w, p, bs in zip(windows, out["p_ai"].numpy(), out["band_scores"].numpy())
        ],
        "_mel": mel, "_frame_logits": out["frame_logits"].numpy(),
        "_mel_ranges": model.mel_ranges,
    }


def plot(res: dict, png: Path) -> None:
    mel, fl, ranges = res["_mel"], res["_frame_logits"], res["_mel_ranges"]
    starts = [w["start_s"] for w in res["windows"]]
    # Windows overlap 50%; keep a non-overlapping chain (every 10 s) for the picture.
    keep = [0]
    for i, s in enumerate(starts):
        if s >= starts[keep[-1]] + 10 - 1e-6:
            keep.append(i)
    spec = np.concatenate([mel[i] for i in keep], axis=1)  # [128, frames]
    heat = np.full((128, len(keep) * fl.shape[2]), np.nan)
    for j, i in enumerate(keep):
        for b, name in enumerate(BAND_NAMES):
            lo, hi = ranges[name]
            heat[lo:hi, j * fl.shape[2] : (j + 1) * fl.shape[2]] = 1 / (1 + np.exp(-fl[i, b]))
    t_end = starts[keep[-1]] + 10

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 3, 1.2]})
    a1.imshow(spec, aspect="auto", origin="lower", cmap="magma", extent=[0, t_end, 0, 128], vmin=-100, vmax=0)
    a1.set_title(f"{Path(res['path']).name}: P(AI) = {res['p_ai']:.2f}")
    a1.set_ylabel("mel bin (20 Hz → 16 kHz)")
    a2.imshow(spec, aspect="auto", origin="lower", cmap="gray", extent=[0, t_end, 0, 128], vmin=-100, vmax=0)
    im = a2.imshow(heat, aspect="auto", origin="lower", cmap="coolwarm", alpha=0.6,
                   extent=[0, t_end, 0, 128], vmin=0, vmax=1)
    for name in BAND_NAMES:
        lo, hi = ranges[name]
        a2.axhline(lo, color="k", lw=0.5)
        a2.text(0.2, (lo + hi) / 2, name, va="center", fontsize=8, color="k",
                bbox={"facecolor": "white", "alpha": 0.6, "lw": 0})
    a2.set_ylabel("AI evidence by band")
    # Attach to all axes so every panel shrinks equally and the time axes stay aligned.
    fig.colorbar(im, ax=[a1, a2, a3], pad=0.01, label="frame P(AI)")
    a3.plot([s + 5 for s in starts], [w["p_ai"] for w in res["windows"]], "o-", ms=3)
    a3.axhline(0.5, color="gray", ls="--", lw=0.8)
    a3.set_ylim(0, 1)
    a3.set_ylabel("window P(AI)")
    a3.set_xlabel("time (s)")
    fig.savefig(png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("scores"))
    ap.add_argument("--no-normalize", action="store_true", help="score at the file's own level")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model)
    for f in args.files:
        res = score_file(model, f, normalize=not args.no_normalize)
        if "error" in res:
            print(f"{f}: {res['error']}")
            continue
        stem = args.out / f.stem
        plot(res, stem.with_suffix(".png"))
        clean = {k: v for k, v in res.items() if not k.startswith("_")}
        stem.with_suffix(".json").write_text(json.dumps(clean, indent=1))
        bands = " ".join(f"{k}={v:.2f}" for k, v in res["band_scores"].items())
        print(f"P(AI)={res['p_ai']:.3f}  [{bands}]  {len(res['windows'])} windows  {f.name}")


if __name__ == "__main__":
    main()
