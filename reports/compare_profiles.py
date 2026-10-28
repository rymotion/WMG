"""Compare AI vs human song profiles produced by features/examples/profile_dir.rs.

    python compare_profiles.py human_talent_profile/profiles.jsonl ai_generated_profile/profiles.jsonl

Prints per-feature distributions and single-feature AUCs (0.5 = indistinguishable; far from 0.5
in either direction = separable). Recording-chain features (codec, bitrate, cutoff, loudness)
separating the classes are confounds the model would learn instead of generator artifacts.
"""

import collections
import json
import sys

import numpy as np


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if "error" not in r], [r for r in rows if "error" in r]


def auc(pos, neg):
    """Mann–Whitney AUC: P(random AI value > random human value)."""
    v = np.concatenate([pos, neg])
    ranks = np.empty(len(v))
    order = v.argsort(kind="mergesort")
    ranks[order] = np.arange(1, len(v) + 1)
    # Average ties.
    for val in np.unique(v):
        idx = v == val
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    r = ranks[: len(pos)].sum()
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def logreg_cv_auc(X, y, folds=5, seed=0, l2=1e-2, iters=2000):
    """Plain-numpy logistic regression, k-fold out-of-fold AUC."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    scores = np.zeros(len(y))
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.c_[(X[tr] - mu) / sd, np.ones(len(tr))]
        w = np.zeros(A.shape[1])
        wpos = len(y[tr]) / (2 * max(y[tr].sum(), 1))
        wneg = len(y[tr]) / (2 * max((1 - y[tr]).sum(), 1))
        sw = np.where(y[tr] == 1, wpos, wneg)
        for _ in range(iters):
            p = 1 / (1 + np.exp(-A @ w))
            w -= 0.1 * (A.T @ (sw * (p - y[tr])) / len(tr) + l2 * w)
        scores[te] = np.c_[(X[te] - mu) / sd, np.ones(len(te))] @ w
    return auc(scores[y == 1], scores[y == 0])


def q(v):
    v = np.asarray(v, float)
    return f"p10={np.percentile(v, 10):8.1f}  med={np.median(v):8.1f}  p90={np.percentile(v, 90):8.1f}"


def main(human_path, ai_path):
    hum, herr = load(human_path)
    ai, aerr = load(ai_path)
    print(f"human {len(hum)} songs ({len(herr)} errors), ai {len(ai)} songs ({len(aerr)} errors)")
    for r in aerr + herr:
        print("  ERR", r["path"], r["error"])
    win_h, win_a = sum(r["n_windows"] for r in hum), sum(r["n_windows"] for r in ai)
    print(f"usable 10 s windows: human {win_h}, ai {win_a} (ratio {win_h / max(win_a, 1):.1f}:1)\n")

    C = collections.Counter
    for name, key in [("codec", lambda r: r["codec"]), ("sample_rate", lambda r: r["sample_rate"]),
                      ("channels", lambda r: r["channels"])]:
        print(f"{name:12s} human {dict(C(map(key, hum)))}\n{'':12s} ai    {dict(C(map(key, ai)))}")
    print()

    feats = {
        "bitrate_kbps": lambda r: r["bitrate_kbps"],
        "cutoff_hz": lambda r: r["cutoff_hz"],
        "duration_s": lambda r: r["duration_s"],
        "rms_dbfs": lambda r: r["rms_dbfs"],
        "peak_dbfs": lambda r: r["peak_dbfs"],
        "crest_db": lambda r: r["crest_db"],
        "side_to_mid_db": lambda r: max(r["side_to_mid_db"], -60),
        "band_bass_db": lambda r: r["band_mean_db"]["bass"],
        "band_mid_db": lambda r: r["band_mean_db"]["mid"],
        "band_treble_db": lambda r: r["band_mean_db"]["treble"],
        "treble_minus_mid": lambda r: r["band_mean_db"]["treble"] - r["band_mean_db"]["mid"],
        "bass_minus_mid": lambda r: r["band_mean_db"]["bass"] - r["band_mean_db"]["mid"],
        "treble_flatness": lambda r: r["treble_flatness"] * 100,
        "spectral_flux": lambda r: r["spectral_flux"] * 10,
    }
    print(f"{'feature':18s} {'AUC':>5s}   human{'':38s} ai")
    for name, f in feats.items():
        h, a = np.array([f(r) for r in hum]), np.array([f(r) for r in ai])
        flag = " <-- separates" if abs(auc(a, h) - 0.5) > 0.2 else ""
        print(f"{name:18s} {auc(a, h):5.2f}   {q(h)}   {q(a)}{flag}")

    # Level-normalised mel profile: subtract each song's mean so only spectral *shape* remains.
    mh = np.array([r["mel_mean_db"] for r in hum])
    ma = np.array([r["mel_mean_db"] for r in ai])
    sh, sa = mh - mh.mean(1, keepdims=True), ma - ma.mean(1, keepdims=True)
    bin_auc = np.array([auc(sa[:, i], sh[:, i]) for i in range(mh.shape[1])])
    diff = sa.mean(0) - sh.mean(0)
    print("\nspectral shape (level-normalised mel mean), AI minus human, dB, by 8-bin group:")
    hz = mel_centres()
    for g in range(0, 128, 8):
        print(f"  bins {g:3d}-{g + 7:3d} ({hz[g]:6.0f}-{hz[g + 7]:6.0f} Hz)  diff {diff[g:g + 8].mean():+6.2f}  "
              f"AUC {bin_auc[g:g + 8].mean():.2f}")

    # Native-rate LTAS above 16 kHz (invisible to the detector, but a codec fingerprint).
    def hf(r):
        f, d = np.array(r["ltas_hz"]), np.array(r["ltas_db"])
        ref = np.median(d[(f > 200) & (f < 4000)])
        return d[(f > 16000) & (f < 20000)].mean() - ref
    h, a = np.array([hf(r) for r in hum]), np.array([hf(r) for r in ai])
    print(f"\n16-20 kHz level re mids (above detector fmax): AUC {auc(a, h):.2f}  human {q(h)}  ai {q(a)}")

    # Spec acceptance gate: codec-only baseline (bitrate + sample rate + loudness).
    y = np.r_[np.ones(len(ai)), np.zeros(len(hum))]
    X = np.array([[r["bitrate_kbps"], r["sample_rate"], r["rms_dbfs"]] for r in ai + hum])
    cb = logreg_cv_auc(X, y)
    print(f"\nconfound baseline (bitrate+sample rate+RMS, 5-fold logreg) AUC = {cb:.2f}  "
          f"({'PASS' if cb <= 0.55 else 'FAIL'}, gate 0.55)")
    Xs = np.c_[X, [[r["cutoff_hz"]] for r in ai + hum]]
    print(f"  + cutoff_hz: AUC = {logreg_cv_auc(Xs, y):.2f}")
    Xshape = np.r_[sa, sh]
    print(f"shape-only baseline (128 level-normalised mel means, logreg) AUC = {logreg_cv_auc(Xshape, y):.2f}")


def mel_centres():
    mel = lambda f: 3 * f / 200 if f < 1000 else 15 + 27 * np.log(f / 1000) / np.log(6.4)
    inv = lambda m: 200 * m / 3 if m < 15 else 1000 * np.exp((m - 15) * np.log(6.4) / 27)
    lo, hi = mel(20), mel(16000)
    return [inv(lo + (hi - lo) * (i + 1) / 129) for i in range(128)]


if __name__ == "__main__":
    main(*sys.argv[1:3])
