"""Feature cache of real phone captures (`reports/phone_captures/*/segments.csv`).

    python -m wmg_train.phone_cache --out cache-phone

Every labelled segment, whether matched from a playlist or labelled as a whole session, is
level-normalised like the clean training data (-14 dBFS RMS +- 6 dB jitter) and windowed with
wmg_features. No codec/speaker simulation: the phone chain is already real. All windows are
kept (phone audio is scarce). The cache is derived data and is rebuilt from scratch each run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import features
from .augment import GAIN_JITTER_DB, TARGET_RMS_DBFS
from .playlist import CAPTURES

VARIANTS = 2  # level-jittered copies, matching the clean cache's two augmented variants


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", type=Path, default=CAPTURES)
    ap.add_argument("--out", type=Path, default=Path("cache-phone"))
    args = ap.parse_args()

    segs = []
    for csv_path in sorted(args.captures.glob("*/segments.csv")):
        df = pd.read_csv(csv_path)
        df["capture_dir"] = str(csv_path.parent)
        segs.append(df)
    if not segs:
        raise SystemExit(f"no segments.csv under {args.captures}; run playlist match / label-session first")
    segs = pd.concat(segs, ignore_index=True)

    args.out.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    with open(args.out / "windows.f16", "wb") as f:
        for s in segs.itertuples():
            x, sr = sf.read(str(Path(s.capture_dir) / s.segment), dtype="float64", always_2d=True)
            mono = x.mean(axis=1)
            rms = np.sqrt(np.mean(mono**2))
            if rms <= 0:
                continue
            in_db = 20 * np.log10(rms)
            for v in range(VARIANTS):
                seed = int(hashlib.sha1(f"{s.session_id}:{s.segment}:{v}".encode()).hexdigest()[:8], 16)
                level = TARGET_RMS_DBFS + float(np.random.default_rng(seed).uniform(-GAIN_JITTER_DB, GAIN_JITTER_DB))
                for start_s, w_rms, mel in features.extract_windows(mono * (10 ** (level / 20) / rms), sr):
                    f.write(np.ascontiguousarray(mel.astype(np.float16)).tobytes())
                    rows.append({
                        "song_id": s.song_id, "path": f"{s.capture_dir}/{s.segment}", "label": int(s.label),
                        "group": s.group, "variant": v, "start_s": start_s, "window_rms_dbfs": w_rms,
                        "orig_sr": sr, "codec": "phone", "bitrate_kbps": 0, "gain_db": round(level, 2),
                        "noise_snr_db": None, "lowpass_hz": None, "speaker": "real",
                        "session_id": s.session_id, "input_rms_dbfs": round(float(in_db), 1),
                    })
                    n += 1
    index = pd.DataFrame(rows)
    index["row"] = np.arange(len(index))
    index.to_parquet(args.out / "index.parquet")
    (args.out / "meta.json").write_text(json.dumps({
        "n": len(index), "shape": [128, 1000], "dtype": "float16", "source": "phone",
        "feature_spec_hash": features.spec_hash(), "feature_crate_commit": features.crate_commit(),
        "sessions": sorted(segs.session_id.unique().tolist()),
    }, indent=2))
    assert (args.out / "windows.f16").stat().st_size == len(index) * 128 * 1000 * 2
    by = index.groupby("label").agg(windows=("row", "size"), sessions=("session_id", "nunique"),
                                    groups=("group", "nunique"))
    print(f"{len(index)} phone windows from {segs.session_id.nunique()} sessions -> {args.out}")
    for label, r in by.iterrows():
        print(f"  {'AI   ' if label else 'human'}: {r.windows} windows, {r.sessions} sessions, {r.groups} groups")


if __name__ == "__main__":
    main()
