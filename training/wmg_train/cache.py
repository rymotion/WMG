"""Build the augmented log-mel cache for baseline training.

    python -m wmg_train.cache --data ../training_data --out cache

Writes `windows.f16` (raw float16, [N, 128, 1000]) and `index.parquet` (one row per window).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import features
from .augment import augment
from .groups import Song, inventory

VARIANTS = 2
MAX_WINDOWS = {0: 12, 1: 30}  # per song per variant, by label


def _seed(song_id: str, variant: int) -> int:
    return int(hashlib.sha1(f"{song_id}:{variant}".encode()).hexdigest()[:8], 16)


def process(song: Song) -> tuple[list[dict], np.ndarray | None, str | None]:
    rows, mats = [], []
    try:
        for v in range(VARIANTS):
            rng = np.random.default_rng(_seed(song.song_id, v))
            mono, sr, meta = augment(song.path, rng)
            windows = features.extract_windows(mono, sr)
            k = min(MAX_WINDOWS[song.label], len(windows))
            for i in sorted(rng.choice(len(windows), size=k, replace=False)):
                start_s, rms, mel = windows[i]
                mats.append(mel.astype(np.float16))
                rows.append({
                    "song_id": song.song_id, "path": str(song.path), "label": song.label,
                    "group": song.group, "variant": v, "start_s": start_s,
                    "window_rms_dbfs": rms, "orig_sr": sr,
                    **(asdict(meta) | {"speaker": json.dumps(meta.speaker) if meta.speaker else None}),
                })
    except Exception as e:  # noqa: BLE001 - report and continue with the other songs
        return [], None, f"{song.path}: {e}"
    return rows, (np.stack(mats) if mats else None), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("../training_data"))
    ap.add_argument("--out", type=Path, default=Path("cache"))
    # Each worker holds a decoded song plus FFT buffers (~0.5-1 GB); keep this low on a
    # machine already under memory pressure, or macOS swap eats the disk.
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    args = ap.parse_args()
    free_gb = shutil.disk_usage(args.out.parent if not args.out.exists() else args.out).free / 1e9
    if free_gb < args.min_free_gb + 2.5:
        raise SystemExit(f"need ~{args.min_free_gb + 2.5:.0f} GB free for the cache, have {free_gb:.1f} GB")

    songs = inventory(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(songs)} songs ({sum(s.label for s in songs)} AI) in "
          f"{len({s.group for s in songs})} groups", flush=True)

    rows: list[dict] = []
    errors: list[str] = []
    with open(args.out / "windows.f16", "wb") as f, ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(process, s) for s in songs]
        for n, fut in enumerate(as_completed(futures), 1):
            r, m, err = fut.result()
            free_gb = shutil.disk_usage(args.out).free / 1e9
            if free_gb < args.min_free_gb:
                pool.shutdown(cancel_futures=True)
                raise SystemExit(f"stopping: only {free_gb:.1f} GB free (< --min-free-gb {args.min_free_gb}); "
                                 f"partial cache left in {args.out}")
            if err:
                errors.append(err)
            elif m is not None:
                f.write(np.ascontiguousarray(m).tobytes())
                rows.extend(r)
            if n % 25 == 0 or n == len(songs):
                print(f"{n}/{len(songs)} songs, {len(rows)} windows", flush=True)

    index = pd.DataFrame(rows)
    index["row"] = np.arange(len(index))
    index.to_parquet(args.out / "index.parquet")
    (args.out / "meta.json").write_text(json.dumps({
        "n": len(index), "shape": [128, 1000], "dtype": "float16",
        "feature_spec_hash": features.spec_hash(), "feature_crate_commit": features.crate_commit(),
        "errors": errors,
    }, indent=2))
    size = (args.out / "windows.f16").stat().st_size
    assert size == len(index) * 128 * 1000 * 2, "index/memmap size mismatch"
    print(f"done: {len(index)} windows, {size / 1e9:.2f} GB, {len(errors)} errors")
    for e in errors:
        print("  ERR", e)


if __name__ == "__main__":
    main()
