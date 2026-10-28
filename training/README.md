# wmg_train — dataset checks and student model for the AI-vs-human detector

Python side of the training pipeline. Features (log-mel) come **only** from the Rust
`wmg_features` module (`../features`, built with maturin). This package never re-implements them.

```
wmg_train/
  manifest.py   walk the data root, validate sidecars, one row per 10 s window -> Parquet + CSV
  splits.py     group split 70/15/15: positives by generator, negatives by artist_id
  balance.py    per-bucket class balance (subgenre, decade, codec+bitrate)
  confound.py   logistic regression on bitrate + sample rate + LUFS; test AUC must be <= 0.55
  features.py   thin wrapper over the Rust wmg_features module
  models/student.py   BandNet student (~0.39M params) + ONNX export
  cli.py        `wmg-check`: runs all of the above
configs/data.yaml, configs/student.yaml
```

## Data folder

```
training_data/
  ai_generated/     label 1
  human_talent/     label 0
  dj_mixes/         label 0
  computer_aided/   label 0 (human music produced with computers / DAWs)
  ai_hybrid/        label 1, hybrid = true (AI stems + human parts), always in test (eval only)
```

Files can sit in any subfolder. Supported: `.wav .flac .mp3 .ogg .aiff .aif` (libsndfile)
and `.m4a .aac` (needs a working `ffmpeg`/`ffprobe` on PATH). Every audio file needs a
**sidecar** with the same stem: `Song.mp3` → `Song.json`.

| field | who needs it |
|---|---|
| `track_id`, `subgenre`, `decade` (e.g. `"1990s"`), `license`, `source_codec`, `source_bitrate_kbps` (number) | everyone |
| `generator`, `generator_version`, `prompt_text`, `seed`, `generated_at` | `ai_generated`, `ai_hybrid` |
| `artist_id` | `human_talent`, `dj_mixes`, `computer_aided`, `ai_hybrid` |
| `collected_at` | optional (falls back to `generated_at`, then the file's mtime) |

A required field that is missing, `null` or `""` fails the build. If a value really doesn't apply
or is unknown, write the literal `"n/a"`. Fields that don't apply to a class get `"n/a"`
automatically. `track_id` must be unique across the whole tree.

```json
// ai_generated/suno/track-0001.json
{"track_id": "suno-0001", "subgenre": "deep house", "decade": "2020s", "license": "internal",
 "source_codec": "mp3", "source_bitrate_kbps": 320, "generator": "suno",
 "generator_version": "v4.5", "prompt_text": "warm deep house, 122 bpm", "seed": "n/a",
 "generated_at": "2026-09-30T12:00:00Z"}

// human_talent/track-0042.json
{"track_id": "ht-0042", "subgenre": "deep house", "decade": "2010s", "license": "label-owned",
 "source_codec": "flac", "source_bitrate_kbps": 1411, "artist_id": "artist-0042"}
```

The manifest also computes `sample_rate_orig`, `lufs` (ITU-R BS.1770 via pyloudnorm),
`window_start_s`/`duration_s` (10 s windows, 5 s hop, clips of 3–10 s padded to one window,
windows under −45 dBFS dropped, same as `wmg_features`), `split` and `feature_crate_commit`
(from the installed `wmg_features`). Beyond the plan's columns it adds `decade`, which the
balance check needs.

## Acceptance checks

`wmg-check` exits non-zero unless all of these pass:

1. **No nulls.** Every offending file and field is listed.
2. **No split leakage.** No generator, artist or track appears in two splits. Test has at
   least `min_test_generators` generators and `min_test_artists` (M) artists that are absent
   from train and val. Hybrids go to test only.
3. **Balance.** In every subgenre, decade and codec+bitrate bucket, the AI fraction is within
   ±10 points of the dataset-wide fraction, and no bucket is more than 80% one class. Failing
   buckets go to `<out>/balance_report.csv`.
4. **Confound.** Logistic regression on bitrate, sample rate and LUFS gets test AUC ≤ 0.55.

All thresholds live in `configs/data.yaml`.

## Setup and running

```bash
cd training
python3 -m venv .venv
.venv/bin/pip install -e '.[export]'
# x86_64 (Rosetta) Python on macOS tops out at torch 2.2.2 and needs NumPy 1.x:
#   .venv/bin/pip install -c constraints-macos-x86.txt -e '.[export]'
.venv/bin/pytest

# feature crate (needed for real runs: records feature_crate_commit, checks config vs crate spec)
.venv/bin/pip install maturin && .venv/bin/maturin develop --release -m ../features/pyproject.toml

.venv/bin/wmg-check --data-root ../training_data --out manifests/
```

The tests build a throwaway tree of short sine/noise WAVs with sidecars. They cover the null
check, balance failure on an 85/15 bucket, split leakage, confound AUC, and BandNet shapes,
parameter budget, allowed layer types, ONNX export and ONNX Runtime parity.

## Training device (local Mac, MPS)

BandNet trains locally with PyTorch on Apple-silicon MPS. Use `wmg_train.device.pick_device()`,
which tries cuda, then mps, then cpu (set `WMG_DEVICE=cpu|mps|cuda` to force one).

**Real training needs an arm64 Python.** `/usr/local/bin/python3*` on this Mac is an x86_64
(Rosetta) build: it gets torch 2.2.2 and has no MPS, so `pick_device()` returns CPU. Check with
`python3 -c "import platform; print(platform.machine())"` (it must print `arm64`). Get an arm64
interpreter, for example from arm64 Homebrew under `/opt/homebrew` or the python.org universal2
installer, and make a separate venv:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-arm64
.venv-arm64/bin/pip install -e '.[export]'
.venv-arm64/bin/python -c "from wmg_train.device import pick_device; print(pick_device())"  # mps
```

The x86 `.venv` is still fine for running the pytest suite on CPU.

## BandNet

Input: log-mel `[B, 1, 128, 1000]`. A shared depthwise-separable stem halves frequency and
quarters time. Three branches then each see only their mel bins, computed from the slaney mel
scale: bass 20–250 Hz (bins 0–7), mid 250–4000 Hz (8–80) and treble 4000–16000 Hz (81–127).
Each band emits per-frame logits. One attention head pools time (T' = 125, 80 ms frames).

`forward` returns `p_ai [B]`, `band_scores [B,3]`, `stem_scores [B,4]` (vocals/bass/drums/other),
`frame_logits [B,3,T']` and `attention [B,T']`. It also returns `ai_logit`/`stem_logits` for
BCE-with-logits training; those are not exported.

It uses only Conv2d (including depthwise), BatchNorm, ReLU/Hardswish, AvgPool2d, Linear,
Softmax/Sigmoid and elementwise mul/sum. `export_onnx()` writes those five outputs with a
dynamic batch axis.

## Phone-capture training data (playlist → app → labelled segments)

Clean files don't match what the app hears through a phone speaker and mic, so real
captures are collected with a labelled playlist:

1. **Build a playlist** of 60 s excerpts. The default is 30 human + 30 AI songs, about 62 minutes:
   ```bash
   .venv-arm64/bin/python -m wmg_train.playlist make --n-human 30 --n-ai 30 --seed 1
   ```
   This writes `reports/playlists/<playlist_id>/playlist.m4a` and `playlist.json`, which records which song plays when.
2. **Record it.** Copy `playlist.m4a` to the device that will play it, start a live session in the app (32 kHz), then press play. Keep everything else quiet.
   - **Two devices** match real use best: one plays, the phone running the app records.
   - **One phone:** make sure playback keeps going while the app records.
   - Pauses and skips are fine. Only the clip containing the pause or skip is lost.
3. **Match and cut** the session into labelled segments:
   ```bash
   .venv-arm64/bin/python -m wmg_train.playlist match <playlist_id> [--session <uuid>]
   ```
   This writes `reports/phone_captures/<session_id>/segments/*.wav` and `segments.csv`, with the label, song, artist group and alignment confidence for each segment.
   - Each 30 s clip is located on its own by cross-correlating onset envelopes.
   - A clip is accepted only when its offset agrees with a neighbouring clip.

About 1 hour of playback yields roughly 110 segments of 10–30 s. Use a new `--seed` for each playlist so each one draws a different mix of songs. Songs heard on a phone are still grouped by artist for cross-validation.

### Training on phone recordings of a streaming playlist (all one label)

When you play a playlist that isn't in `training_data/` (Spotify, Apple Music, ...), the tool can't match songs. Instead, every clip in the session gets one label.

1. Record the playlist through the app (live mode, 32 kHz). Sessions stop after `LIVE_MAX_MINUTES` (60 by default).
2. Label the session. It defaults to the most recent one; pass `--session <id>` to pick another:
   ```bash
   .venv-arm64/bin/python -m wmg_train.playlist label-session --label human \
       --source "Spotify: Indie Mix" --group spotify-indie-mix
   ```
   Artists aren't known, so the session is one cross-validation group and is held out as a whole. Give recordings of the same playlist the same `--group` so they're held out together.
3. Rebuild the phone cache. It includes every session in `reports/phone_captures/`, from both matched playlists and labelled sessions:
   ```bash
   .venv-arm64/bin/python -m wmg_train.phone_cache --out cache-phone
   ```
4. Train with clean and phone data together:
   ```bash
   .venv-arm64/bin/python -m wmg_train.train_baseline --cache cache-v2 --phone-cache cache-phone
   ```
   By default, 25% of each class's half of every batch comes from real phone windows (`--phone-frac`). `metrics.json` gains a `phone` summary, `phone_*` scores per fold, and an `oof_phone` block, which scores phone clips with models that never saw that artist or session.

**AI playlists** work the same way with `--label ai`. If a whole session is by one artist that's already in `ai_generated/`, pass that artist's existing group, for example `--group "ai:The Velvet Sundown"`. Then cross-validation holds the phone clips out together with that artist's clean files. A mixed playlist gets its own name, for example `--group suno-mix`.

**Balance guard:** training refuses to run if phone audio is more than 80% one class. Training on it would teach the model "phone audio = human", and AI songs played through a phone would then go undetected. Record AI music through the phone too, for example the AI half of a `playlist make` playlist, in similar amounts. `--allow-imbalanced-phone` overrides the guard for experiments only.
