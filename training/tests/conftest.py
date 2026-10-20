"""Synthetic data root: tiny sine/noise WAVs with sidecars for every class folder."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wmg_train.config import load_config

SR = 16000
SECONDS = 3.5  # >= min_partial_seconds -> exactly one zero-padded window per file


def write_track(path: Path, sidecar: dict, rng: np.random.Generator, freq: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(SR * SECONDS)) / SR
    x = 0.3 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(t.size)
    sf.write(path, x.astype(np.float32), SR)
    path.with_suffix(".json").write_text(json.dumps(sidecar))


def _common(track_id: str, i: int) -> dict:
    return {
        "track_id": track_id,
        "subgenre": ["house", "techno"][i % 2],
        "decade": ["2020s", "1990s"][(i // 2) % 2],
        "license": "CC-BY-4.0",
        "source_codec": ["mp3", "flac"][i % 2],
        "source_bitrate_kbps": [320, 900][i % 2],
    }


def build_tree(root: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    n = 0
    for g in range(5):  # 5 generators x 2 tracks
        for k in range(2):
            tid = f"ai-{g}-{k}"
            write_track(root / "ai_generated" / f"gen{g}" / f"{tid}.wav", {
                **_common(tid, n), "generator": f"gen{g}", "generator_version": "v1",
                "prompt_text": "driving four on the floor", "seed": 42 + k,
                "generated_at": "2026-09-01T00:00:00Z"}, rng, 220 + 40 * n)
            n += 1
    for a in range(6):  # 6 artists x 1-2 tracks across two negative folders
        folder = "human_talent" if a < 4 else "dj_mixes"
        for k in range(2 if a < 4 else 1):
            tid = f"hu-{a}-{k}"
            write_track(root / folder / f"{tid}.wav",
                        {**_common(tid, n), "artist_id": f"artist{a}"}, rng, 220 + 40 * n)
            n += 1
    for h in range(2):
        tid = f"ca-{h}"
        write_track(root / "ai_hybrid" / f"{tid}.wav", {
            **_common(tid, n), "artist_id": f"artist{h}", "generator": "gen0",
            "generator_version": "n/a", "prompt_text": "n/a", "seed": "n/a",
            "generated_at": "2026-09-01T00:00:00Z"}, rng, 300 + 20 * h)
        n += 1
    return root


@pytest.fixture
def data_cfg() -> dict:
    cfg = copy.deepcopy(load_config("data"))
    cfg["splits"]["min_test_artists"] = 2  # tiny fixture: 6 artists total
    return cfg


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return build_tree(tmp_path / "data")
