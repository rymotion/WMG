import numpy as np
import pandas as pd
import pytest

from wmg_train.confound import ConfoundError, check_confound, confound_auc
from wmg_train.manifest import build_manifest

CFG = {"features": ["source_bitrate_kbps", "sample_rate_orig", "lufs"], "max_auc": 0.55}


def synthetic(leaky: bool, n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    if leaky:  # AI delivered as 192 kbps / 44.1k / louder, human as 320 kbps / 48k
        bitrate = np.where(label == 1, 192, 320) + rng.normal(0, 5, n)
        sr = np.where(label == 1, 44100, 48000)
        lufs = np.where(label == 1, -8.0, -12.0) + rng.normal(0, 1, n)
    else:
        bitrate = rng.choice([128, 192, 256, 320], n).astype(float)
        sr = rng.choice([44100, 48000], n)
        lufs = rng.normal(-10, 2, n)
    split = rng.choice(["train", "val", "test"], n, p=[0.7, 0.15, 0.15])
    return pd.DataFrame({"label": label, "hybrid": False, "split": split,
                         "source_bitrate_kbps": bitrate, "sample_rate_orig": sr, "lufs": lufs})


def test_clean_metadata_near_chance():
    r = check_confound(synthetic(leaky=False), CFG)
    assert r.passed and r.auc <= 0.55
    assert r.n_train > r.n_test > 0


def test_leaky_metadata_fails():
    r = confound_auc(synthetic(leaky=True), CFG)
    assert r.auc > 0.95
    with pytest.raises(ConfoundError, match="AUC"):
        check_confound(synthetic(leaky=True), CFG)


def test_runs_on_built_manifest(data_root, data_cfg):
    df = build_manifest(data_root, data_cfg, "test0000")
    r = confound_auc(df, data_cfg["confound"])
    assert 0.0 <= r.auc <= 1.0
    assert set(r.coefficients) == set(CFG["features"])
