import pandas as pd
import pytest

from wmg_train.manifest import build_manifest
from wmg_train.splits import SplitError, assign_splits, check_no_leakage, held_out, leakage


def test_no_generator_or_artist_in_two_splits(data_root, data_cfg):
    df = build_manifest(data_root, data_cfg, "test0000")
    assert leakage(df) == {}
    for col, label in (("generator", 1), ("artist_id", 0)):
        sub = df[(df["label"] == label) & ~df["hybrid"]]
        assert (sub.groupby(col)["split"].nunique() == 1).all()
    assert len(held_out(df, "generator", 1)) >= data_cfg["splits"]["min_test_generators"]
    assert len(held_out(df, "artist_id", 0)) >= data_cfg["splits"]["min_test_artists"]
    assert (df.loc[df["hybrid"], "split"] == "test").all()
    for split in ("train", "val", "test"):
        assert set(df.loc[(df["split"] == split) & ~df["hybrid"], "label"]) == {0, 1}


def test_split_is_deterministic(data_root, data_cfg):
    a = build_manifest(data_root, data_cfg, "test0000")
    b = build_manifest(data_root, data_cfg, "test0000")
    assert a["split"].tolist() == b["split"].tolist()


def _synthetic(n_gen: int, n_art: int, per_group: int = 10) -> pd.DataFrame:
    rows = [{"label": 1, "hybrid": False, "generator": f"g{g}", "artist_id": "n/a",
             "track_id": f"g{g}-{i}"} for g in range(n_gen) for i in range(per_group)]
    rows += [{"label": 0, "hybrid": False, "generator": "n/a", "artist_id": f"a{a}",
              "track_id": f"a{a}-{i}"} for a in range(n_art) for i in range(per_group)]
    return pd.DataFrame(rows)


def test_fractions_roughly_70_15_15():
    cfg = {"fractions": {"train": 0.7, "val": 0.15, "test": 0.15}, "seed": 1,
           "min_test_generators": 1, "min_test_artists": 3}
    df = assign_splits(_synthetic(20, 40), cfg)
    frac = df["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.06
    assert abs(frac["val"] - 0.15) < 0.06
    assert abs(frac["test"] - 0.15) < 0.06


def test_too_few_generators_rejected():
    cfg = {"fractions": {"train": 0.7, "val": 0.15, "test": 0.15}, "seed": 1,
           "min_test_generators": 1, "min_test_artists": 1}
    with pytest.raises(SplitError, match="generator"):
        assign_splits(_synthetic(2, 10), cfg)


def test_leakage_detected():
    df = _synthetic(3, 3, per_group=2)
    df["split"] = "train"
    df.loc[0, "split"] = "test"  # one g0 track in test, the other in train
    assert leakage(df)["generator"] == ["g0"]
    with pytest.raises(SplitError, match="leakage"):
        check_no_leakage(df)
