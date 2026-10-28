import json
import math

import pandas as pd
import pytest

from wmg_train.manifest import (
    COLUMNS, ManifestError, build_manifest, null_offenses, read_manifest, write_manifest,
)

COMMIT = "test0000"


def test_builds_complete_manifest(data_root, data_cfg, tmp_path):
    df = build_manifest(data_root, data_cfg, COMMIT)
    assert list(df.columns) == COLUMNS
    assert df["track_id"].nunique() == 22
    assert not null_offenses(df)
    assert set(df["split"]) == {"train", "val", "test"}
    assert (df.loc[df["hybrid"], "split"] == "test").all()
    assert (df.loc[df["composition_type"] == "ai_hybrid", "label"] == 1).all()
    assert (df.loc[df["composition_type"] == "dj_mixes", "label"] == 0).all()
    # Not-applicable fields are the literal "n/a", never null.
    assert (df.loc[df["label"] == 0, "generator"] == "n/a").all()
    assert (df.loc[(df["label"] == 1) & ~df["hybrid"], "artist_id"] == "n/a").all()
    assert (df["sample_rate_orig"] == 16000).all()
    assert df["lufs"].between(-40, 0).all()

    pq, csv = write_manifest(df, tmp_path / "out")
    back = read_manifest(pq)
    pd.testing.assert_frame_equal(back, df.reset_index(drop=True), check_dtype=False)
    assert len(read_manifest(csv)) == len(df)


def _edit_sidecar(root, rel, **changes):
    p = root / rel
    d = json.loads(p.read_text())
    for k, v in changes.items():
        if v is ...:
            d.pop(k, None)
        else:
            d[k] = v
    p.write_text(json.dumps(d))


def test_null_fields_fail_with_offending_rows(data_root, data_cfg):
    _edit_sidecar(data_root, "human_talent/hu-0-0.json", artist_id=None)
    _edit_sidecar(data_root, "ai_generated/gen1/ai-1-0.json", license=...)
    _edit_sidecar(data_root, "ai_generated/gen2/ai-2-1.json", prompt_text="")
    (data_root / "dj_mixes/hu-4-0.json").unlink()

    with pytest.raises(ManifestError) as exc:
        build_manifest(data_root, data_cfg, COMMIT)
    got = {(o.path, o.field) for o in exc.value.offenses}
    assert got == {
        ("human_talent/hu-0-0.wav", "artist_id"),
        ("ai_generated/gen1/ai-1-0.wav", "license"),
        ("ai_generated/gen2/ai-2-1.wav", "prompt_text"),
        ("dj_mixes/hu-4-0.wav", "sidecar"),
    }
    assert "hu-0-0.wav: artist_id" in str(exc.value)


def test_na_literal_is_accepted(data_root, data_cfg):
    _edit_sidecar(data_root, "ai_generated/gen1/ai-1-0.json", seed="n/a")
    df = build_manifest(data_root, data_cfg, COMMIT)
    assert df.loc[df["track_id"] == "ai-1-0", "seed"].item() == "n/a"


def test_missing_feature_commit_is_a_null(data_root, data_cfg, monkeypatch):
    from wmg_train import features
    monkeypatch.setattr(features, "available", lambda: False)
    with pytest.raises(ManifestError) as exc:
        build_manifest(data_root, data_cfg, None)
    assert {o.field for o in exc.value.offenses} == {"feature_crate_commit"}


def test_null_offenses_flags_nan_and_none(data_root, data_cfg):
    df = build_manifest(data_root, data_cfg, COMMIT)
    df.loc[df.index[0], "lufs"] = math.nan
    df.loc[df.index[1], "subgenre"] = None
    found = {(o.clip_id, o.field) for o in null_offenses(df)}
    assert found == {(df.at[df.index[0], "clip_id"], "lufs"), (df.at[df.index[1], "clip_id"], "subgenre")}
