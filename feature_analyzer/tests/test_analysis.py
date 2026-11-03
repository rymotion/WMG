import json
import math

import pytest

from analyzer import analysis as A


def test_wall_vs_fullband(data_dir, wav_wall, wav_full, models_ok):
    a = A.analyze(wav_wall, meta={"category": "ai", "notes": "synthetic"}, models=models_ok, force=True)
    b = A.analyze(wav_full, meta={"category": "human"}, models=models_ok, force=True)

    # Profile
    for x in (a, b):
        assert x.profile["sample_rate"] == 44_100 and x.profile["channels"] == 2
        assert 74 < x.profile["duration_s"] < 76
        assert -30 < x.profile["rms_dbfs"] < 0
        assert x.profile["codec"] in ("pcm", "lpcm", "wav") or "bitrate_kbps" in x.profile

    # Hand-crafted features: the wall shows as a ~16 kHz cutoff, full band reaches Nyquist.
    fa, fb = a.features["raw"], b.features["raw"]
    assert 15_000 < fa["fc_hz"] < 17_000
    assert fb["fc_hz"] > 20_000
    assert -100 <= fa["residue_db"] <= 0 and -100 <= fb["residue_db"] <= 0
    # 128 kbps AAC low-passes the full-band file (afconvert keeps ~18 kHz for noise)
    assert b.features["aac128"]["fc_hz"] < fb["fc_hz"] and b.features["aac128"]["fc_hz"] < 19_500
    for cond in A.CONDITIONS:
        assert set(A.FEATURE_HELP) <= set(a.features[cond])

    # Side features on every window, images written, JSON valid and reloadable.
    assert len(a.side_features) >= 10 and all(k in a.side_features[0] for k in ("flux_p2m", "mod_peak_db", "sfm_mean"))
    assert (a.dir / "spectrum.png").is_file() and (a.dir / "timeline.png").is_file()
    d = json.loads((a.dir / "analysis.json").read_text())
    assert d["category"] == "ai" and d["notes"] == "synthetic"
    assert A.load(a.id).features["raw"]["fc_hz"] == pytest.approx(fa["fc_hz"])
    assert not any(isinstance(v, float) and math.isnan(v) for v in d["features"]["raw"].values())

    if models_ok:
        for x in (a, b):
            for m in A.MODEL_NAMES:
                s = x.scores[m]
                assert "error" not in s, s
                assert 0 <= s["p_ai"] <= 1 and len(s["windows"]) >= 10
                assert set(s["band_scores"]) == {"bass", "mid", "treble"}
            assert (x.dir / "heatmap_10s.png").is_file()
        # 5 s windows are the halves of the 10 s windows, deduplicated
        assert a.scores["5s"]["windows"][1]["start_s"] == 5.0
        assert len(a.scores["5s"]["windows"]) == len(a.scores["10s"]["windows"]) + 1
    else:
        assert a.models_skipped

    # Cache: a second call returns the stored analysis and applies new metadata.
    c = A.analyze(wav_wall, meta={"generator": "synthetic"})
    assert c.created_at == a.created_at and c.generator == "synthetic" and c.category == "ai"
    assert {x.id for x in A.list_analyses()} >= {a.id, b.id}


def test_reference_and_rows(data_dir, wav_wall, models_ok):
    ref = A.reference()
    a = A.analyze(wav_wall, models=False)
    rows = A.feature_rows(a, ref)
    assert [r["name"] for r in rows][:3] == ["fc_hz", "notch_db", "slope_break"]
    if ref.get("folder"):
        fc = next(r for r in rows if r["name"] == "fc_hz")["conds"]["raw"]
        assert fc["human"] and fc["ai"] and fc["human"][0] <= fc["human"][1] <= fc["human"][2]
        assert fc["where"] in ("human", "ai", "both", "neither")
    assert A.where(5.0, [1, 2, 3], [4, 5, 6]) == "ai"
    assert A.where(None, [1, 2, 3], [4, 5, 6]) == ""
    md = A.summary_markdown([a], ref)
    assert "| fc_hz |" in md and a.name in md
