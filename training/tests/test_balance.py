import pandas as pd
import pytest

from wmg_train.balance import BalanceError, check_balance, codec_bitrate_bucket

CFG = {"tolerance": 0.10, "max_majority": 0.80, "min_bucket_rows": 0,
       "buckets": ["subgenre", "decade", "codec_bitrate"]}


def frame(buckets: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """{subgenre: (n_ai, n_human)} with decade/codec held constant."""
    rows = []
    for sub, (n_ai, n_hu) in buckets.items():
        rows += [{"subgenre": sub, "label": lab, "hybrid": False, "decade": "2020s",
                  "source_codec": "mp3", "source_bitrate_kbps": 320}
                 for lab, n in ((1, n_ai), (0, n_hu)) for _ in range(n)]
    return pd.DataFrame(rows)


def test_balanced_buckets_pass():
    report = check_balance(frame({"house": (50, 50), "techno": (45, 55), "trance": (55, 45)}), CFG)
    assert report["ok"].all()


def test_85_15_bucket_fails():
    df = frame({"house": (200, 200), "techno": (200, 200), "dubstep": (85, 15)})
    with pytest.raises(BalanceError) as exc:
        check_balance(df, CFG)
    bad = {(v.bucket_type, v.bucket) for v in exc.value.violations}
    assert ("subgenre", "dubstep") in bad
    assert ("subgenre", "house") not in bad
    assert ">80% one class" in str(exc.value)


def test_outside_tolerance_fails_even_below_80():
    # techno 70% AI vs global ~57%: under the 80% cap but >10 points off.
    df = frame({"house": (100, 100), "techno": (70, 30)})
    with pytest.raises(BalanceError, match="off global"):
        check_balance(df, CFG)


def test_hybrids_ignored():
    df = frame({"house": (50, 50)})
    hybrids = frame({"house": (40, 0)}).assign(hybrid=True)
    check_balance(pd.concat([df, hybrids]), CFG)


def test_codec_bitrate_buckets():
    assert codec_bitrate_bucket("MP3", 320) == "mp3_le320"
    assert codec_bitrate_bucket("mp3", 128) == "mp3_le128"
    assert codec_bitrate_bucket("aac", 256) == "aac_le256"
    assert codec_bitrate_bucket("flac", 900) == "flac_lossless"
