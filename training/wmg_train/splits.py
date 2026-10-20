"""Group-wise train/val/test split: positives grouped by generator, negatives by artist_id.

No generator or artist appears in more than one split, so test measures generalisation to
unseen generators and artists. Hybrids (ai_hybrid) always go to test.
"""

from __future__ import annotations

import random

import pandas as pd

SPLITS = ("train", "val", "test")


class SplitError(ValueError):
    pass


def group_key(row: pd.Series) -> str:
    if bool(row["hybrid"]):
        return f"hybrid:{row['track_id']}"
    if int(row["label"]) == 1:
        return f"generator:{row['generator']}"
    return f"artist:{row['artist_id']}"


def _allocate(sizes: dict[str, int], fractions: dict[str, float], min_test: int,
              rng: random.Random, what: str) -> dict[str, str]:
    """Assign groups to splits: seed each split, then greedily fill the largest deficit."""
    groups = sorted(sizes)
    rng.shuffle(groups)
    need = min_test + 2  # min_test in test, >=1 in train and val
    if len(groups) < need:
        raise SplitError(f"need >= {need} distinct {what}s to split (min_test={min_test} plus "
                         f"one each for train and val), found {len(groups)}: {sorted(sizes)}")
    assign: dict[str, str] = {}
    totals = dict.fromkeys(SPLITS, 0)
    seeds = ["test"] * min_test + ["val", "train"]
    for g, split in zip(groups, seeds):
        assign[g] = split
        totals[split] += sizes[g]
    target_total = sum(sizes.values())
    # Largest groups first so the small ones can even out the fractions.
    for g in sorted(groups[len(seeds):], key=lambda g: -sizes[g]):
        split = max(SPLITS, key=lambda s: fractions[s] * target_total - totals[s])
        assign[g] = split
        totals[split] += sizes[g]
    return assign


def assign_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fractions = cfg["fractions"]
    if abs(sum(fractions[s] for s in SPLITS) - 1.0) > 1e-6:
        raise SplitError(f"split fractions must sum to 1: {fractions}")
    rng = random.Random(cfg.get("seed", 0))
    out = df.copy()
    keys = out.apply(group_key, axis=1)

    assign: dict[str, str] = {}
    pos = keys[(out["label"] == 1) & ~out["hybrid"].astype(bool)]
    neg = keys[out["label"] == 0]
    assign |= _allocate(pos.value_counts().to_dict(), fractions,
                        cfg.get("min_test_generators", 1), rng, "generator")
    assign |= _allocate(neg.value_counts().to_dict(), fractions,
                        cfg.get("min_test_artists", 1), rng, "artist")
    out["split"] = [("test" if k.startswith("hybrid:") else assign[k]) for k in keys]
    check_no_leakage(out)
    return out


def leakage(df: pd.DataFrame) -> dict[str, list[str]]:
    """Generators / artists / tracks that appear in more than one split (hybrids excluded)."""
    data = df[~df["hybrid"].astype(bool)]
    found: dict[str, list[str]] = {}
    for col, subset in (("generator", data[data["label"] == 1]),
                        ("artist_id", data[data["label"] == 0]),
                        ("track_id", df)):
        n_splits = subset.groupby(col)["split"].nunique()
        bad = sorted(str(v) for v in n_splits[n_splits > 1].index if v != "n/a")
        if bad:
            found[col] = bad
    return found


def check_no_leakage(df: pd.DataFrame) -> None:
    found = leakage(df)
    if found:
        raise SplitError(f"split leakage: {found}")


def held_out(df: pd.DataFrame, col: str, label: int) -> set[str]:
    """Values of `col` (generator/artist_id) that appear in test but not in train/val."""
    data = df[(df["label"] == label) & ~df["hybrid"].astype(bool)]
    test = set(data.loc[data["split"] == "test", col])
    seen = set(data.loc[data["split"] != "test", col])
    return test - seen
