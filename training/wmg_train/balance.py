"""Per-bucket class-balance checks (subgenre, decade, codec+bitrate).

A bucket fails when its positive fraction is more than `tolerance` away from the dataset-wide
positive fraction, or when either class makes up more than `max_majority` of it. Hybrids are
eval-only and excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Upper bounds (kbps) for lossy bitrate buckets; anything above the last is "hi".
BITRATE_EDGES = [(96, "le96"), (128, "le128"), (192, "le192"), (256, "le256"), (320, "le320")]
LOSSLESS_CODECS = {"wav", "pcm", "flac", "alac", "aiff", "aif"}


def bitrate_bucket(kbps: float) -> str:
    for edge, name in BITRATE_EDGES:
        if kbps <= edge:
            return name
    return "hi"


def codec_bitrate_bucket(codec: str, kbps: float) -> str:
    codec = str(codec).lower()
    if codec in LOSSLESS_CODECS:
        return f"{codec}_lossless"
    return f"{codec}_{bitrate_bucket(float(kbps))}"


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["codec_bitrate"] = [
        codec_bitrate_bucket(c, b) for c, b in zip(out["source_codec"], out["source_bitrate_kbps"])
    ]
    return out


@dataclass(frozen=True)
class BalanceViolation:
    bucket_type: str
    bucket: str
    n: int
    pos_frac: float
    reason: str

    def __str__(self) -> str:
        return (f"{self.bucket_type}={self.bucket} (n={self.n}, AI={self.pos_frac:.0%}, "
                f"human={1 - self.pos_frac:.0%}): {self.reason}")


class BalanceError(ValueError):
    def __init__(self, violations: list[BalanceViolation], report: pd.DataFrame):
        self.violations = violations
        self.report = report
        super().__init__(f"{len(violations)} unbalanced bucket(s):\n  "
                         + "\n  ".join(str(v) for v in violations))


def bucket_report(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[BalanceViolation]]:
    data = add_buckets(df[~df["hybrid"].astype(bool)])
    if data.empty:
        raise ValueError("no non-hybrid rows to check")
    global_pos = float(data["label"].mean())
    tol, max_major = cfg["tolerance"], cfg["max_majority"]
    min_rows = cfg.get("min_bucket_rows", 0)

    records, violations = [], []
    for bucket_type in cfg["buckets"]:
        for bucket, g in data.groupby(bucket_type, sort=True):
            n, pos = len(g), float(g["label"].mean())
            reasons = []
            if max(pos, 1 - pos) > max_major:
                reasons.append(f">{max_major:.0%} one class")
            if abs(pos - global_pos) > tol:
                reasons.append(f"AI fraction off global {global_pos:.0%} by more than {tol:.0%}")
            enforced = n >= min_rows
            records.append({"bucket_type": bucket_type, "bucket": str(bucket), "n": n,
                            "n_ai": int(g["label"].sum()), "pos_frac": pos,
                            "ok": not reasons, "enforced": enforced, "reason": "; ".join(reasons)})
            if reasons and enforced:
                violations.append(BalanceViolation(bucket_type, str(bucket), n, pos, "; ".join(reasons)))
    return pd.DataFrame.from_records(records), violations


def check_balance(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return the per-bucket report; raise BalanceError if any enforced bucket fails."""
    report, violations = bucket_report(df, cfg)
    if violations:
        raise BalanceError(violations, report)
    return report
