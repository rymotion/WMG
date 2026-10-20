"""Build the clip manifest from a data root of class folders with JSON sidecars.

Layout:
    <root>/<class>/<any/sub/dirs>/<name>.<wav|flac|mp3|...>
    <root>/<class>/<any/sub/dirs>/<name>.json      # sidecar, same stem

One manifest row per 10 s analysis window that survives the RMS gate. Any null/empty required
field fails the whole build with the full list of offending rows; fields that don't apply to a
class are filled with the literal "n/a" (a sidecar may also say "n/a" explicitly).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import audio, features

NA = "n/a"

COLUMNS = [
    "clip_id", "path", "label", "hybrid", "split", "composition_type", "subgenre", "decade",
    "generator", "generator_version", "prompt_text", "seed", "artist_id", "track_id",
    "window_start_s", "duration_s", "source_codec", "source_bitrate_kbps", "sample_rate_orig",
    "lufs", "augmentations_applied", "feature_crate_commit", "license", "collected_at",
]

COMMON_FIELDS = ["track_id", "subgenre", "decade", "license", "source_codec", "source_bitrate_kbps"]
GENERATOR_FIELDS = ["generator", "generator_version", "prompt_text", "seed", "generated_at"]
ARTIST_FIELDS = ["artist_id"]


def required_fields(class_name: str, label: int, hybrid: bool) -> list[str]:
    if hybrid:  # human work made with AI tools: both provenance sides apply
        return COMMON_FIELDS + GENERATOR_FIELDS + ARTIST_FIELDS
    if label == 1:
        return COMMON_FIELDS + GENERATOR_FIELDS
    return COMMON_FIELDS + ARTIST_FIELDS


@dataclass(frozen=True)
class Offense:
    path: str
    field: str
    problem: str
    clip_id: str | None = None

    def __str__(self) -> str:
        where = self.clip_id or self.path
        return f"{where}: {self.field}: {self.problem}"


class ManifestError(ValueError):
    def __init__(self, offenses: list[Offense], limit: int = 200):
        self.offenses = offenses
        lines = [str(o) for o in offenses[:limit]]
        if len(offenses) > limit:
            lines.append(f"... and {len(offenses) - limit} more")
        super().__init__(f"{len(offenses)} manifest problem(s):\n  " + "\n  ".join(lines))


def _is_null(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and v.strip() == ""


def _iter_audio(class_dir: Path, exts: set[str]):
    for p in sorted(class_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."):
            yield p


def _collected_at(sidecar: dict, path: Path) -> str:
    for key in ("collected_at", "generated_at"):
        v = sidecar.get(key)
        if not _is_null(v) and v != NA:
            return str(v)
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    return mtime.isoformat(timespec="seconds")


def scan(
    data_root: Path, cfg: dict, feature_crate_commit: str | None = None
) -> tuple[pd.DataFrame, list[Offense]]:
    """Read every audio file + sidecar. Returns (rows without `split`, problems found)."""
    data_root = Path(data_root)
    exts = {e.lower() for e in cfg["audio_extensions"]}
    if feature_crate_commit is None and features.available():
        feature_crate_commit = features.crate_commit()

    rows: list[dict[str, Any]] = []
    offenses: list[Offense] = []
    track_paths: dict[str, str] = {}

    for class_name, spec in cfg["classes"].items():
        label, hybrid = int(spec["label"]), bool(spec["hybrid"])
        class_dir = data_root / class_name
        if not class_dir.is_dir():
            continue
        required = required_fields(class_name, label, hybrid)
        for path in _iter_audio(class_dir, exts):
            rel = str(path.relative_to(data_root))
            sidecar_path = path.with_suffix(".json")
            if not sidecar_path.is_file():
                offenses.append(Offense(rel, "sidecar", f"missing {sidecar_path.name}"))
                continue
            try:
                sidecar = json.loads(sidecar_path.read_text())
            except json.JSONDecodeError as e:
                offenses.append(Offense(rel, "sidecar", f"invalid JSON: {e}"))
                continue

            bad = [f for f in required if _is_null(sidecar.get(f))]
            offenses += [Offense(rel, f, "required field is null or missing") for f in bad]
            if bad:
                continue

            try:
                bitrate = float(sidecar["source_bitrate_kbps"])
            except (TypeError, ValueError):
                offenses.append(Offense(rel, "source_bitrate_kbps", "must be a number"))
                continue

            track_id = str(sidecar["track_id"])
            if track_id in track_paths:
                offenses.append(Offense(rel, "track_id", f"duplicate of {track_paths[track_id]}"))
                continue
            track_paths[track_id] = rel

            try:
                info = audio.probe(path, cfg["windows"])
            except Exception as e:  # unreadable / unsupported codec
                offenses.append(Offense(rel, "path", f"cannot decode audio: {e}"))
                continue
            if not info.window_starts_s:
                offenses.append(Offense(rel, "window_start_s", "no window survives length/RMS gate"))
                continue

            def field(name: str) -> str:
                v = sidecar.get(name)
                return NA if _is_null(v) else str(v)

            base = {
                "path": rel,
                "label": label,
                "hybrid": hybrid,
                "split": None,
                "composition_type": class_name,
                "subgenre": field("subgenre"),
                "decade": field("decade"),
                "generator": field("generator"),
                "generator_version": field("generator_version"),
                "prompt_text": field("prompt_text"),
                "seed": field("seed"),
                "artist_id": field("artist_id"),
                "track_id": track_id,
                "source_codec": field("source_codec").lower(),
                "source_bitrate_kbps": bitrate,
                "sample_rate_orig": info.sample_rate,
                "lufs": info.lufs,
                "augmentations_applied": "none",
                "feature_crate_commit": feature_crate_commit,
                "license": field("license"),
                "collected_at": _collected_at(sidecar, path),
            }
            win = cfg["windows"]["seconds"]
            for start in info.window_starts_s:
                rows.append({
                    **base,
                    "clip_id": f"{track_id}@{start:.2f}",
                    "window_start_s": round(start, 3),
                    "duration_s": round(min(win, info.duration_s - start), 3),
                })

    df = pd.DataFrame(rows, columns=COLUMNS)
    return df, offenses


def null_offenses(df: pd.DataFrame, columns: list[str] | None = None) -> list[Offense]:
    """Every (row, column) that is null/NaN/empty. `"n/a"` is a value, not a null."""
    out: list[Offense] = []
    for col in columns or COLUMNS:
        s = df[col]
        mask = s.isna() | (s.astype(str).str.strip() == "")
        for idx in df.index[mask]:
            out.append(Offense(str(df.at[idx, "path"]), col, "null", clip_id=str(df.at[idx, "clip_id"])))
    return out


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ManifestError([Offense("<manifest>", c, "column missing") for c in missing])
    offenses = null_offenses(df)
    if offenses:
        raise ManifestError(offenses)


def build_manifest(data_root: Path, cfg: dict, feature_crate_commit: str | None = None) -> pd.DataFrame:
    """scan -> fail on any problem -> assign group splits -> fail on any null."""
    from .splits import assign_splits

    df, offenses = scan(data_root, cfg, feature_crate_commit)
    offenses += [o for o in null_offenses(df) if o.field != "split"]
    if offenses:
        raise ManifestError(offenses)
    if df.empty:
        raise ManifestError([Offense(str(data_root), "path", "no audio files found")])
    df = assign_splits(df, cfg["splits"])
    validate(df)
    return df


def write_manifest(df: pd.DataFrame, out_dir: Path, stem: str = "manifest") -> tuple[Path, Path]:
    validate(df)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq, csv = out_dir / f"{stem}.parquet", out_dir / f"{stem}.csv"
    df.to_parquet(pq, index=False)
    df.to_csv(csv, index=False)
    return pq, csv


def read_manifest(path: Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
