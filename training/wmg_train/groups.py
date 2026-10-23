"""Song inventory for the baseline: label, artist group and duplicate handling.

Groups are what cross-validation holds out, so every song by the same artist (including
collaborations) must share one group, or the model can be scored on artists it trained on.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

AUDIO_EXT = {".m4a", ".mp3", ".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp4", ".aac"}
# Preferred copy when the same AI track exists in several formats.
_EXT_RANK = {".wav": 0, ".aiff": 0, ".aif": 0, ".flac": 0, ".m4a": 1, ".mp3": 2}


@dataclass(frozen=True)
class Song:
    song_id: str
    path: Path
    label: int  # 1 = AI-generated, 0 = human
    group: str


def song_id(rel: str) -> str:
    return hashlib.sha1(rel.encode()).hexdigest()[:12]


def _audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def human_songs(root: Path, prefix: str = "human") -> list[Song]:
    """Top folder = artist. `Music/<x>` copies of an existing `<x>` are skipped. Artists joined
    by " & " are merged with each member (so "Beezy & Faja the Poet" groups with "Faja the Poet")."""
    files = _audio_files(root)
    rels = {p.relative_to(root) for p in files}
    keep = []
    for p in files:
        rel = p.relative_to(root)
        if rel.parts[0] == "Music" and Path(*rel.parts[1:]) in rels:
            continue
        keep.append((p, rel))

    uf = _UnionFind()
    artist_of = {}
    for p, rel in keep:
        artist = rel.parts[1] if rel.parts[0] == "Music" and len(rel.parts) > 2 else rel.parts[0]
        if len(rel.parts) == 1:  # flat folder: "Artist - Title.ext"
            artist = p.stem.split(" - ")[0].strip() if " - " in p.stem else p.stem
        artist_of[p] = artist
        members = [m.strip() for m in artist.split("&")]
        for m in members:
            uf.union(m, members[0])
        uf.find(artist)
        uf.union(artist, members[0])
    return [
        Song(song_id(f"{prefix}/{rel}"), p, 0, "human:" + uf.find(artist_of[p]))
        for p, rel in keep
    ]


def ai_artist(stem: str) -> str:
    if stem.startswith("AI D-42"):
        return "The Universe Sings"
    if " - " in stem:
        artist = stem.rsplit(" - ", 1)[1].split(",")[0].strip()
        if artist.upper() != "ORIGIN":
            return artist
    return "unknown"


def _ai_key(p: Path, root: Path) -> str:
    """Dedupe key: folder + title with download suffixes removed, so "Pulse - usesuno.com.wav"
    and "Pulse [f802dde3].mp3" count as one song."""
    stem = re.sub(r"\s*\[[0-9a-f]{6,}\]$", "", p.stem.strip())  # Suno MP3 id suffix
    stem = re.sub(r"\s*-\s*usesuno\.com$", "", stem, flags=re.I)  # Suno WAV suffix
    folder = p.parent.relative_to(root).as_posix()
    return folder + "/" + re.sub(r"\s+", " ", stem.lower())


def ai_songs(root: Path) -> list[Song]:
    """`ai_generated/`: flat "Title - Artist.ext" files, plus one subfolder per generator
    (e.g. `suno/`), grouped as `ai:<folder>`. The same song in several formats is kept once,
    preferring lossless."""
    best: dict[str, Path] = {}
    for p in _audio_files(root):
        key = _ai_key(p, root)
        cur = best.get(key)
        if cur is None or _EXT_RANK.get(p.suffix.lower(), 3) < _EXT_RANK.get(cur.suffix.lower(), 3):
            best[key] = p
    def group(p: Path) -> str:
        rel = p.relative_to(root)
        return "ai:" + (rel.parts[0] if len(rel.parts) > 1 else ai_artist(p.stem))

    return [Song(song_id("ai/" + str(p.relative_to(root))), p, 1, group(p)) for p in sorted(best.values())]


# Human-made folders. computer_aided = produced with computers/DAWs, which counts as human;
# music containing AI-generated stems belongs in ai_hybrid/ (eval only, not used here).
HUMAN_FOLDERS = ("human_talent", "computer_aided", "dj_mixes")


def inventory(data_root: Path) -> list[Song]:
    songs = []
    for folder in HUMAN_FOLDERS:
        if (data_root / folder).is_dir():
            # Groups are named by artist, so the same artist in two folders shares a CV group
            # (collaboration merging only happens within a folder).
            songs += human_songs(data_root / folder, prefix="human" if folder == "human_talent" else folder)
    return songs + ai_songs(data_root / "ai_generated")
