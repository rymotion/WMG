"""Registry of labelled audio on this Mac (`data/sources.json`) and the per-file sidecar JSON
`wmg_train.manifest` expects next to each training file."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from wmg_train import labels
from wmg_train.manifest import NA

from . import config

HYBRID = {"ai_assisted"}


@dataclass
class Source:
    path: str
    category: str                      # human | ai_assisted | ai
    generator: str = ""
    artist: str = ""
    notes: str = ""
    added_at: str = ""
    sidecar_written_at: str = ""
    extra: dict = field(default_factory=dict)


class Registry:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else config.SOURCES_JSON
        self.items: list[Source] = []
        self.load()

    def load(self) -> None:
        if self.path.is_file():
            data = json.loads(self.path.read_text() or "[]")
            self.items = [Source(**{k: v for k, v in d.items() if k in Source.__dataclass_fields__}) for d in data]
        else:
            self.items = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps([asdict(s) for s in self.items], indent=1))
        tmp.replace(self.path)

    def find(self, path: str | Path) -> Source | None:
        p = str(Path(path).resolve())
        return next((s for s in self.items if s.path == p), None)

    def add(self, path: str | Path, category: str, generator: str = "", artist: str = "", notes: str = "") -> Source:
        if category not in labels.CATEGORIES:
            raise ValueError(f"category must be one of {sorted(labels.CATEGORIES)}, not {category!r}")
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        s = self.find(p)
        if s is None:
            s = Source(path=str(p), category=category, added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            self.items.append(s)
        s.category, s.generator, s.artist, s.notes = category, generator, artist, notes
        self.save()
        return s

    def remove(self, path: str | Path) -> bool:
        s = self.find(path)
        if s is None:
            return False
        self.items.remove(s)
        self.save()
        return True


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------

def sidecar_path(audio: str | Path) -> Path:
    return Path(audio).with_suffix(".json")


def sidecar_fields(s: Source, profile: dict | None = None, track_id: str | None = None) -> dict:
    """Every field `wmg_train.manifest.required_fields` can ask for, "n/a" where it doesn't apply."""
    from .analysis import profile_file, sha1_of
    p = Path(s.path)
    prof = profile if profile is not None else profile_file(p)
    label, hybrid = labels.label_of(s.category), s.category in HYBRID
    is_ai = label == 1 or hybrid
    is_human = label != 1 or hybrid
    ex = s.extra or {}
    return {
        "track_id": track_id or ex.get("track_id") or sha1_of(p)[:16],
        "category": s.category,
        "label": 1 if label == 1 else 0,
        "hybrid": hybrid,
        "subgenre": ex.get("subgenre") or NA,
        "decade": ex.get("decade") or NA,
        "license": ex.get("license") or NA,
        "source_codec": prof.get("codec") or p.suffix.lstrip(".").lower() or NA,
        "source_bitrate_kbps": prof.get("bitrate_kbps") or 0,
        "generator": (s.generator or NA) if is_ai else NA,
        "generator_version": (ex.get("generator_version") or NA) if is_ai else NA,
        "prompt_text": (s.notes or NA) if is_ai else NA,
        "seed": (ex.get("seed") or NA) if is_ai else NA,
        "generated_at": (ex.get("generated_at") or NA) if is_ai else NA,
        "artist_id": (s.artist or NA) if is_human else NA,
        "notes": s.notes,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_sidecar(s: Source, registry: Registry | None = None, profile: dict | None = None) -> Path:
    """Write (or update) `<audio>.json` next to the file. Existing keys we don't manage are kept."""
    sc = sidecar_path(s.path)
    existing = {}
    if sc.is_file():
        try:
            existing = json.loads(sc.read_text())
        except ValueError:
            existing = {}
    fields = sidecar_fields(s, profile, track_id=existing.get("track_id") if isinstance(existing, dict) else None)
    merged = {**(existing if isinstance(existing, dict) else {}), **fields}
    sc.write_text(json.dumps(merged, indent=1, ensure_ascii=False))
    s.sidecar_written_at = fields["collected_at"]
    if registry is not None:
        registry.save()
    return sc
