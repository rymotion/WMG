import json
from pathlib import Path

import pytest
import yaml

from analyzer import config
from analyzer.sources import Registry, sidecar_path, write_sidecar


def test_registry_round_trip(data_dir, wav_wall, wav_full):
    reg = Registry(data_dir / "sources_rt.json")
    s = reg.add(wav_wall, "ai", generator="synthetic", notes="wall")
    reg.add(wav_full, "human", artist="noise band")
    reg.add(wav_wall, "ai", generator="synthetic v2")  # same path: updated, not duplicated
    assert len(reg.items) == 2 and s.generator == "synthetic v2"

    again = Registry(data_dir / "sources_rt.json")
    assert [x.path for x in again.items] == [str(wav_wall.resolve()), str(wav_full.resolve())]
    assert again.find(wav_full).artist == "noise band"
    assert again.remove(wav_full) and not again.find(wav_full)
    assert len(Registry(data_dir / "sources_rt.json").items) == 1
    with pytest.raises(ValueError):
        reg.add(wav_wall, "robot")
    with pytest.raises(FileNotFoundError):
        reg.add(data_dir / "nope.wav", "ai")


def test_sidecar_is_accepted_by_manifest(data_dir, tmp_path, wav_wall, wav_full):
    """Lay out <root>/<class>/file.wav + sidecar and let wmg_train.manifest.scan validate it."""
    from wmg_train import manifest
    cfg = yaml.safe_load((config.TRAINING / "configs" / "data.yaml").read_text())
    root = tmp_path / "data_root"
    ai, human, hybrid = root / "ai_generated", root / "human_talent", root / "ai_hybrid"
    for d in (ai, human, hybrid):
        d.mkdir(parents=True)
    ai_file, human_file, hybrid_file = ai / "a.wav", human / "h.wav", hybrid / "x.wav"
    ai_file.write_bytes(wav_wall.read_bytes())
    human_file.write_bytes(wav_full.read_bytes())
    # a third, distinct file: track_id is the content hash and must be unique per track
    import numpy as np, soundfile as sf
    sf.write(str(hybrid_file), (np.random.default_rng(7).standard_normal((44_100 * 20, 2)) * 0.3).astype("float32"), 44_100)

    reg = Registry(data_dir / "sources2.json")
    paths = []
    for f, cat, kw in ((ai_file, "ai", {"generator": "suno"}), (human_file, "human", {"artist": "noise band"}),
                       (hybrid_file, "ai_assisted", {"generator": "udio", "artist": "someone"})):
        s = reg.add(f, cat, **kw)
        paths.append(write_sidecar(s, reg))
    assert all(p == sidecar_path(f) and p.is_file() for p, f in zip(paths, (ai_file, human_file, hybrid_file)))

    sc = json.loads(paths[0].read_text())
    assert sc["generator"] == "suno" and sc["artist_id"] == "n/a" and sc["source_codec"] and sc["track_id"]
    sc_h = json.loads(paths[1].read_text())
    assert sc_h["artist_id"] == "noise band" and sc_h["generator"] == "n/a"
    sc_x = json.loads(paths[2].read_text())
    assert sc_x["generator"] == "udio" and sc_x["artist_id"] == "someone"

    # Rewriting keeps the track_id and any extra keys already in the file.
    extra = {**sc, "my_key": 1}
    paths[0].write_text(json.dumps(extra))
    write_sidecar(reg.find(ai_file), reg)
    sc2 = json.loads(paths[0].read_text())
    assert sc2["track_id"] == sc["track_id"] and sc2["my_key"] == 1

    df, offenses = manifest.scan(root, cfg)
    assert offenses == [], [str(o) for o in offenses]
    assert set(df.composition_type) == {"ai_generated", "human_talent", "ai_hybrid"}
    assert set(df.label) == {0, 1} and df[df.composition_type == "ai_hybrid"].hybrid.all()
    assert reg.find(ai_file).sidecar_written_at
