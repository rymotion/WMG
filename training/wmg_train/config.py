"""YAML config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load `configs/<name>.yaml`, or an explicit path to a YAML file."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    with path.open() as f:
        return yaml.safe_load(f)
