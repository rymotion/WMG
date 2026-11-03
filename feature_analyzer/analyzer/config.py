"""Paths shared by the app, the CLI and the tests."""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # feature_analyzer/
REPO = ROOT.parent                      # WMG/
TRAINING = REPO / "training"
RUNS = TRAINING / "runs"
HANDCRAFTED_REPORTS = REPO / "reports" / "handcrafted"
REPORTS_OUT = REPO / "reports" / "analyzer"

DATA = Path(os.environ.get("ANALYZER_DATA", ROOT / "data"))
UPLOADS = DATA / "uploads"
ANALYSES = DATA / "analyses"
SOURCES_JSON = DATA / "sources.json"

HOST, PORT = "127.0.0.1", 9200
RETRAIN_URL = os.environ.get("ANALYZER_RETRAIN_URL", "http://127.0.0.1:9100")
RETRAIN_HINT = "double-click training/Retrain Service.command"

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a", ".aac", ".mp4", ".caf"}
CATEGORIES = ("human", "ai_assisted", "ai")
