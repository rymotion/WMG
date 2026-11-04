# feature_analyzer

Local web app + CLI for pattern analysis of possibly AI-generated audio. Reuses `training/wmg_train`
(decode, hand-crafted top-end features, BandNet scoring) and the Rust `wmg_features` module (side features).
Audio never leaves the Mac; `data/` (uploads, cached analyses, sources registry) is git-ignored.

Start: double-click `Analyzer.command` (or `../training/.venv-arm64/bin/python -m analyzer serve`), then open
http://127.0.0.1:9200.

CLI (from this folder, same python):

    python -m analyzer analyze <files or folders…> [--out ../reports/analyzer/<ts>] [--category ai --generator suno]
    python -m analyzer sources add <paths…> --category human --artist "X" [--sidecar]
    python -m analyzer sources list
    python -m analyzer retrain --trainer local|modal     # through the retrain service on :9100

Tests: `../training/.venv-arm64/bin/python -m pytest tests` (model-dependent checks are skipped when
`training/runs` has no installed run; set `ANALYZER_REQUIRE_MODELS=1` to make that a failure).
