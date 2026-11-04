#!/bin/zsh
# Double-click (or leave running) so the macOS app's "Retrain models" button can reach the
# trainer. Training uses the Mac's GPU by default and runs here rather than in the Docker backend;
# with WMG_TRAINER=modal (or `--trainer modal`) the two models train on a Modal GPU instead
# (one-time: .venv-arm64/bin/modal setup && .venv-arm64/bin/modal deploy modal_app.py).
cd "$(dirname "$0")"
export LANG=en_US.UTF-8
export PATH="$HOME/SDKs/flutter/bin:$PATH"
.venv-arm64/bin/python -m wmg_train.retrain serve "$@"
