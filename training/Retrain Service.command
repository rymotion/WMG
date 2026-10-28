#!/bin/zsh
# Double-click (or leave running) so the macOS app's "Retrain models" button can reach the
# trainer. Training needs the Mac's GPU, so it runs here rather than in the Docker backend.
cd "$(dirname "$0")"
export LANG=en_US.UTF-8
export PATH="$HOME/SDKs/flutter/bin:$PATH"
.venv-arm64/bin/python -m wmg_train.retrain serve "$@"
