#!/bin/zsh
# Double-click to retrain both detector models from the app's labelled recordings and, if they
# are not worse, install them and rebuild the macOS app. Worse models wait for your y/N.
# Needs the backend running (it holds the recordings' audio): docker compose up -d in backend/.
cd "$(dirname "$0")"
export LANG=en_US.UTF-8
.venv-arm64/bin/python -m wmg_train.retrain run --build-app "$@"
echo
read "?Press Return to close."
