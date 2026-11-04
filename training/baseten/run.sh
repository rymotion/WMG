#!/bin/sh
# Start command of the Baseten training job (baseten/config.py). Runs in the job's workdir,
# which holds this folder plus wmg_train/, configs/ and any caches shipped with the job.
set -eu

# Same versions as training/.venv-arm64 and modal_app.py so cloud and Mac metrics compare.
# pip's cache lives in the project cache, so only the first job of a project downloads torch.
export PIP_CACHE_DIR="${BT_PROJECT_CACHE_DIR:-/root/.cache/user_artifacts}/pip"
pip install --quiet --disable-pip-version-check \
    torch==2.14.1 numpy==2.5.4 pandas==3.0.6 pyarrow==26.0.0 scikit-learn==1.9.1 \
    pyyaml==6.0.3 onnx==1.23.2

exec python -m wmg_train.baseten_cloud job
