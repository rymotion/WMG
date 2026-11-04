#!/bin/zsh
# Double-click to run the feature analyzer at http://127.0.0.1:9200 (uses training/.venv-arm64).
cd "$(dirname "$0")"
export LANG=en_US.UTF-8
PY=../training/.venv-arm64/bin/python
if [ ! -x "$PY" ]; then
  echo "missing $PY: create training/.venv-arm64 first (see training/README.md)"; read -k 1; exit 1
fi
"$PY" -c "import fastapi, uvicorn, jinja2, multipart" 2>/dev/null || \
  uv pip install --python "$PY" "fastapi>=0.115" "uvicorn[standard]>=0.30" "jinja2>=3.1" "python-multipart>=0.0.9"
mkdir -p data
(sleep 1.5; open "http://127.0.0.1:9200/") &
exec "$PY" -m analyzer serve "$@"
