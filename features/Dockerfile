# Builds and tests the wmg_features Python wheel for Linux.
#   docker build --build-arg WMG_FEATURES_COMMIT=$(git rev-parse HEAD) \
#     --target export -o type=local,dest=dist features/
# The training image copies the wheel from the `wheel` stage.

FROM rust:1-slim-bookworm AS wheel
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /venv \
 && /venv/bin/pip install --no-cache-dir "maturin>=1.8,<2" numpy
WORKDIR /src
COPY . .
ARG WMG_FEATURES_COMMIT=unknown
ENV WMG_FEATURES_COMMIT=${WMG_FEATURES_COMMIT}
RUN cargo test --release \
 && /venv/bin/maturin build --release -o /wheels \
 && /venv/bin/pip install --no-cache-dir /wheels/*.whl \
 && /venv/bin/python -I tests/test_python.py

FROM scratch AS export
COPY --from=wheel /wheels /
