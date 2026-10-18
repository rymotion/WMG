# syntax=docker/dockerfile:1

# ---- deps: cache compiled dependencies with cargo-chef ----
FROM rust:1-slim-bookworm AS chef
RUN cargo install cargo-chef --locked
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release --locked --bin spectro-api

# ---- test: `docker build --target test .` ----
FROM builder AS test
RUN cargo test --release --locked

# ---- runtime ----
FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home app \
    && mkdir -p /data && chown app:app /data
COPY --from=builder /app/target/release/spectro-api /usr/local/bin/spectro-api
USER app
ENV PORT=9000 DATA_DIR=/data RUST_LOG=info
EXPOSE 9000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["spectro-api", "--healthcheck"]
CMD ["spectro-api"]
