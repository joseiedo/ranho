# syntax=docker/dockerfile:1
# Platform is controlled by the caller:
#   local (aarch64):  docker buildx build -t rinha-2026:local .
#   submission:       docker buildx build --platform linux/amd64 \
#                       --build-arg API_RUSTFLAGS="-C target-cpu=haswell" \
#                       -t ghcr.io/iedo/rinha-2026:latest --push .

# ── Stage 1: compile ──────────────────────────────────────────────────────────
FROM rust:1.86-slim AS builder

# API_RUSTFLAGS is empty by default (native build). Set to "-C target-cpu=haswell"
# when building for the submission (linux/amd64 + Haswell evaluation machine).
ARG API_RUSTFLAGS=""

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY api ./api
COPY preprocessor ./preprocessor

# preprocessor has no SIMD dependency — compile without extra flags.
RUN cargo build --release -p preprocessor
# API gets the optional SIMD flags.
RUN RUSTFLAGS="${API_RUSTFLAGS}" cargo build --release -p api

# ── Stage 2: run preprocessor to produce binary index ────────────────────────
FROM debian:bookworm-slim AS indexer

WORKDIR /data
COPY --from=builder /build/target/release/preprocessor ./
# references.json.gz must be present in the build context under resources/
COPY resources/references.json.gz ./

RUN ./preprocessor references.json.gz index.bin

# ── Stage 3: minimal runtime image ───────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder  /build/target/release/api ./
COPY --from=indexer  /data/index.bin            ./resources/
COPY resources/mcc_risk.json                    ./resources/

ENV INDEX_PATH=/app/resources/index.bin
ENV MCC_RISK_PATH=/app/resources/mcc_risk.json

CMD ["./api"]
