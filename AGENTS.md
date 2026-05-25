# AGENTS.md

This file provides guidance to coding agents working in this repository.

## Commands

```bash
# Run all tests
cargo test

# Run tests for the API crate
cargo test -p api

# Run a single test by name
cargo test -p api dim0_normal

# Lint
cargo clippy --workspace --all-targets

# Build release binaries
cargo build --release

# Build the API with Haswell tuning for the submission target
RUSTFLAGS="-C target-cpu=haswell" cargo build --release -p api

# Build the IVF index locally
cargo run --release -p preprocessor -- \
  ./spec/resources/references.json.gz \
  ./resources/index.bin

# Run the API locally over a Unix socket
INDEX_PATH=./resources/index.bin \
SOCKET_PATH=/tmp/api.sock \
cargo run --release -p api

# Build local container image
docker buildx build -t rinha-2026:local .

# Start local stack
docker compose up -d

# Submission image (linux/amd64 + Haswell)
docker buildx build --platform linux/amd64 \
  --build-arg API_RUSTFLAGS="-C target-cpu=haswell" \
  -t ghcr.io/joseiedo/rinha-de-backend-2026:latest --push .
```

## Architecture

This repository is a Rinha de Backend 2026 submission for fraud detection. The API receives a transaction payload, converts it to a 14-dimensional vector, finds the 5 nearest labeled reference vectors, and returns a fraud score plus approval decision.

Workspace crates:
- `api` - production HTTP service built with Axum, served over a Unix socket behind nginx
- `preprocessor` - offline index builder that converts `references.json.gz` into the binary IVF index consumed by `api`

Request pipeline:

```text
POST /fraud-score
  -> parse request body
  -> vectorize payload into [f32; 14]
  -> quantize into [i16; 14] with scale 10_000
  -> probe IVF centroids and scan candidate vectors
  -> count fraud labels among top-5 neighbors
  -> return one of 6 precomputed JSON responses
```

## Module Responsibilities

- `api/src/main.rs` - boots the single-thread Tokio runtime, opens the index, warms it up, binds the Unix socket, and serves `POST /fraud-score` plus `GET /ready`
- `api/src/vectorizer.rs` - converts request payloads into the 14-dimension feature vector
- `api/src/normalization.rs` - normalizes raw request values into the expected numeric ranges used by the vectorizer
- `api/src/search.rs` - memory-maps the IVF index, selects centroids, scans candidate vectors, and returns the top-5 labels
- `api/src/scorer.rs` - score helpers and approval logic
- `api/src/types.rs` - request and response types plus the `Label` enum
- `preprocessor/src/main.rs` - parses reference data, trains quantized IVF centroids, assigns all vectors to clusters, and writes the binary index

## Important Implementation Details

- Quantization uses `i16` with scale `10_000.0`, not `i8`
- The IVF file magic is `RINHIVF4`
- Stored centroid stride is 16 lanes: 14 dimensions plus 2 padding lanes
- The IVF payload layout is:
  - header: magic, vector count, cluster count, dims, stride
  - centroid table as `float32`
  - per-cluster bbox min/max as `i16`
  - cumulative cluster offsets
  - cluster-sorted vectors in column-major `i16` layout
  - cluster-sorted labels
- The search path probes up to 8 clusters first, prunes with bbox lower bounds, and expands to up to 100 clusters when the partial top-5 remains ambiguous with 2 or 3 fraud neighbors
- Responses are precomputed static JSON byte slices to avoid per-request serialization
- If the index fails to open or the request body is invalid, the handler returns the safest fallback response: `{"approved":true,"fraud_score":0.0}`
- The API intentionally runs on a current-thread Tokio runtime to stay within the challenge resource budget
- The runtime allocator is `mimalloc`

## Data And Runtime Assumptions

- `spec/resources/references.json.gz` is the local reference dataset used to rebuild `resources/index.bin`
- The API expects `resources/index.bin` at runtime unless `INDEX_PATH` overrides it
- `SOCKET_PATH` defaults to `/tmp/api.sock`
- `docker-compose.yml` budgets the stack to 1.0 CPU and 350 MB total:
  - `nginx`: 0.2 CPU / 30 MB
  - `api1`: 0.4 CPU / 160 MB
  - `api2`: 0.4 CPU / 160 MB

## Practical Guidance For Agents

- Prefer changes that preserve the no-allocation hot path in `api/src/main.rs` and `api/src/search.rs`
- Treat `api/src/search.rs` as performance-sensitive code; avoid unnecessary bounds checks, allocations, or format conversions in the request path
- Keep `preprocessor/src/main.rs` and `api/src/search.rs` in lockstep when changing centroid stride, bbox encoding, offsets, or column-major data layout
- Keep fallback behavior stable unless the user explicitly wants to change challenge strategy
- If you change vector semantics, update both API tests and preprocessor assumptions
- If you change the binary index layout, update both `preprocessor/src/main.rs` and `api/src/search.rs` together
