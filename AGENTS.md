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
  ./resources/references.json.gz \
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
  -> parse request body (serde_json::from_slice)
  -> vectorize payload into [f32; 14] (lookup tables for hour/dow, linear normalizations with clamp)
  -> quantize into [i16; 14] with scale 10_000.0
  -> probe IVF centroids (i64 distances, insertion-sort top-152)
  -> scan candidate vectors (AVX2 or scalar i64 distance, radius-based pruning)
  -> count fraud labels among top-5 neighbors
  -> return one of 6 precomputed static &[u8] JSON responses (indexed by fraud_count)
```

## Module Responsibilities

- `api/src/main.rs` - boots the single-thread Tokio runtime, opens the index, warms it up, binds the Unix socket, and serves `POST /fraud-score` plus `GET /ready`. Contains precomputed `FRAUD_RESPONSES` static array and the inline fraud-count-to-response logic.
- `api/src/vectorizer.rs` - converts request payloads into the 14-dimension feature vector with `clamp01()`. Provides `quantize()` for f32-to-i16 conversion. Contains `HOUR_LUT`, `DOW_LUT`, and normalization constants.
- `api/src/normalization.rs` - normalizes raw request values into the expected numeric ranges used by the vectorizer (linear scaling with saturation constants).
- `api/src/search.rs` - memory-maps the IVF index, selects centroids, scans candidate vectors via two-phase probe with radius-based pruning, and returns the top-5 labels. Contains both AVX2 and scalar distance paths with runtime detection.
- `api/src/scorer.rs` - score helpers and approval logic (`FRAUD_THRESHOLD = 0.6`). This module exists but the hot path in `main.rs` uses inline fraud-count logic instead of calling `scorer::score()`.
- `api/src/types.rs` - request and response types (`TransactionPayload`, `Transaction`, `Customer`, `Merchant`, `Terminal`, `LastTransaction`) plus the `Label` enum (`Legit`/`Fraud`).
- `preprocessor/src/main.rs` - parses reference data, runs sampled k-means with deterministic init (not k-means++), assigns all vectors to clusters via Rayon, computes per-cluster radii, and writes the binary index.

## Important Implementation Details

- Quantization uses `i16` with scale `10_000.0`; values in `[-1.0, 1.0]` map to `[-10000, 10000]`, sentinels at `-1.0` become `-10000`
- The IVF file magic is `RINHIVF6` (8 bytes)
- Index layout: `header(20) | centroids(k×14×2) | radii(k×4) | offsets((k+1)×4) | data(n×32) | labels(n×1)`
- `HEADER_SIZE = 20` (magic + k_clusters + n + dims, all u32 LE)
- Stored vector stride is 32 bytes: 14 `i16` values (28 bytes) plus 4 zero-padding bytes
- Per-cluster radii are `i32` (integer sqrt of max squared `i64` distance to centroid), used for cluster-level pruning
- Number of clusters: `NLIST = 2048` (min of target and actual reference count)
- K-means: deterministic init (evenly-spaced sample), 25 iterations max, sample size `131072`, parallel assignment via Rayon

### Two-Phase Probe Strategy

| Phase | Clusters Probed | Trigger |
|-------|----------------|---------|
| Fast | 8 (`NPROBE_FAST`) | Always |
| Slow | 144 additional (`NPROBE_RETRY`) | `fraud_count ∈ {2, 3}` from fast phase |

Total centroids sorted: 152 (`NPROBE_SLOW`). Top-152 nearest centroids are found via insertion-sort into a fixed-size `[(u64, usize); NPROBE_SLOW]` best array.

The slow phase uses **radius-based cluster pruning**: for each cluster, compute `lb = (isqrt(dist_to_centroid) - cluster_radius).max(0)^2`. Skip the cluster entirely if `lb >= worst_dist` (the current K-th best distance).

### Distance Computation

- All distances are computed in `i64` integer arithmetic (no `f32` in the hot path)
- **AVX2 path** (detected at runtime): loads query + record as `__m256i`, uses `_mm256_sub_epi16` + `_mm256_madd_epi16` + horizontal reduction via `_mm_hadd_epi32`
- **Scalar path**: iterates 14 dimensions, subtracts `i16` values, accumulates squared differences
- Query is padded to 16 `i16` elements for AVX2 256-bit alignment (`PADDED_DIMS = 16`)
- Integer square root: Newton's method (`isqrt`), avoids `libm` calls

### Precomputed Responses

Six static `&[u8]` slices at index `fraud_count` (0-5):

| fraud_count | fraud_score | approved |
|-------------|-------------|----------|
| 0 | 0.0 | true |
| 1 | 0.2 | true |
| 2 | 0.4 | true |
| 3 | 0.6 | false |
| 4 | 0.8 | false |
| 5 | 1.0 | false |

The handler in `main.rs` counts fraud labels inline and directly indexes `FRAUD_RESPONSES[fraud_count]`. The `scorer::score()` function is defined but unused in the hot path.

### Fallback Behavior

- If the index fails to open (`SearchIndex::open()` returns `Err`), `state.index` is `None`
- If `state.index` is `None`, or if JSON parsing fails, the handler returns `FRAUD_RESPONSES[0]` (`{"approved":true,"fraud_score":0.0}`)
- This is the safest fallback: approve by default when uncertain

### Runtime Configuration

- The API intentionally runs on a `current_thread` Tokio runtime (single OS thread)
- The runtime allocator is `mimalloc` (`#[global_allocator]`)
- Warmup: forces page faults by reading every 64th byte of the data section, then runs 500 dummy searches with pseudo-random query vectors (LCG)

### Release Profile

```toml
[profile.release]
lto = true
codegen-units = 1
panic = "abort"
opt-level = 3
overflow-checks = false
strip = true
```

## Data And Runtime Assumptions

- `resources/references.json.gz` must exist before building the Docker image because the image build runs the `preprocessor`
- The API expects `resources/index.bin` at runtime unless `INDEX_PATH` overrides it
- `SOCKET_PATH` defaults to `/tmp/api.sock`
- `docker-compose.yml` budgets the stack to 1.0 CPU and 350 MB total:
  - `nginx`: 0.2 CPU / 30 MB
  - `api1`: 0.4 CPU / 160 MB
  - `api2`: 0.4 CPU / 160 MB

## Practical Guidance For Agents

- Prefer changes that preserve the no-allocation hot path in `api/src/main.rs` and `api/src/search.rs`
- Treat `api/src/search.rs` as performance-sensitive code; avoid unnecessary bounds checks, allocations, or format conversions in the request path
- The hot path `fraud_count` computation in `main.rs` is inline; keep `scorer.rs` in sync if used
- Keep fallback behavior stable unless the user explicitly wants to change challenge strategy
- If you change vector semantics, update both API tests and preprocessor assumptions
- If you change the binary index layout, update both `preprocessor/src/main.rs` and `api/src/search.rs` together
- When changing probe counts (`NPROBE_FAST`, `NPROBE_RETRY`, `NPROBE_SLOW`), update both the constants in `search.rs` and the documentation here
- AVX2 runtime detection is automatic; do not gate AVX2 at compile time unless changing target platform assumptions
- The k-means initialization is deterministic (evenly-spaced), not k-means++; changing this affects index quality
