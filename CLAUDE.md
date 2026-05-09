# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
cargo test

# Run tests for a single crate
cargo test -p api

# Run a single test by name
cargo test -p api dim0_normal

# Build release binary (with AVX2 for Haswell)
RUSTFLAGS="-C target-cpu=haswell" cargo build --release

# Run the API locally (needs mcc_risk.json and index.bin)
MCC_RISK_PATH=../rinha-de-backend-2026/resources/mcc_risk.json \
INDEX_PATH=../rinha-de-backend-2026/resources/index.bin \
cargo run -p api

# Search benchmark (against full 3M index)
cargo bench --bench search_bench

# Lint
cargo clippy

# ── Docker (local dev — native aarch64) ───────────────────────────────────
# Prerequisite: copy references.json.gz into resources/ before building.
cp ../rinha-de-backend-2026/resources/references.json.gz resources/

# Build native image and start all containers
docker buildx build -t rinha-2026:local .
docker compose up -d

# ── Docker (submission — linux/amd64 + Haswell AVX2) ──────────────────────
docker buildx build --platform linux/amd64 \
  --build-arg API_RUSTFLAGS="-C target-cpu=haswell" \
  -t ghcr.io/iedo/rinha-2026:latest --push .

# On the submission branch: replace docker-compose.yml with docker-compose.submission.yml
# git checkout submission
# cp docker-compose.submission.yml docker-compose.yml
# cp nginx.conf info.json .
# git add docker-compose.yml nginx.conf info.json && git commit && git push
```

## Architecture

This is a Rinha de Backend 2026 submission — a fraud detection API. The challenge: find the 5 nearest neighbors of an incoming transaction vector among 3 million pre-labeled reference vectors, return a fraud score and approval decision. Full challenge spec is in `../rinha-de-backend-2026/`.

**Workspace crates:**
- `api` — axum HTTP server, runs in production
- `preprocessor` — build-time binary that converts `references.json.gz` → flat binary index (Phase 2+)

**Request pipeline (api):**
```
POST /fraud-score → parse JSON → vectorizer → quantize → search → scorer → JSON response
```

**Module responsibilities:**
- `types.rs` — all serde structs for the HTTP request/response plus the `Label` enum (`Legit`/`Fraud`)
- `vectorizer.rs` — converts a `TransactionPayload` into a `[f32; 14]` vector, then `Vectorizer::quantize()` produces `[i8; 14]`. The 14-dimension formula is spec-defined; tests cover every dimension individually.
- `search.rs` — currently a stub returning `[Label::Legit; 5]`. Phase 2 replaces this with a real brute-force KNN over the mmap'd binary index.
- `scorer.rs` — `fraud_score = fraud_count / 5.0`; `approved = fraud_score < 0.6`

**Key design decisions:**
- The handler never returns HTTP 5xx. Any error falls back to `{ approved: true, fraud_score: 0.0 }` — this avoids the 5× scoring penalty for HTTP errors.
- int8 quantization: `(v * 127.0).round().clamp(-127.0, 127.0) as i8`. The sentinel `-1.0` (used when `last_transaction` is null, dims 5 and 6) maps naturally to `-127` through this formula.
- The vectorizer is dependency-injected with the MCC risk map (`HashMap<String, f32>`) making it straightforward to test without touching the filesystem.
- Production build requires `RUSTFLAGS="-C target-cpu=haswell"` for AVX2 auto-vectorization in the search loop (the test machine is a Mac Mini Late 2014, Haswell).

**Resource budget (docker-compose):** 1.0 CPU + 350 MB RAM total across nginx (0.1 CPU / 30 MB) + 2× API instances (0.45 CPU / 155 MB each).
