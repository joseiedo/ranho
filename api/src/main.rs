use mimalloc::MiMalloc;

// mimalloc replaces the system allocator for better allocation performance
// under low concurrency (single-threaded runtime).
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use api::search::SearchIndex;
use api::types::Label;
use api::vectorizer::Vectorizer;
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Router,
};

use bytes::Bytes;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use std::{collections::HashMap, sync::Arc};
use tower_service::Service;

// Precomputed static JSON responses indexed by fraud_count (0..=5).
// fraud_score = fraud_count / 5. Approved when fraud_score < 0.6 (i.e., ≤ 2 fraud).
//
// This eliminates all per-request serialization: we only select from 6 possible
// byte slices. The hot path computes fraud_count and uses it as an array index.
//
// | fraud_count | fraud_score | approved |
// |-------------|-------------|----------|
// | 0           | 0.0         | true     |
// | 1           | 0.2         | true     |
// | 2           | 0.4         | true     |
// | 3           | 0.6         | false    |
// | 4           | 0.8         | false    |
// | 5           | 1.0         | false    |
static FRAUD_RESPONSES: [&[u8]; 6] = [
    br#"{"approved":true,"fraud_score":0.0}"#,
    br#"{"approved":true,"fraud_score":0.2}"#,
    br#"{"approved":true,"fraud_score":0.4}"#,
    br#"{"approved":false,"fraud_score":0.6}"#,
    br#"{"approved":false,"fraud_score":0.8}"#,
    br#"{"approved":false,"fraud_score":1.0}"#,
];

struct AppState {
    vectorizer: Vectorizer,
    index: Option<SearchIndex>,
}

async fn ready() -> StatusCode {
    StatusCode::OK
}

// POST /fraud-score — the hot path.
//
// Pipeline (zero heap allocation after deserialization):
//   JSON body → TransactionPayload → [f32; 14] → [i16; 14] → IVF search → [Label; 5] → index into static JSON.
//
// Fallback behavior:
//   - Index failed to open → return approved (fraud_score: 0.0). Safest default.
//   - JSON parse error      → same fallback.
//   This is a design choice: prefer false negatives over false positives
//   when the system is in a degraded state.
//
// Note: scorer::score() exists but is not called here — the hot path computes
// fraud_count directly and indexes FRAUD_RESPONSES inline, avoiding the function
// call overhead and the intermediate (f32, bool) tuple.
// https://github.com/zanfranceschi/rinha-de-backend-2026/blob/main/docs/br/REGRAS_DE_DETECCAO.md
async fn fraud_score(State(state): State<Arc<AppState>>, body: Bytes) -> impl IntoResponse {
    const JSON: [(header::HeaderName, &str); 1] = [(header::CONTENT_TYPE, "application/json")];

    let Some(index) = &state.index else {
        return (JSON, FRAUD_RESPONSES[0]);
    };

    let payload = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(_) => {
            return (JSON, FRAUD_RESPONSES[0]);
        }
    };

    let vector = state.vectorizer.vectorize(&payload);

    // IVF-PQ approach: search with original vector, but use quantized version for distance calculations.
    // https://docs.rapids.ai/api/cuvs/nightly/neighbors/ivfpq/#ivf-pq
    let quantized = Vectorizer::quantize(&vector);
    let neighbors = index.search_with_vector(&vector, &quantized);

    let fraud_count = neighbors.iter().filter(|&&l| l == Label::Fraud).count();

    (JSON, FRAUD_RESPONSES[fraud_count])
}

fn main() {
    // Hardcoded MCC → fraud risk map. Unknown MCCs default to 0.5.
    // These values are domain knowledge about merchant category fraud patterns:
    // grocery/drug stores have low fraud rates, gambling/betting have high rates.
    let mcc_risk: HashMap<String, f32> = [
        ("5411", 0.15),
        ("5812", 0.30),
        ("5912", 0.20),
        ("5944", 0.45),
        ("7801", 0.80),
        ("7802", 0.75),
        ("7995", 0.85),
        ("4511", 0.35),
        ("5311", 0.25),
        ("5999", 0.50),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();

    let index_path =
        std::env::var("INDEX_PATH").unwrap_or_else(|_| "./resources/index.bin".to_string());

    // Open the IVF index. If it fails (missing file, invalid magic, wrong dims),
    // the service still starts but returns the safe fallback (approved) for all requests.
    let index = match SearchIndex::open(&index_path) {
        Ok(idx) => Some(idx),
        Err(_) => None,
    };

    // Force page faults and warm up the CPU cache/branch predictor.
    if let Some(ref idx) = index {
        idx.warmup();
    }

    let vectorizer = Vectorizer::new(mcc_risk);
    let state = Arc::new(AppState { vectorizer, index });

    // Single-threaded Tokio runtime: one OS thread serves all requests sequentially.
    // Rationale: under the challenge's 0.4 CPU budget per instance, a single thread
    // avoids context-switch overhead and keeps the working set in L1/L2 cache.
    // Connections are accepted in an accept loop with tokio::spawn, but the executor
    // is current_thread, so only one task runs at a time (cooperative multitasking).
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async move {
            let app = Router::new()
                .route("/ready", get(ready))
                .route("/fraud-score", post(fraud_score))
                .with_state(state);

            let socket_path =
                std::env::var("SOCKET_PATH").unwrap_or_else(|_| "/tmp/api.sock".to_string());

            // Unix socket behind nginx: avoids TCP stack overhead for local communication.
            let _ = std::fs::remove_file(&socket_path);
            let listener = tokio::net::UnixListener::bind(&socket_path).unwrap();
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o777)).unwrap();
            let builder = Builder::new(TokioExecutor::new());

            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let io = TokioIo::new(stream);
                let app = app.clone();
                let builder = builder.clone();
                tokio::spawn(async move {
                    builder
                        .serve_connection(
                            io,
                            hyper::service::service_fn(move |req| app.clone().call(req)),
                        )
                        .await
                        .ok();
                });
            }
        });
}
