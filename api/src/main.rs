use mimalloc::MiMalloc;

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
use std::{collections::HashMap, sync::Arc};

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
    let quantized = Vectorizer::quantize(&vector);
    let neighbors = index.search_with_vector(&vector, &quantized);
    let fraud_count = neighbors.iter().filter(|&&l| l == Label::Fraud).count();

    (JSON, FRAUD_RESPONSES[fraud_count])
}

fn main() {
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

    let index = match SearchIndex::open(&index_path) {
        Ok(idx) => Some(idx),
        Err(_) => None,
    };

    if let Some(ref idx) = index {
        idx.warmup();
    }

    let vectorizer = Vectorizer::new(mcc_risk);
    let state = Arc::new(AppState { vectorizer, index });

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async move {
            let app = Router::new()
                .route("/ready", get(ready))
                .route("/fraud-score", post(fraud_score))
                .with_state(state);

            let socket_path = std::env::var("SOCKET_PATH")
                .unwrap_or_else(|_| "/tmp/api.sock".to_string());

            let _ = std::fs::remove_file(&socket_path);
            let listener = tokio::net::UnixListener::bind(&socket_path).unwrap();
            axum::serve(listener, app).await.unwrap();
        });
}
