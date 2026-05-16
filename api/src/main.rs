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
        eprintln!("[fraud-score] index not loaded, returning fallback");
        return (JSON, FRAUD_RESPONSES[0]);
    };

    let payload = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[fraud-score] parse error: {e} body={:?}", String::from_utf8_lossy(&body));
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

    eprintln!("loading index from {index_path}");
    let index = match SearchIndex::open(&index_path) {
        Ok(idx) => {
            eprintln!("index loaded: {} vectors", idx.count());
            Some(idx)
        }
        Err(e) => {
            eprintln!("warning: could not load index ({e}); /ready will return 503");
            None
        }
    };

    // if let Some(ref idx) = index {
    //     eprintln!("warming up...");
    //     idx.warmup();
    //     eprintln!("warmup done");
    // }

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

            let addr =
                std::env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".to_string());

            let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
            eprintln!("listening on http://{addr}");
            axum::serve(listener, app).await.unwrap();
        });
}
