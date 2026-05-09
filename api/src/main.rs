use api::search::SearchIndex;
use api::types::FraudResponse;
use api::vectorizer::Vectorizer;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use std::{collections::HashMap, sync::Arc};

struct AppState {
    vectorizer: Vectorizer,
    index: Option<SearchIndex>,
}

async fn ready(State(state): State<Arc<AppState>>) -> StatusCode {
    if state.index.is_some() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn fraud_score(State(state): State<Arc<AppState>>, body: Bytes) -> Json<FraudResponse> {
    let fallback = FraudResponse { approved: true, fraud_score: 0.0 };

    let Some(index) = &state.index else {
        return Json(fallback);
    };

    let payload = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(_) => return Json(fallback),
    };

    let vector = state.vectorizer.vectorize(&payload);
    let quantized = Vectorizer::quantize(&vector);
    let neighbors = index.search(&quantized);
    let (fraud_score, approved) = api::scorer::score(neighbors);

    Json(FraudResponse { approved, fraud_score })
}

#[tokio::main]
async fn main() {
    let mcc_risk_path = std::env::var("MCC_RISK_PATH")
        .unwrap_or_else(|_| "./resources/mcc_risk.json".to_string());

    let mcc_risk: HashMap<String, f32> = std::fs::read_to_string(&mcc_risk_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    let index_path = std::env::var("INDEX_PATH")
        .unwrap_or_else(|_| "./resources/index.bin".to_string());

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

    let state = Arc::new(AppState {
        vectorizer: Vectorizer::new(mcc_risk),
        index,
    });

    let app = Router::new()
        .route("/ready", get(ready))
        .route("/fraud-score", post(fraud_score))
        .with_state(state);

    let addr = "0.0.0.0:9999";
    eprintln!("listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
