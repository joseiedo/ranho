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
use hyper::body::Incoming;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use std::{collections::HashMap, convert::Infallible, sync::Arc};
use tower_service::Service;

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

async fn ready(State(state): State<Arc<AppState>>) -> StatusCode {
    if state.index.is_some() {
        eprintln!("[ready] 200 OK");
        StatusCode::OK
    } else {
        eprintln!("[ready] 503 index not loaded");
        StatusCode::SERVICE_UNAVAILABLE
    }
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

fn unwrap_infallible<T>(result: Result<T, Infallible>) -> T {
    match result {
        Ok(v) => v,
        Err(e) => match e {},
    }
}

#[tokio::main(worker_threads = 1)]
async fn main() {
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
            eprintln!("warming up...");
            idx.warmup();
            eprintln!("warmup done");
            Some(idx)
        }
        Err(e) => {
            eprintln!("warning: could not load index ({e}); /ready will return 503");
            None
        }
    };

    let vectorizer = Vectorizer::new(mcc_risk);

    let state = Arc::new(AppState { vectorizer, index });

    let app = Router::new()
        .route("/ready", get(ready))
        .route("/fraud-score", post(fraud_score))
        .with_state(state);

    let addr = std::env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".to_string());

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    eprintln!("listening on http://{addr}");
    let mut make_service = app.into_make_service();

    loop {
        let (socket, _) = listener.accept().await.unwrap();
        let svc: axum::Router = unwrap_infallible(make_service.call(()).await);

        tokio::spawn(async move {
            let hyper_svc = hyper::service::service_fn(move |req: hyper::Request<Incoming>| {
                svc.clone().call(req)
            });
            Builder::new(TokioExecutor::new())
                .serve_connection(TokioIo::new(socket), hyper_svc)
                .await
                .ok();
        });
    }
}
