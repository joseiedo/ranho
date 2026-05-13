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

// ── App state ─────────────────────────────────────────────────────────────────

struct AppState {
    vectorizer: Vectorizer,
    index: Option<SearchIndex>,
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn ready(State(state): State<Arc<AppState>>) -> StatusCode {
    if state.index.is_some() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn fraud_score(State(state): State<Arc<AppState>>, body: Bytes) -> impl IntoResponse {
    const JSON: [(header::HeaderName, &str); 1] = [(header::CONTENT_TYPE, "application/json")];

    let Some(index) = &state.index else {
        return (JSON, FRAUD_RESPONSES[0]);
    };

    let payload = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(_) => return (JSON, FRAUD_RESPONSES[0]),
    };

    let vector = state.vectorizer.vectorize(&payload);
    let quantized = Vectorizer::quantize(&vector);
    let neighbors = index.search(&quantized);
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
    let mcc_risk_path =
        std::env::var("MCC_RISK_PATH").unwrap_or_else(|_| "./resources/mcc_risk.json".to_string());

    eprintln!("loading mcc_risk from {mcc_risk_path}");
    let mcc_risk: HashMap<String, f32> = std::fs::read_to_string(&mcc_risk_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    eprintln!("mcc_risk loaded: {} entries", mcc_risk.len());

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

    let socket_path = std::env::var("SOCKET_PATH").unwrap_or_else(|_| "/tmp/api.sock".to_string());

    eprintln!("opening socket at {socket_path}");
    let _ = std::fs::remove_file(&socket_path);
    if let Some(parent) = std::path::Path::new(&socket_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let listener = tokio::net::UnixListener::bind(&socket_path).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o777)).ok();
    }
    eprintln!("listening on unix:{socket_path}");
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
