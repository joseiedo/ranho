use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

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
use hyper::body::Incoming;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use std::{collections::HashMap, convert::Infallible, sync::Arc};
use tower_service::Service;

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

fn unwrap_infallible<T>(result: Result<T, Infallible>) -> T {
    match result {
        Ok(v) => v,
        Err(e) => match e {},
    }
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
            eprintln!("warming up...");
            idx.warmup();
            eprintln!("warm");
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

    // Default to /tmp for local dev; docker-compose sets the real path per instance.
    let socket_path = std::env::var("SOCKET_PATH")
        .unwrap_or_else(|_| "/tmp/api.sock".to_string());

    // Remove stale socket and ensure the parent directory exists.
    let _ = std::fs::remove_file(&socket_path);
    if let Some(parent) = std::path::Path::new(&socket_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    eprintln!("listening on unix:{socket_path}");
    let listener = tokio::net::UnixListener::bind(&socket_path).unwrap();
    // Allow nginx (different user) to connect: unix socket connect requires write permission.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o777)).ok();
    }
    let mut make_service = app.into_make_service();

    loop {
        let (socket, _) = listener.accept().await.unwrap();
        // IntoMakeService is always ready — no poll_ready needed.
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
