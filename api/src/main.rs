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
use serde::Serialize;
use std::{
    collections::HashMap,
    convert::Infallible,
    sync::{
        atomic::{AtomicU64, Ordering::Relaxed},
        Arc,
    },
    time::Instant,
};
use tower_service::Service;

// ── Metrics ──────────────────────────────────────────────────────────────────

const BUCKETS_US: [u64; 5] = [200, 500, 1_000, 2_000, 5_000];

#[derive(Default)]
struct Histogram {
    counts: [AtomicU64; 6],
}

impl Histogram {
    fn record(&self, total_us: u64) {
        let i = BUCKETS_US.iter().position(|&b| total_us < b).unwrap_or(5);
        self.counts[i].fetch_add(1, Relaxed);
    }
}

#[derive(Default)]
struct StageMetrics {
    total_ns: AtomicU64,
    max_ns: AtomicU64,
    count: AtomicU64,
}

impl StageMetrics {
    fn record(&self, ns: u64) {
        self.total_ns.fetch_add(ns, Relaxed);
        self.count.fetch_add(1, Relaxed);
        let mut cur = self.max_ns.load(Relaxed);
        while ns > cur {
            match self.max_ns.compare_exchange_weak(cur, ns, Relaxed, Relaxed) {
                Ok(_) => break,
                Err(v) => cur = v,
            }
        }
    }
}

#[derive(Default)]
struct Metrics {
    requests: AtomicU64,
    parse: StageMetrics,
    vectorize: StageMetrics,
    quantize: StageMetrics,
    centroid: StageMetrics,
    scan_fast: StageMetrics,
    scan_full: StageMetrics,
    score: StageMetrics,
    total: StageMetrics,
    hist: Histogram,
}

fn avg_us(m: &StageMetrics) -> f64 {
    let count = m.count.load(Relaxed);
    if count == 0 {
        return 0.0;
    }
    m.total_ns.load(Relaxed) as f64 / count as f64 / 1000.0
}

fn max_us(m: &StageMetrics) -> f64 {
    m.max_ns.load(Relaxed) as f64 / 1000.0
}

#[derive(Serialize)]
struct StageStat {
    avg_us: f64,
    max_us: f64,
}

#[derive(Serialize)]
struct MetricsResponse {
    requests: u64,
    hist: HistResponse,
    stages: StagesResponse,
}

#[derive(Serialize)]
struct HistResponse {
    lt_200us: u64,
    lt_500us: u64,
    lt_1ms: u64,
    lt_2ms: u64,
    lt_5ms: u64,
    gte_5ms: u64,
}

#[derive(Serialize)]
struct StagesResponse {
    parse: StageStat,
    vectorize: StageStat,
    quantize: StageStat,
    centroid: StageStat,
    scan_fast: StageStat,
    scan_full: StageStat,
    score: StageStat,
    total: StageStat,
}

// ── App state ─────────────────────────────────────────────────────────────────

struct AppState {
    vectorizer: Vectorizer,
    index: Option<SearchIndex>,
    metrics: Metrics,
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn ready(State(state): State<Arc<AppState>>) -> StatusCode {
    if state.index.is_some() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn metrics(State(state): State<Arc<AppState>>) -> Json<MetricsResponse> {
    let m = &state.metrics;
    let h = &m.hist.counts;
    Json(MetricsResponse {
        requests: m.requests.load(Relaxed),
        hist: HistResponse {
            lt_200us: h[0].load(Relaxed),
            lt_500us: h[1].load(Relaxed),
            lt_1ms:   h[2].load(Relaxed),
            lt_2ms:   h[3].load(Relaxed),
            lt_5ms:   h[4].load(Relaxed),
            gte_5ms:  h[5].load(Relaxed),
        },
        stages: StagesResponse {
            parse:     StageStat { avg_us: avg_us(&m.parse),     max_us: max_us(&m.parse) },
            vectorize: StageStat { avg_us: avg_us(&m.vectorize), max_us: max_us(&m.vectorize) },
            quantize:  StageStat { avg_us: avg_us(&m.quantize),  max_us: max_us(&m.quantize) },
            centroid:  StageStat { avg_us: avg_us(&m.centroid),  max_us: max_us(&m.centroid) },
            scan_fast: StageStat { avg_us: avg_us(&m.scan_fast), max_us: max_us(&m.scan_fast) },
            scan_full: StageStat { avg_us: avg_us(&m.scan_full), max_us: max_us(&m.scan_full) },
            score:     StageStat { avg_us: avg_us(&m.score),     max_us: max_us(&m.score) },
            total:     StageStat { avg_us: avg_us(&m.total),     max_us: max_us(&m.total) },
        },
    })
}

async fn fraud_score(State(state): State<Arc<AppState>>, body: Bytes) -> Json<FraudResponse> {
    let fallback = FraudResponse { approved: true, fraud_score: 0.0 };

    let Some(index) = &state.index else {
        return Json(fallback);
    };

    let t_total = Instant::now();

    let t = Instant::now();
    let payload = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(_) => return Json(fallback),
    };
    let parse_ns = t.elapsed().as_nanos() as u64;
    state.metrics.parse.record(parse_ns);

    let t = Instant::now();
    let vector = state.vectorizer.vectorize(&payload);
    state.metrics.vectorize.record(t.elapsed().as_nanos() as u64);

    let t = Instant::now();
    let quantized = Vectorizer::quantize(&vector);
    state.metrics.quantize.record(t.elapsed().as_nanos() as u64);

    let (neighbors, timings) = index.search(&quantized);

    state.metrics.centroid.record(timings.centroid_ns);
    state.metrics.scan_fast.record(timings.scan_fast_ns);
    if timings.scan_full_ns > 0 {
        state.metrics.scan_full.record(timings.scan_full_ns);
    }

    let t = Instant::now();
    let (fraud_score, approved) = api::scorer::score(neighbors);
    state.metrics.score.record(t.elapsed().as_nanos() as u64);

    let total_ns = t_total.elapsed().as_nanos() as u64;
    state.metrics.total.record(total_ns);
    state.metrics.hist.record(total_ns / 1_000);
    state.metrics.requests.fetch_add(1, Relaxed);

    if total_ns > 1_000_000 {
        eprintln!(
            "SLOW {}µs | parse={}µs centroid={}µs scan_fast={}µs scan_full={}µs",
            total_ns / 1_000,
            parse_ns / 1_000,
            timings.centroid_ns / 1_000,
            timings.scan_fast_ns / 1_000,
            timings.scan_full_ns / 1_000,
        );
    }

    Json(FraudResponse { approved, fraud_score })
}

fn unwrap_infallible<T>(result: Result<T, Infallible>) -> T {
    match result {
        Ok(v) => v,
        Err(e) => match e {},
    }
}

#[tokio::main(worker_threads = 2)]
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
        metrics: Metrics::default(),
    });

    let app = Router::new()
        .route("/ready", get(ready))
        .route("/fraud-score", post(fraud_score))
        .route("/metrics", get(metrics))
        .with_state(state);

    let socket_path = std::env::var("SOCKET_PATH")
        .unwrap_or_else(|_| "/tmp/api.sock".to_string());

    let _ = std::fs::remove_file(&socket_path);
    if let Some(parent) = std::path::Path::new(&socket_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    eprintln!("listening on unix:{socket_path}");
    let listener = tokio::net::UnixListener::bind(&socket_path).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o777)).ok();
    }
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
