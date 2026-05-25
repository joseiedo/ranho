use api::search::SearchIndex;
use api::types::TransactionPayload;
use api::vectorizer::Vectorizer;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const DEFAULT_QUERY_LIMIT: usize = 4096;
const DEFAULT_WARMUP_PASSES: usize = 3;
const DEFAULT_MEASURE_PASSES: usize = 10;

#[derive(Deserialize)]
struct TestDataFile {
    entries: Vec<TestDataEntry>,
}

#[derive(Deserialize)]
struct TestDataEntry {
    request: TransactionPayload,
}

#[derive(Clone, Copy)]
struct Query {
    vector: [f32; 14],
    quantized: [i16; 14],
}

struct LoadedBenchData {
    index: SearchIndex,
    queries: Vec<Query>,
}

fn main() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("api crate should live under the workspace root");

    let index_path = resolve_index_path(workspace_root);
    let test_data_path = workspace_root.join("spec/test/test-data.json");
    let mcc_risk_path = workspace_root.join("spec/resources/mcc_risk.json");

    let query_limit = parse_env_usize("BENCH_QUERY_LIMIT", DEFAULT_QUERY_LIMIT);
    let warmup_passes = parse_env_usize("BENCH_WARMUP_PASSES", DEFAULT_WARMUP_PASSES);
    let measure_passes = parse_env_usize("BENCH_MEASURE_PASSES", DEFAULT_MEASURE_PASSES);

    eprintln!("search_bench: index={}", index_path.display());
    eprintln!("search_bench: test_data={}", test_data_path.display());
    eprintln!(
        "search_bench: query_limit={query_limit} warmup_passes={warmup_passes} measure_passes={measure_passes}"
    );

    let loaded = load_bench_data(&index_path, &test_data_path, &mcc_risk_path, query_limit);

    eprintln!(
        "search_bench: loaded {} queries against {} indexed vectors",
        loaded.queries.len(),
        loaded.index.count()
    );

    loaded.index.warmup();

    run_benchmark(
        "search_with_vector",
        &loaded.queries,
        warmup_passes,
        measure_passes,
        |query| {
            std::hint::black_box(
                loaded
                    .index
                    .search_with_vector(&query.vector, &query.quantized),
            );
        },
    );

    run_benchmark(
        "search",
        &loaded.queries,
        warmup_passes,
        measure_passes,
        |query| {
            std::hint::black_box(loaded.index.search(&query.quantized));
        },
    );
}

fn run_benchmark<F>(
    name: &str,
    queries: &[Query],
    warmup_passes: usize,
    measure_passes: usize,
    mut run_query: F,
) where
    F: FnMut(&Query),
{
    for _ in 0..warmup_passes {
        for query in queries {
            run_query(query);
        }
    }

    let started_at = Instant::now();
    for _ in 0..measure_passes {
        for query in queries {
            run_query(query);
        }
    }
    let elapsed = started_at.elapsed();
    print_summary(name, elapsed, queries.len(), measure_passes);
}

fn print_summary(name: &str, elapsed: Duration, query_count: usize, passes: usize) {
    let total_queries = query_count * passes;
    let elapsed_secs = elapsed.as_secs_f64();
    let queries_per_sec = total_queries as f64 / elapsed_secs;
    let nanos_per_query = elapsed.as_nanos() as f64 / total_queries as f64;

    println!(
        "{name}: total_queries={total_queries} elapsed={elapsed_secs:.6}s qps={queries_per_sec:.2} ns_per_query={nanos_per_query:.2}"
    );
}

fn load_bench_data(
    index_path: &Path,
    test_data_path: &Path,
    mcc_risk_path: &Path,
    query_limit: usize,
) -> LoadedBenchData {
    let mcc_risk: HashMap<String, f32> = serde_json::from_reader(
        File::open(mcc_risk_path).unwrap_or_else(|err| {
            panic!("failed to open {}: {err}", mcc_risk_path.display())
        }),
    )
    .unwrap_or_else(|err| panic!("failed to parse {}: {err}", mcc_risk_path.display()));

    let test_data: TestDataFile = serde_json::from_reader(
        File::open(test_data_path).unwrap_or_else(|err| {
            panic!("failed to open {}: {err}", test_data_path.display())
        }),
    )
    .unwrap_or_else(|err| panic!("failed to parse {}: {err}", test_data_path.display()));

    let vectorizer = Vectorizer::new(mcc_risk);
    let query_count = query_limit.min(test_data.entries.len());
    let mut queries = Vec::with_capacity(query_count);

    for entry in test_data.entries.into_iter().take(query_count) {
        let vector = vectorizer.vectorize(&entry.request);
        let quantized = Vectorizer::quantize(&vector);
        queries.push(Query { vector, quantized });
    }

    let index = SearchIndex::open(index_path.to_str().expect("index path should be valid UTF-8"))
        .unwrap_or_else(|err| panic!("failed to open {}: {err}", index_path.display()));

    LoadedBenchData { index, queries }
}

fn resolve_index_path(workspace_root: &Path) -> PathBuf {
    if let Ok(path) = std::env::var("INDEX_PATH") {
        return PathBuf::from(path);
    }

    let candidates = [
        workspace_root.join("resources/index.bin"),
        workspace_root.join("spec/resources/index.bin"),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return candidate;
        }
    }

    panic!(
        "index not found. Set INDEX_PATH or build one first, for example:\n  cargo run --release -p preprocessor -- ./spec/resources/references.json.gz ./resources/index.bin"
    );
}

fn parse_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
