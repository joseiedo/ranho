use api::search::SearchIndex;
use criterion::{criterion_group, criterion_main, Criterion};

/// Path to the full 3M-vector binary index produced by the preprocessor.
/// Override with INDEX_BIN env var; skips the benchmark if the file is absent.
const DEFAULT_INDEX_PATH: &str =
    "../../rinha-de-backend-2026/resources/index.bin";

fn bench_single_query(c: &mut Criterion) {
    let path = std::env::var("INDEX_BIN")
        .unwrap_or_else(|_| DEFAULT_INDEX_PATH.to_string());

    let idx = match SearchIndex::open(&path) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("skipping search_bench: cannot open {path}: {e}");
            return;
        }
    };

    // A representative query vector (all zeros → quantized 0).
    let query = [0i8; 14];

    c.bench_function("search_3m_single_query", |b| {
        b.iter(|| {
            let _ = idx.search(std::hint::black_box(&query)).0;
        });
    });
}

criterion_group!(benches, bench_single_query);
criterion_main!(benches);
