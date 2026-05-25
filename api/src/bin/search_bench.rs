use api::search::SearchIndex;
use std::time::Instant;

#[derive(Clone)]
struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }
}

fn percentile(sorted: &[u128], numerator: usize, denominator: usize) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) * numerator) / denominator;
    sorted[idx]
}

fn main() {
    let index_path = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("INDEX_PATH").ok())
        .unwrap_or_else(|| "./resources/index.bin".to_string());
    let queries = std::env::var("BENCH_QUERIES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10_000usize);
    let warmup = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1_000usize);

    let index = SearchIndex::open(&index_path).expect("failed to open index");
    if index.count() == 0 {
        eprintln!("bench: index is empty");
        return;
    }

    let mut rng = Lcg64::new(0xd7f151ea5eed);
    for _ in 0..warmup {
        let idx = (rng.next_u64() as usize) % index.count();
        let query = index.vector_at(idx).expect("warmup query out of bounds");
        let _ = index.search(&query);
    }

    let mut latencies = Vec::with_capacity(queries);
    let mut centroid_total = 0u128;
    let mut scan_total = 0u128;
    let mut scanned_vectors_total = 0usize;
    let mut scanned_clusters_total = 0usize;
    let mut pruned_clusters_total = 0usize;

    for _ in 0..queries {
        let idx = (rng.next_u64() as usize) % index.count();
        let mut query = index.vector_at(idx).expect("query out of bounds");
        let delta = (((rng.next_u64() >> 8) % 101) as i16) - 50;
        query[0] = query[0].saturating_add(delta);

        let started = Instant::now();
        let (_, metrics) = index.search_with_metrics(&query);
        let total = started.elapsed().as_nanos().max(metrics.total_time_ns());
        latencies.push(total);
        centroid_total += metrics.centroid_time_ns;
        scan_total += metrics.scan_time_ns;
        scanned_vectors_total += metrics.scanned_vectors;
        scanned_clusters_total += metrics.scanned_clusters;
        pruned_clusters_total += metrics.pruned_clusters;
    }

    latencies.sort_unstable();
    let avg_total_ns = latencies.iter().sum::<u128>() / queries as u128;
    let avg_centroid_ns = centroid_total / queries as u128;
    let avg_scan_ns = scan_total / queries as u128;
    let avg_vectors = scanned_vectors_total as f64 / queries as f64;
    let avg_scanned_clusters = scanned_clusters_total as f64 / queries as f64;
    let avg_pruned_clusters = pruned_clusters_total as f64 / queries as f64;

    println!("index={index_path}");
    println!("queries={queries}");
    println!("avg_total_us={:.3}", avg_total_ns as f64 / 1_000.0);
    println!("avg_centroid_us={:.3}", avg_centroid_ns as f64 / 1_000.0);
    println!("avg_scan_us={:.3}", avg_scan_ns as f64 / 1_000.0);
    println!("p95_us={:.3}", percentile(&latencies, 95, 100) as f64 / 1_000.0);
    println!("p99_us={:.3}", percentile(&latencies, 99, 100) as f64 / 1_000.0);
    println!("avg_scanned_vectors={avg_vectors:.2}");
    println!("avg_scanned_clusters={avg_scanned_clusters:.2}");
    println!("avg_pruned_clusters={avg_pruned_clusters:.2}");
}
