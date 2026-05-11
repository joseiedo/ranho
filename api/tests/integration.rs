/// Full pipeline integration test against the official test-data.json.
///
/// Marked `#[ignore]` so it only runs when explicitly requested.
/// By default tests the first 500 entries; set INTEGRATION_LIMIT=0 for all 54K.
///
/// Requires the binary index built by the preprocessor.
///
/// Run:
///   INDEX_PATH=/path/to/index.bin \
///   MCC_RISK_PATH=/path/to/mcc_risk.json \
///   TEST_DATA_PATH=/path/to/test-data.json \
///   cargo test -p api --test integration --release -- --ignored --nocapture
use api::scorer::score;
use api::search::SearchIndex;
use api::vectorizer::Vectorizer;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct TestFile {
    entries: Vec<TestEntry>,
}

#[derive(Deserialize)]
struct TestEntry {
    request: serde_json::Value,
    expected_approved: bool,
    expected_fraud_score: f32,
}

#[test]
#[ignore]
fn full_pipeline_failure_rate_under_5pct() {
    let index_path = std::env::var("INDEX_PATH")
        .unwrap_or_else(|_| "../../rinha-de-backend-2026/resources/index.bin".to_string());

    if !std::path::Path::new(&index_path).exists() {
        eprintln!("SKIP: index not found at {index_path}; run the preprocessor first");
        return;
    }

    let mcc_risk_path = std::env::var("MCC_RISK_PATH")
        .unwrap_or_else(|_| "../../rinha-de-backend-2026/resources/mcc_risk.json".to_string());

    let test_data_path = std::env::var("TEST_DATA_PATH")
        .unwrap_or_else(|_| "../../rinha-de-backend-2026/test/test-data.json".to_string());

    // Default to 500 entries to keep test time manageable; set to 0 for all.
    let limit: usize = std::env::var("INTEGRATION_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    let mcc_risk: HashMap<String, f32> = std::fs::read_to_string(&mcc_risk_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    let test_file: TestFile = serde_json::from_reader(
        std::fs::File::open(&test_data_path).expect("test-data.json not found"),
    )
    .expect("failed to parse test-data.json");

    let index = SearchIndex::open(&index_path).expect("failed to open index");
    let vectorizer = Vectorizer::new(mcc_risk);

    let entries = if limit == 0 {
        &test_file.entries[..]
    } else {
        &test_file.entries[..limit.min(test_file.entries.len())]
    };

    let n = entries.len();
    let mut failures = 0usize;

    for entry in entries {
        let payload: api::types::TransactionPayload =
            serde_json::from_value(entry.request.clone()).expect("bad request payload");

        let vector = vectorizer.vectorize(&payload);
        let quantized = Vectorizer::quantize(&vector);
        let (neighbors, _) = index.search(&quantized);
        let (fraud_score, approved) = score(neighbors);

        let is_failure = approved != entry.expected_approved
            || (fraud_score - entry.expected_fraud_score).abs() > 0.01;
        if is_failure {
            failures += 1;
        }
    }

    let failure_rate = failures as f64 / n as f64;
    eprintln!(
        "integration: {failures}/{n} failures ({:.2}%)",
        failure_rate * 100.0
    );
    assert!(
        failure_rate < 0.05,
        "failure rate {:.2}% exceeds 5% threshold",
        failure_rate * 100.0
    );
}
