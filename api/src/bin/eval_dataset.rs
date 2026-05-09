use api::scorer::score;
use api::search::SearchIndex;
use api::types::{Label, TransactionPayload};
use api::vectorizer::Vectorizer;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Deserialize)]
struct Dataset {
    entries: Vec<DatasetEntry>,
}

#[derive(Debug, Deserialize)]
struct DatasetEntry {
    request: Value,
    expected_approved: bool,
    expected_fraud_score: f32,
}

#[derive(Default)]
struct Counters {
    tp: usize,
    tn: usize,
    fp: usize,
    fn_: usize,
}

struct Config {
    dataset_path: String,
    index_path: String,
    edge_only: bool,
    show_all_mismatches: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let dataset = load_dataset(&config.dataset_path)?;
    let index = SearchIndex::open(&config.index_path)?;
    let vectorizer = Vectorizer::new(default_mcc_risk());

    let mut counters = Counters::default();
    let mut processed = 0usize;
    let mut mismatches = Vec::new();

    for entry in dataset.entries {
        if config.edge_only && !is_edge_case(entry.expected_fraud_score) {
            continue;
        }

        let request_id = request_id(&entry.request).to_string();
        let payload: TransactionPayload = serde_json::from_value(entry.request.clone())?;
        let vector = vectorizer.vectorize(&payload);
        let quantized = Vectorizer::quantize(&vector);
        let neighbors = index.search_with_vector(&vector, &quantized);
        let (actual_score, actual_approved) = score(neighbors);

        processed += 1;

        match (entry.expected_approved, actual_approved) {
            (true, true) => counters.tn += 1,
            (false, false) => counters.tp += 1,
            (true, false) => {
                counters.fp += 1;
                mismatches.push(format_mismatch(
                    &request_id,
                    entry.expected_approved,
                    entry.expected_fraud_score,
                    actual_approved,
                    actual_score,
                    &neighbors,
                    "FP",
                ));
            }
            (false, true) => {
                counters.fn_ += 1;
                mismatches.push(format_mismatch(
                    &request_id,
                    entry.expected_approved,
                    entry.expected_fraud_score,
                    actual_approved,
                    actual_score,
                    &neighbors,
                    "FN",
                ));
            }
        }
    }

    println!("dataset_path={}", config.dataset_path);
    println!("index_path={}", config.index_path);
    println!("processed={processed}");
    println!("tp={}", counters.tp);
    println!("tn={}", counters.tn);
    println!("fp={}", counters.fp);
    println!("fn={}", counters.fn_);

    if processed > 0 {
        let failure_rate = (counters.fp + counters.fn_) as f64 / processed as f64;
        println!("failure_rate={:.6}", failure_rate);
    }

    if mismatches.is_empty() {
        println!("mismatches=0");
    } else {
        println!("mismatches={}", mismatches.len());
        if config.show_all_mismatches {
            for mismatch in mismatches {
                println!("{mismatch}");
            }
        } else {
            for mismatch in mismatches.iter().take(20) {
                println!("{mismatch}");
            }
            if mismatches.len() > 20 {
                println!("... {} more mismatches omitted", mismatches.len() - 20);
            }
        }
    }

    Ok(())
}

fn parse_args() -> Result<Config, String> {
    let mut dataset_path = String::from("spec/test/test-data.json");
    let mut index_path = String::from("resources/index.bin");
    let mut edge_only = false;
    let mut show_all_mismatches = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dataset" => {
                dataset_path = args
                    .next()
                    .ok_or_else(|| String::from("missing value for --dataset"))?;
            }
            "--index" => {
                index_path = args
                    .next()
                    .ok_or_else(|| String::from("missing value for --index"))?;
            }
            "--edge-only" => edge_only = true,
            "--all-mismatches" => show_all_mismatches = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(format!("unknown argument: {other}"));
            }
        }
    }

    Ok(Config {
        dataset_path,
        index_path,
        edge_only,
        show_all_mismatches,
    })
}

fn print_help() {
    println!("Usage: cargo run -p api --bin eval_dataset -- [options]");
    println!("  --dataset <path>         Path to test-data.json");
    println!("  --index <path>           Path to resources/index.bin");
    println!("  --edge-only              Only evaluate entries with expected_fraud_score == 0.6");
    println!("  --all-mismatches         Print every mismatch instead of the first 20");
}

fn load_dataset(path: &str) -> Result<Dataset, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&contents)?)
}

fn is_edge_case(expected_fraud_score: f32) -> bool {
    (expected_fraud_score - 0.6).abs() < f32::EPSILON
}

fn format_mismatch(
    request_id: &str,
    expected_approved: bool,
    expected_fraud_score: f32,
    actual_approved: bool,
    actual_score: f32,
    neighbors: &[Label; 5],
    kind: &str,
) -> String {
    format!(
        "{kind} id={} expected_approved={} expected_score={} actual_approved={} actual_score={} fraud_neighbors={} neighbors={}",
        request_id,
        expected_approved,
        expected_fraud_score,
        actual_approved,
        actual_score,
        fraud_neighbor_count(neighbors),
        format_neighbors(neighbors),
    )
}

fn request_id(request: &Value) -> &str {
    request
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("<missing-id>")
}

fn fraud_neighbor_count(neighbors: &[Label; 5]) -> usize {
    neighbors
        .iter()
        .filter(|&&label| label == Label::Fraud)
        .count()
}

fn format_neighbors(neighbors: &[Label; 5]) -> String {
    let names: Vec<&str> = neighbors
        .iter()
        .map(|label| match label {
            Label::Legit => "L",
            Label::Fraud => "F",
        })
        .collect();
    format!("[{}]", names.join(","))
}

fn default_mcc_risk() -> HashMap<String, f32> {
    [
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
    .collect()
}
