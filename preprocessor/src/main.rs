use flate2::read::GzDecoder;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufWriter, Write};

/// Binary index format:
///   [0..8]   magic: b"RINHA026"
///   [8..12]  count: u32 le
///   [12..16] dims: u32 le (= 14)
///   per record: [i8; 14] vector + u8 label (0=legit, 1=fraud)
pub const MAGIC: &[u8; 8] = b"RINHA026";
pub const DIMS: u32 = 14;

#[derive(Deserialize)]
struct Reference {
    vector: [f32; 14],
    label: String,
}

fn quantize(v: f32) -> i8 {
    (v * 127.0).round().clamp(-127.0, 127.0) as i8
}

fn main() {
    let input_path = std::env::args().nth(1).unwrap_or_else(|| {
        "../rinha-de-backend-2026/resources/references.json.gz".to_string()
    });
    let output_path = std::env::args().nth(2).unwrap_or_else(|| {
        "../rinha-de-backend-2026/resources/index.bin".to_string()
    });

    eprintln!("preprocessor: reading {input_path}");
    let file = File::open(&input_path).expect("failed to open input");
    let gz = GzDecoder::new(file);

    eprintln!("preprocessor: parsing JSON...");
    let refs: Vec<Reference> = serde_json::from_reader(gz).expect("failed to parse JSON");
    let count = refs.len() as u32;
    eprintln!("preprocessor: parsed {count} records");

    let out = File::create(&output_path).expect("failed to create output");
    let mut writer = BufWriter::new(out);

    writer.write_all(MAGIC).unwrap();
    writer.write_all(&count.to_le_bytes()).unwrap();
    writer.write_all(&DIMS.to_le_bytes()).unwrap();

    for r in &refs {
        for &v in &r.vector {
            writer.write_all(&[quantize(v) as u8]).unwrap();
        }
        let label: u8 = if r.label == "fraud" { 1 } else { 0 };
        writer.write_all(&[label]).unwrap();
    }

    writer.flush().unwrap();
    eprintln!("preprocessor: wrote {output_path} ({} bytes)", 16 + count as u64 * 15);
}
