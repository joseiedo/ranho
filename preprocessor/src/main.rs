use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufWriter, Write};

/// IVF binary index format:
///   [0..8]               magic: b"RINHIVF1"
///   [8..12]              K: u32 le  (number of clusters)
///   [12..16]             N: u32 le  (total vector count)
///   [16..20]             dims: u32 le (= 14)
///   [20 .. 20+K*56]      centroids: K × [f32; 14], row-major, le
///   [.. .. +K*56+(K+1)*4] offsets: (K+1) × u32 le — record index of each cluster start
///   [.. ..]              flat data: N × ([i8; 14] quantized + u8 label), sorted by cluster
pub const IVF_MAGIC: &[u8; 8] = b"RINHIVF1";
const DIMS: usize = 14;
const K: usize = 1024;
const KMEANS_ITERS: usize = 25;

#[derive(Deserialize)]
struct Reference {
    vector: [f32; 14],
    label: String,
}

#[inline]
fn quantize(v: f32) -> i8 {
    (v * 127.0).round().clamp(-127.0, 127.0) as i8
}

#[inline]
fn sq_dist_f32(a: &[f32; DIMS], b: &[f32; DIMS]) -> f32 {
    let mut d = 0.0f32;
    for i in 0..DIMS {
        let diff = a[i] - b[i];
        d += diff * diff;
    }
    d
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
    let n = refs.len();
    eprintln!("preprocessor: parsed {n} records");

    let vectors: Vec<[f32; DIMS]> = refs.iter().map(|r| r.vector).collect();
    let labels: Vec<u8> = refs
        .iter()
        .map(|r| if r.label == "fraud" { 1u8 } else { 0u8 })
        .collect();
    drop(refs);

    let k_actual = K.min(n);
    eprintln!("preprocessor: running k-means (K={k_actual}, iters={KMEANS_ITERS})...");

    let mut centroids: Vec<[f32; DIMS]> = (0..k_actual)
        .map(|i| vectors[i * (n / k_actual)])
        .collect();

    let mut assignments = vec![0u32; n];
    let mut counts = vec![0u32; k_actual];

    for iter in 0..KMEANS_ITERS {
        let new_assignments: Vec<u32> = vectors
            .par_iter()
            .map(|v| {
                let mut best = 0u32;
                let mut best_d = f32::MAX;
                for (ci, centroid) in centroids.iter().enumerate() {
                    let d = sq_dist_f32(v, centroid);
                    if d < best_d {
                        best_d = d;
                        best = ci as u32;
                    }
                }
                best
            })
            .collect();

        let changed = new_assignments
            .iter()
            .zip(assignments.iter())
            .filter(|(a, b)| a != b)
            .count();
        assignments = new_assignments;

        let mut sums = vec![[0.0f32; DIMS]; k_actual];
        counts.fill(0);
        for (i, &c) in assignments.iter().enumerate() {
            let ci = c as usize;
            counts[ci] += 1;
            for d in 0..DIMS {
                sums[ci][d] += vectors[i][d];
            }
        }
        for ci in 0..k_actual {
            if counts[ci] > 0 {
                let cnt = counts[ci] as f32;
                for d in 0..DIMS {
                    centroids[ci][d] = sums[ci][d] / cnt;
                }
            }
        }

        eprintln!(
            "  iter {}/{KMEANS_ITERS}: {changed} reassignments",
            iter + 1
        );
        if changed == 0 {
            break;
        }
    }

    eprintln!("preprocessor: sorting vectors by cluster...");

    let mut cluster_starts = vec![0u32; k_actual + 1];
    let mut running: u32 = 0;
    for ci in 0..k_actual {
        cluster_starts[ci] = running;
        running += counts[ci];
    }
    cluster_starts[k_actual] = running;

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| assignments[i]);

    eprintln!("preprocessor: writing IVF index...");

    let out = File::create(&output_path).expect("failed to create output");
    let mut writer = BufWriter::new(out);

    // Header (20 bytes)
    writer.write_all(IVF_MAGIC).unwrap();
    writer.write_all(&(k_actual as u32).to_le_bytes()).unwrap();
    writer.write_all(&(n as u32).to_le_bytes()).unwrap();
    writer.write_all(&(DIMS as u32).to_le_bytes()).unwrap();

    // Centroids: K × DIMS × f32
    for centroid in &centroids {
        for &v in centroid {
            writer.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    // Cluster start offsets: (K+1) × u32
    for &o in &cluster_starts {
        writer.write_all(&o.to_le_bytes()).unwrap();
    }

    // Flat data in cluster order: each record is [i8; DIMS] + u8 label (15 bytes)
    for &idx in &order {
        for &v in &vectors[idx] {
            writer.write_all(&[quantize(v) as u8]).unwrap();
        }
        writer.write_all(&[labels[idx]]).unwrap();
    }

    writer.flush().unwrap();

    let file_size =
        20 + k_actual * DIMS * 4 + (k_actual + 1) * 4 + n * (DIMS + 1);
    eprintln!(
        "preprocessor: wrote {output_path} ({file_size} bytes, IVF K={k_actual})"
    );
}
