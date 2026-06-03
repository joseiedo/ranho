use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufWriter, Write};

pub const IVF_MAGIC: &[u8; 8] = b"RINHIVF5";
const DIMS: usize = 14;
const NLIST: usize = 2048;
const KMEANS_ITERS: usize = 10;
const SAMPLE_SIZE: usize = 131_072;
const QUANT_SCALE: f32 = 10_000.0;

#[derive(Deserialize)]
struct Reference {
    vector: [f32; 14],
    label: String,
}

#[inline]
fn quantize(v: f32) -> i16 {
    (v * QUANT_SCALE)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn quantize_vector(f32v: &[f32; DIMS]) -> [i16; DIMS] {
    let mut out = [0i16; DIMS];
    for (i, &v) in f32v.iter().enumerate() {
        out[i] = quantize(v);
    }
    out
}

#[inline(always)]
fn sq_dist_i16(v: &[i16; DIMS], centroids: &[i16], ci: usize) -> i64 {
    let base = ci * DIMS;
    let mut d: i64 = 0;
    for k in 0..DIMS {
        let diff = v[k] as i64 - centroids[base + k] as i64;
        d += diff * diff;
    }
    d
}

#[inline]
fn nearest_i16(v: &[i16; DIMS], centroids: &[i16], nlist: usize) -> u32 {
    let mut best = 0u32;
    let mut best_d = i64::MAX;
    for ci in 0..nlist {
        let d = sq_dist_i16(v, centroids, ci);
        if d < best_d {
            best_d = d;
            best = ci as u32;
        }
    }
    best
}

fn deterministic_init(vectors: &[[i16; DIMS]], sample: &[usize], nlist: usize) -> Vec<i16> {
    let mut flat = vec![0i16; nlist * DIMS];
    for c in 0..nlist {
        let si = (c as u64 * sample.len() as u64 / nlist as u64) as usize;
        flat[c * DIMS..(c + 1) * DIMS].copy_from_slice(&vectors[sample[si]]);
    }
    flat
}

fn main() {
    let input_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./resources/references.json.gz".to_string());
    let output_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "./resources/index.bin".to_string());

    eprintln!("preprocessor: reading {input_path}");
    let file = File::open(&input_path).expect("failed to open input");
    let gz = GzDecoder::new(file);

    eprintln!("preprocessor: parsing JSON...");
    let refs: Vec<Reference> = serde_json::from_reader(gz).expect("failed to parse JSON");
    let n = refs.len();
    eprintln!("preprocessor: parsed {n} records");

    let vectors_f32: Vec<[f32; DIMS]> = refs.iter().map(|r| r.vector).collect();
    let labels: Vec<u8> = refs
        .iter()
        .map(|r| if r.label == "fraud" { 1u8 } else { 0u8 })
        .collect();
    drop(refs);

    let vectors: Vec<[i16; DIMS]> = vectors_f32.iter().map(quantize_vector).collect();
    drop(vectors_f32);

    let nlist_actual = NLIST.min(n);
    let sample_size = SAMPLE_SIZE.min(n);
    let sample: Vec<usize> = (0..sample_size).map(|i| i * (n / sample_size)).collect();
    eprintln!(
        "preprocessor: k-means on sample={sample_size} (NLIST={nlist_actual}, iters={KMEANS_ITERS})"
    );

    eprintln!("preprocessor: deterministic init...");
    let mut centroids_i16 = deterministic_init(&vectors, &sample, nlist_actual);

    let mut sample_assignments = vec![0u32; sample_size];
    let nthreads = rayon::current_num_threads();

    for iter in 0..KMEANS_ITERS {
        let new_assignments: Vec<u32> = sample
            .par_iter()
            .map(|&si| nearest_i16(&vectors[si], &centroids_i16, nlist_actual))
            .collect();

        let changed = new_assignments
            .iter()
            .zip(sample_assignments.iter())
            .filter(|(a, b)| a != b)
            .count();
        sample_assignments = new_assignments;

        let chunk = (sample_size + nthreads - 1) / nthreads;
        let thread_results: Vec<(Vec<i64>, Vec<u32>)> = (0..nthreads)
            .into_par_iter()
            .map(|tid| {
                let start = tid * chunk;
                let end = (start + chunk).min(sample_size);
                let mut local_sums = vec![0i64; nlist_actual * DIMS];
                let mut local_counts = vec![0u32; nlist_actual];
                for i in start..end {
                    let ci = sample_assignments[i] as usize;
                    local_counts[ci] += 1;
                    let v = &vectors[sample[i]];
                    let base = ci * DIMS;
                    for d in 0..DIMS {
                        local_sums[base + d] += v[d] as i64;
                    }
                }
                (local_sums, local_counts)
            })
            .collect();

        let mut global_sums = vec![0i64; nlist_actual * DIMS];
        let mut global_counts = vec![0u32; nlist_actual];
        for (sums, counts) in &thread_results {
            for i in 0..nlist_actual * DIMS {
                global_sums[i] += sums[i];
            }
            for i in 0..nlist_actual {
                global_counts[i] += counts[i];
            }
        }

        for ci in 0..nlist_actual {
            if global_counts[ci] > 0 {
                let base = ci * DIMS;
                for d in 0..DIMS {
                    let mean = global_sums[base + d] as f64 / global_counts[ci] as f64;
                    centroids_i16[base + d] = mean.round() as i16;
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

    eprintln!("preprocessor: assigning all {n} vectors to final centroids...");
    let assignments: Vec<u32> = vectors
        .par_iter()
        .map(|v| nearest_i16(v, &centroids_i16, nlist_actual))
        .collect();

    eprintln!("preprocessor: sorting vectors by cluster...");
    let mut counts = vec![0u32; nlist_actual];
    for &c in &assignments {
        counts[c as usize] += 1;
    }

    let mut cluster_starts = vec![0u32; nlist_actual + 1];
    let mut running: u32 = 0;
    for ci in 0..nlist_actual {
        cluster_starts[ci] = running;
        running += counts[ci];
    }
    cluster_starts[nlist_actual] = running;

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| assignments[i]);

    eprintln!("preprocessor: computing bounding boxes...");
    let mut sorted = vec![0usize; n];
    for (i, &idx) in order.iter().enumerate() {
        sorted[idx] = i;
    }

    let mut bbox_min = vec![i16::MAX; nlist_actual * DIMS];
    let mut bbox_max = vec![i16::MIN; nlist_actual * DIMS];
    for ci in 0..nlist_actual {
        let base = ci * DIMS;
        let start = cluster_starts[ci] as usize;
        let end = cluster_starts[ci + 1] as usize;
        for j in start..end {
            let v = &vectors[order[j]];
            for d in 0..DIMS {
                let vd = v[d];
                let idx = base + d;
                if vd < bbox_min[idx] {
                    bbox_min[idx] = vd;
                }
                if vd > bbox_max[idx] {
                    bbox_max[idx] = vd;
                }
            }
        }
        if start == end {
            for d in 0..DIMS {
                bbox_min[base + d] = 0;
                bbox_max[base + d] = 0;
            }
        }
    }

    eprintln!("preprocessor: writing IVF index...");
    let out = File::create(&output_path).expect("failed to create output");
    let mut writer = BufWriter::new(out);

    writer.write_all(IVF_MAGIC).unwrap();
    writer
        .write_all(&(nlist_actual as u32).to_le_bytes())
        .unwrap();
    writer.write_all(&(n as u32).to_le_bytes()).unwrap();
    writer.write_all(&(DIMS as u32).to_le_bytes()).unwrap();

    for &v in &centroids_i16 {
        writer.write_all(&v.to_le_bytes()).unwrap();
    }

    for &v in &bbox_min {
        writer.write_all(&v.to_le_bytes()).unwrap();
    }

    for &v in &bbox_max {
        writer.write_all(&v.to_le_bytes()).unwrap();
    }

    for &o in &cluster_starts {
        writer.write_all(&o.to_le_bytes()).unwrap();
    }

    for &idx in &order {
        for &v in &vectors[idx] {
            writer.write_all(&v.to_le_bytes()).unwrap();
        }
        writer.write_all(&[0u8; 4]).unwrap();
    }

    for &idx in &order {
        writer.write_all(&[labels[idx]]).unwrap();
    }

    writer.flush().unwrap();

    let file_size = 20
        + nlist_actual * DIMS * 2
        + nlist_actual * DIMS * 2
        + nlist_actual * DIMS * 2
        + (nlist_actual + 1) * 4
        + n * 32
        + n;
    eprintln!(
        "preprocessor: wrote {output_path} ({file_size} bytes, IVF NLIST={nlist_actual})"
    );
}
