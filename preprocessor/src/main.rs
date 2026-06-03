use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufWriter, Write};

pub const IVF_MAGIC: &[u8; 8] = b"RINHIVF4";
const DIMS: usize = 14;
const NLIST: usize = 4096;
const KMEANS_ITERS: usize = 25;
const SAMPLE_SIZE: usize = 60_000;
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

fn kmeans_pp_init_i16(
    vectors: &[[i16; DIMS]],
    sample: &[usize],
    nlist: usize,
) -> Vec<i16> {
    let mut flat = vec![0i16; nlist * DIMS];
    let mut dmin = vec![i64::MAX; sample.len()];

    flat[..DIMS].copy_from_slice(&vectors[sample[0]]);

    for c in 1..nlist {
        let prev_base = (c - 1) * DIMS;
        for (i, &si) in sample.iter().enumerate() {
            let v = &vectors[si];
            let mut d: i64 = 0;
            for k in 0..DIMS {
                let diff = v[k] as i64 - flat[prev_base + k] as i64;
                d += diff * diff;
            }
            if d < dmin[i] {
                dmin[i] = d;
            }
        }

        let total: f64 = dmin.iter().map(|&x| x as f64).sum();
        let chosen = if total <= 0.0 {
            0
        } else {
            let threshold =
                total * ((c as u64).wrapping_mul(0x9e3779b97f4a7c15u64) as f64 / u64::MAX as f64);
            let mut acc = 0.0f64;
            let mut chosen = sample.len() - 1;
            for (i, &d) in dmin.iter().enumerate() {
                acc += d as f64;
                if acc >= threshold {
                    chosen = i;
                    break;
                }
            }
            chosen
        };

        flat[c * DIMS..(c + 1) * DIMS].copy_from_slice(&vectors[sample[chosen]]);

        if c % 512 == 0 {
            eprintln!("  kmeans++ init: {c}/{nlist}");
        }
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

    eprintln!("preprocessor: kmeans++ init...");
    let mut centroids_i16 = kmeans_pp_init_i16(&vectors, &sample, nlist_actual);

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

    let file_size = 20 + nlist_actual * DIMS * 2 + (nlist_actual + 1) * 4 + n * 32 + n;
    eprintln!("preprocessor: wrote {output_path} ({file_size} bytes, IVF NLIST={nlist_actual})");
}
