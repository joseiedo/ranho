use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufWriter, Write};

/// IVF binary index format:
///   [0..8]                magic: b"RINHIVF2"
///   [8..12]               K: u32 le  (number of clusters)
///   [12..16]              N: u32 le  (total vector count)
///   [16..20]              dims: u32 le (= 14)
///   [20 .. 20+K*56]       centroids: K × [f32; 14], row-major, le
///   [.. .. +K*56+(K+1)*4] offsets: (K+1) × u32 le — record index of each cluster start
///   [.. ..]               flat data: N × ([i8; 14] quantized + u8 label), sorted by cluster
pub const IVF_MAGIC: &[u8; 8] = b"RINHIVF2";
const DIMS: usize = 14;
const NLIST: usize = 4096;
const KMEANS_ITERS: usize = 25;
// k-means runs on this many vectors; final assignment is one pass over all N
const SAMPLE_SIZE: usize = 60_000;

#[derive(Deserialize)]
struct Reference {
    vector: [f32; 14],
    label: String,
}

#[inline]
fn quantize(v: f32) -> i8 {
    (v * 127.0).round().clamp(-127.0, 127.0) as i8
}

/// Squared L2 distance between a vector and a centroid stored in a flat slice.
/// `centroids_flat` is laid out as [c0d0, c0d1, ..., c0d13, c1d0, ...].
#[inline(always)]
fn sq_dist_to_centroid(v: &[f32; DIMS], centroids_flat: &[f32], ci: usize) -> f32 {
    let base = ci * DIMS;
    let mut d = 0.0f32;
    // Fixed-size loop — compiler will unroll + auto-vectorize (AVX2 on x86)
    for k in 0..DIMS {
        let diff = v[k] - centroids_flat[base + k];
        d += diff * diff;
    }
    d
}

/// Find the nearest centroid index for vector `v`.
#[inline]
fn nearest(v: &[f32; DIMS], centroids_flat: &[f32], nlist: usize) -> u32 {
    let mut best = 0u32;
    let mut best_d = f32::MAX;
    for ci in 0..nlist {
        let d = sq_dist_to_centroid(v, centroids_flat, ci);
        if d < best_d {
            best_d = d;
            best = ci as u32;
        }
    }
    best
}

/// k-means++ initialisation over the sample indices.
fn kmeans_pp_init(vectors: &[[f32; DIMS]], sample: &[usize], nlist: usize) -> Vec<f32> {
    let mut flat = vec![0.0f32; nlist * DIMS];
    let mut dmin = vec![f32::MAX; sample.len()];

    // Pick first centroid at index 0 of the sample
    flat[..DIMS].copy_from_slice(&vectors[sample[0]]);

    for c in 1..nlist {
        // Update dmin against the centroid we just added
        let prev_base = (c - 1) * DIMS;
        for (i, &si) in sample.iter().enumerate() {
            let v = &vectors[si];
            let mut d = 0.0f32;
            for k in 0..DIMS {
                let diff = v[k] - flat[prev_base + k];
                d += diff * diff;
            }
            if d < dmin[i] {
                dmin[i] = d;
            }
        }

        // Weighted pick: use a cheap deterministic threshold based on centroid index
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

    // ── 1. Parse ─────────────────────────────────────────────────────────────
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

    let nlist_actual = NLIST.min(n);

    // ── 2. Sample ─────────────────────────────────────────────────────────────
    // Evenly-spaced sample — no extra deps, good coverage
    let sample_size = SAMPLE_SIZE.min(n);
    let sample: Vec<usize> = (0..sample_size).map(|i| i * (n / sample_size)).collect();
    eprintln!(
        "preprocessor: k-means on sample={sample_size} (NLIST={nlist_actual}, iters={KMEANS_ITERS})"
    );

    // ── 3. k-means++ init on sample ───────────────────────────────────────────
    eprintln!("preprocessor: kmeans++ init...");
    let mut centroids_flat = kmeans_pp_init(&vectors, &sample, nlist_actual);

    // ── 4. k-means iterations on sample (parallel assignment + parallel reduce)
    let mut sample_assignments = vec![0u32; sample_size];
    let nthreads = rayon::current_num_threads();

    for iter in 0..KMEANS_ITERS {
        // Parallel assignment over sample
        let new_assignments: Vec<u32> = sample
            .par_iter()
            .map(|&si| nearest(&vectors[si], &centroids_flat, nlist_actual))
            .collect();

        let changed = new_assignments
            .iter()
            .zip(sample_assignments.iter())
            .filter(|(a, b)| a != b)
            .count();
        sample_assignments = new_assignments;

        // Parallel accumulation with per-thread buffers, then reduce
        let chunk = (sample_size + nthreads - 1) / nthreads;
        let thread_results: Vec<(Vec<f64>, Vec<u32>)> = (0..nthreads)
            .into_par_iter()
            .map(|tid| {
                let start = tid * chunk;
                let end = (start + chunk).min(sample_size);
                let mut local_sums = vec![0.0f64; nlist_actual * DIMS];
                let mut local_counts = vec![0u32; nlist_actual];
                for i in start..end {
                    let ci = sample_assignments[i] as usize;
                    local_counts[ci] += 1;
                    let v = &vectors[sample[i]];
                    let base = ci * DIMS;
                    for d in 0..DIMS {
                        local_sums[base + d] += v[d] as f64;
                    }
                }
                (local_sums, local_counts)
            })
            .collect();

        // Reduce into global buffers
        let mut global_sums = vec![0.0f64; nlist_actual * DIMS];
        let mut global_counts = vec![0u32; nlist_actual];
        for (sums, counts) in &thread_results {
            for i in 0..nlist_actual * DIMS {
                global_sums[i] += sums[i];
            }
            for i in 0..nlist_actual {
                global_counts[i] += counts[i];
            }
        }

        // Update centroids
        for ci in 0..nlist_actual {
            if global_counts[ci] > 0 {
                let inv = 1.0 / global_counts[ci] as f64;
                let base = ci * DIMS;
                for d in 0..DIMS {
                    centroids_flat[base + d] = (global_sums[base + d] * inv) as f32;
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

    // ── 5. Assign ALL vectors — single parallel pass over full dataset ─────────
    eprintln!("preprocessor: assigning all {n} vectors to final centroids...");
    let assignments: Vec<u32> = vectors
        .par_iter()
        .map(|v| nearest(v, &centroids_flat, nlist_actual))
        .collect();

    // ── 6. Count + sort by cluster ────────────────────────────────────────────
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

    // ── 7. Write index ────────────────────────────────────────────────────────
    eprintln!("preprocessor: writing IVF index...");
    let out = File::create(&output_path).expect("failed to create output");
    let mut writer = BufWriter::new(out);

    // Header (20 bytes)
    writer.write_all(IVF_MAGIC).unwrap();
    writer
        .write_all(&(nlist_actual as u32).to_le_bytes())
        .unwrap();
    writer.write_all(&(n as u32).to_le_bytes()).unwrap();
    writer.write_all(&(DIMS as u32).to_le_bytes()).unwrap();

    // Centroids: NLIST × DIMS × f32 (flat buffer, already correct layout)
    for &v in &centroids_flat {
        writer.write_all(&v.to_le_bytes()).unwrap();
    }

    // Cluster start offsets: (NLIST+1) × u32
    for &o in &cluster_starts {
        writer.write_all(&o.to_le_bytes()).unwrap();
    }

    // Vectors in cluster order: N × [i8; 14] + [0u8; 2] padding = 16 bytes each
    for &idx in &order {
        for &v in &vectors[idx] {
            writer.write_all(&[quantize(v) as u8]).unwrap();
        }
        writer.write_all(&[0u8, 0u8]).unwrap();
    }

    // Labels in cluster order: N × u8
    for &idx in &order {
        writer.write_all(&[labels[idx]]).unwrap();
    }

    writer.flush().unwrap();

    let file_size = 20 + nlist_actual * DIMS * 4 + (nlist_actual + 1) * 4 + n * 16 + n;
    eprintln!("preprocessor: wrote {output_path} ({file_size} bytes, IVF NLIST={nlist_actual})");
}
