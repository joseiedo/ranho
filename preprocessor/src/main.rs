use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufWriter, Write};

pub const IVF_MAGIC: &[u8; 8] = b"RINHIVF4";
const HEADER_SIZE: usize = 20;
const DIMS: usize = 14;
const PADDED_DIMS: usize = 16;
const STRIDE: usize = PADDED_DIMS * 2;
const DEFAULT_NLIST: usize = 4096;
const KMEANS_ITERS: usize = 25;
const DEFAULT_SAMPLE_SIZE: usize = 120_000;
const QUANT_SCALE: f32 = 10_000.0;
const SAMPLE_SEED: u64 = 0x4d595df4d0f33173;

#[derive(Deserialize)]
struct Reference {
    vector: [f32; 14],
    label: String,
}

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

    fn gen_range(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() % upper as u64) as usize
    }
}

#[inline]
fn quantize(v: f32) -> i16 {
    (v * QUANT_SCALE)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

#[inline(always)]
fn sq_dist_to_centroid(v: &[f32; DIMS], centroids_flat: &[f32], ci: usize) -> f32 {
    let base = ci * DIMS;
    let mut d = 0.0f32;
    for k in 0..DIMS {
        let diff = v[k] - centroids_flat[base + k];
        d += diff * diff;
    }
    d
}

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

fn make_sample_indices(n: usize, sample_size: usize, seed: u64) -> Vec<usize> {
    let sample_size = sample_size.min(n);
    let mut sample: Vec<usize> = (0..sample_size).collect();
    let mut rng = Lcg64::new(seed);

    for i in sample_size..n {
        let slot = rng.gen_range(i + 1);
        if slot < sample_size {
            sample[slot] = i;
        }
    }

    sample.sort_unstable();
    sample
}

fn percentile(sorted: &[u32], numerator: usize, denominator: usize) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) * numerator) / denominator;
    sorted[idx]
}

fn report_cluster_balance(counts: &[u32]) {
    let mut sorted = counts.to_vec();
    sorted.sort_unstable();
    let min = *sorted.first().unwrap_or(&0);
    let median = percentile(&sorted, 1, 2);
    let p95 = percentile(&sorted, 95, 100);
    let max = *sorted.last().unwrap_or(&0);

    eprintln!(
        "preprocessor: cluster sizes min={min} median={median} p95={p95} max={max}"
    );
}

fn kmeans_pp_init(vectors: &[[f32; DIMS]], sample: &[usize], nlist: usize) -> Vec<f32> {
    let mut flat = vec![0.0f32; nlist * DIMS];
    let mut dmin = vec![f32::MAX; sample.len()];
    let mut rng = Lcg64::new(SAMPLE_SEED ^ sample.len() as u64 ^ nlist as u64);

    flat[..DIMS].copy_from_slice(&vectors[sample[0]]);

    for c in 1..nlist {
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

        let total: f64 = dmin.iter().map(|&x| x as f64).sum();
        let chosen = if total <= 0.0 {
            0
        } else {
            let threshold = total * (rng.next_u64() as f64 / u64::MAX as f64);
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
    let nlist = std::env::var("NLIST")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_NLIST);
    let sample_cap = std::env::var("SAMPLE_SIZE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_SAMPLE_SIZE);

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

    let nlist_actual = nlist.min(n.max(1));
    let sample_size = sample_cap.min(n);
    let sample = make_sample_indices(n, sample_size, SAMPLE_SEED);
    eprintln!(
        "preprocessor: k-means on sample={sample_size} (NLIST={nlist_actual}, iters={KMEANS_ITERS}, seed={SAMPLE_SEED:#x})"
    );

    eprintln!("preprocessor: kmeans++ init...");
    let mut centroids_flat = kmeans_pp_init(&vectors, &sample, nlist_actual);

    let mut sample_assignments = vec![0u32; sample_size];
    let nthreads = rayon::current_num_threads();

    for iter in 0..KMEANS_ITERS {
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

        let chunk = (sample_size + nthreads - 1) / nthreads.max(1);
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
            "  iter {}/{}: {} reassignments",
            iter + 1,
            KMEANS_ITERS,
            changed
        );
        if changed == 0 {
            break;
        }
    }

    eprintln!("preprocessor: assigning all {n} vectors to final centroids...");
    let assignments: Vec<u32> = vectors
        .par_iter()
        .map(|v| nearest(v, &centroids_flat, nlist_actual))
        .collect();

    let mut counts = vec![0u32; nlist_actual];
    for &c in &assignments {
        counts[c as usize] += 1;
    }
    report_cluster_balance(&counts);

    eprintln!("preprocessor: sorting vectors by cluster...");
    let mut cluster_starts = vec![0u32; nlist_actual + 1];
    let mut running: u32 = 0;
    for ci in 0..nlist_actual {
        cluster_starts[ci] = running;
        running += counts[ci];
    }
    cluster_starts[nlist_actual] = running;

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| assignments[i]);

    let mut radii = vec![0.0f32; nlist_actual];
    for ci in 0..nlist_actual {
        let base = ci * DIMS;
        let centroid = &centroids_flat[base..base + DIMS];
        let centroid_q: [f32; DIMS] =
            std::array::from_fn(|d| centroid[d] * QUANT_SCALE);
        let start = cluster_starts[ci] as usize;
        let end = cluster_starts[ci + 1] as usize;

        for &idx in &order[start..end] {
            let mut dist = 0.0f32;
            for d in 0..DIMS {
                let diff = quantize(vectors[idx][d]) as f32 - centroid_q[d];
                dist += diff * diff;
            }
            radii[ci] = radii[ci].max(dist.sqrt());
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

    for &v in &centroids_flat {
        writer.write_all(&v.to_le_bytes()).unwrap();
    }

    for &offset in &cluster_starts {
        writer.write_all(&offset.to_le_bytes()).unwrap();
    }

    for &radius in &radii {
        writer.write_all(&radius.to_le_bytes()).unwrap();
    }

    for &idx in &order {
        for &v in &vectors[idx] {
            writer.write_all(&quantize(v).to_le_bytes()).unwrap();
        }
        writer.write_all(&[0u8; STRIDE - DIMS * 2]).unwrap();
    }

    for &idx in &order {
        writer.write_all(&[labels[idx]]).unwrap();
    }

    writer.flush().unwrap();

    let file_size =
        HEADER_SIZE + nlist_actual * DIMS * 4 + (nlist_actual + 1) * 4 + nlist_actual * 4 + n * STRIDE + n;
    eprintln!("preprocessor: wrote {output_path} ({file_size} bytes, IVF NLIST={nlist_actual})");
}
