use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufWriter, Write};

pub const IVF_MAGIC: &[u8; 8] = b"RINHIVF4";
const DIMS: usize = 14;
const STRIDE: usize = 16;
const DEFAULT_NLIST: usize = 2048;
const DEFAULT_KMEANS_ITERS: usize = 10;
const DEFAULT_SAMPLE_SIZE: usize = 131_072;
const QUANT_SCALE: f32 = 10_000.0;

#[derive(Deserialize)]
struct Reference {
    vector: [f32; DIMS],
    label: String,
}

#[inline]
fn quantize(v: f32) -> i16 {
    (v * QUANT_SCALE)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

#[inline(always)]
fn l2_squared(a: &[i16], a_base: usize, b: &[i16], b_base: usize) -> i64 {
    let mut dist = 0i64;
    for d in 0..STRIDE {
        let diff = a[a_base + d] as i64 - b[b_base + d] as i64;
        dist += diff * diff;
    }
    dist
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "./resources/references.json.gz".to_string());
    let output_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "./resources/index.bin".to_string());
    let requested_nlist = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NLIST);
    let requested_sample = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SAMPLE_SIZE);
    let kmeans_iters = args
        .get(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_KMEANS_ITERS);

    eprintln!("preprocessor: reading {input_path}");
    let file = File::open(&input_path).expect("failed to open input");
    let gz = GzDecoder::new(file);

    eprintln!("preprocessor: parsing JSON...");
    let refs: Vec<Reference> = serde_json::from_reader(gz).expect("failed to parse JSON");
    let n = refs.len();
    assert!(n > 0, "references dataset must not be empty");
    eprintln!("preprocessor: parsed {n} records");

    let mut vectors = vec![0i16; n * STRIDE];
    let mut labels = vec![0u8; n];
    for (i, r) in refs.iter().enumerate() {
        let base = i * STRIDE;
        for d in 0..DIMS {
            vectors[base + d] = quantize(r.vector[d]);
        }
        labels[i] = u8::from(r.label == "fraud");
    }
    drop(refs);

    let nlist = requested_nlist.min(n);
    let sample_size = if requested_sample == 0 {
        n
    } else {
        requested_sample.min(n)
    };
    eprintln!(
        "preprocessor: k-means on sample={sample_size} (NLIST={nlist}, iters={kmeans_iters})"
    );

    let (centroids_f32, centroids_i16) =
        kmeans_short(&vectors, n, nlist, kmeans_iters, sample_size);

    eprintln!("preprocessor: assigning all {n} vectors to final centroids...");
    let assignments: Vec<usize> = (0..n)
        .into_par_iter()
        .map(|i| {
            let vec_base = i * STRIDE;
            let mut best_cluster = 0usize;
            let mut best_dist = i64::MAX;
            for c in 0..nlist {
                let centroid_base = c * STRIDE;
                let dist = l2_squared(&vectors, vec_base, &centroids_i16, centroid_base);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = c;
                }
            }
            best_cluster
        })
        .collect();

    let mut cluster_sizes = vec![0u32; nlist];
    for &cluster in &assignments {
        cluster_sizes[cluster] += 1;
    }

    let mut offsets = vec![0u32; nlist + 1];
    for c in 0..nlist {
        offsets[c + 1] = offsets[c] + cluster_sizes[c];
    }

    eprintln!("preprocessor: writing cluster-sorted column-major vectors...");
    let mut dim_data = vec![0i16; DIMS * n];
    let mut sorted_labels = vec![0u8; n];
    let mut write_pos = offsets[..nlist].to_vec();

    for i in 0..n {
        let cluster = assignments[i];
        let pos = write_pos[cluster] as usize;
        write_pos[cluster] += 1;

        let src_base = i * STRIDE;
        for d in 0..DIMS {
            dim_data[d * n + pos] = vectors[src_base + d];
        }
        sorted_labels[pos] = labels[i];
    }

    eprintln!("preprocessor: computing cluster bounding boxes...");
    let mut bbox_min = vec![0i16; nlist * DIMS];
    let mut bbox_max = vec![0i16; nlist * DIMS];
    for c in 0..nlist {
        let start = offsets[c] as usize;
        let end = offsets[c + 1] as usize;
        let bbox_base = c * DIMS;

        if start == end {
            continue;
        }

        for d in 0..DIMS {
            let dim_base = d * n;
            let mut min_v = i16::MAX;
            let mut max_v = i16::MIN;
            for i in start..end {
                let value = dim_data[dim_base + i];
                if value < min_v {
                    min_v = value;
                }
                if value > max_v {
                    max_v = value;
                }
            }
            bbox_min[bbox_base + d] = min_v;
            bbox_max[bbox_base + d] = max_v;
        }
    }

    eprintln!("preprocessor: writing IVF index...");
    let out = File::create(&output_path).expect("failed to create output");
    let mut writer = BufWriter::new(out);

    writer.write_all(IVF_MAGIC).unwrap();
    writer.write_all(&(n as u32).to_le_bytes()).unwrap();
    writer.write_all(&(nlist as u32).to_le_bytes()).unwrap();
    writer.write_all(&(DIMS as u32).to_le_bytes()).unwrap();
    writer.write_all(&(STRIDE as u32).to_le_bytes()).unwrap();

    for &v in &centroids_f32 {
        writer.write_all(&v.to_le_bytes()).unwrap();
    }
    for &v in &bbox_min {
        writer.write_all(&v.to_le_bytes()).unwrap();
    }
    for &v in &bbox_max {
        writer.write_all(&v.to_le_bytes()).unwrap();
    }
    for &v in &offsets {
        writer.write_all(&v.to_le_bytes()).unwrap();
    }
    for &v in &dim_data {
        writer.write_all(&v.to_le_bytes()).unwrap();
    }
    writer.write_all(&sorted_labels).unwrap();
    writer.flush().unwrap();

    let file_size = 24
        + centroids_f32.len() * std::mem::size_of::<f32>()
        + bbox_min.len() * std::mem::size_of::<i16>()
        + bbox_max.len() * std::mem::size_of::<i16>()
        + offsets.len() * std::mem::size_of::<u32>()
        + dim_data.len() * std::mem::size_of::<i16>()
        + sorted_labels.len();
    eprintln!("preprocessor: wrote {output_path} ({file_size} bytes, IVF NLIST={nlist})");
}

fn kmeans_short(
    vectors: &[i16],
    n: usize,
    requested_nlist: usize,
    kmeans_iters: usize,
    requested_sample_size: usize,
) -> (Vec<f32>, Vec<i16>) {
    let nlist = requested_nlist.min(n);
    let sample_size = requested_sample_size.clamp(nlist, n);
    let mut centroids = vec![0.0f64; nlist * STRIDE];

    let mut sample_idx = vec![0usize; sample_size];
    for (i, slot) in sample_idx.iter_mut().enumerate() {
        *slot = i * n / sample_size;
    }

    eprintln!("preprocessor: deterministic init...");
    for c in 0..nlist {
        let sample_pos = c * sample_size / nlist;
        let src_base = sample_idx[sample_pos] * STRIDE;
        let dst_base = c * STRIDE;
        for d in 0..STRIDE {
            centroids[dst_base + d] = vectors[src_base + d] as f64;
        }
    }

    let mut assignments = vec![0usize; sample_size];
    let mut previous = vec![usize::MAX; sample_size];
    let mut quantized_centroids = vec![0i16; nlist * STRIDE];
    let mut counts = vec![0u32; nlist];
    let mut sums = vec![0.0f64; nlist * STRIDE];

    for iter in 0..kmeans_iters {
        for i in 0..centroids.len() {
            quantized_centroids[i] =
                centroids[i].round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
        }

        assignments
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, slot)| {
                let vec_base = sample_idx[i] * STRIDE;
                let mut best_cluster = 0usize;
                let mut best_dist = i64::MAX;
                for c in 0..nlist {
                    let centroid_base = c * STRIDE;
                    let dist = l2_squared(vectors, vec_base, &quantized_centroids, centroid_base);
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }
                *slot = best_cluster;
            });

        let mut changed = 0usize;
        for i in 0..sample_size {
            if assignments[i] != previous[i] {
                changed += 1;
                previous[i] = assignments[i];
            }
        }

        counts.fill(0);
        sums.fill(0.0);
        for i in 0..sample_size {
            let cluster = assignments[i];
            counts[cluster] += 1;
            let src_base = sample_idx[i] * STRIDE;
            let dst_base = cluster * STRIDE;
            for d in 0..STRIDE {
                sums[dst_base + d] += vectors[src_base + d] as f64;
            }
        }

        for c in 0..nlist {
            let count = counts[c];
            if count == 0 {
                continue;
            }
            let dst_base = c * STRIDE;
            let inv = 1.0 / count as f64;
            for d in 0..STRIDE {
                centroids[dst_base + d] = sums[dst_base + d] * inv;
            }
        }

        eprintln!(
            "  iter {}/{}: {} reassignments",
            iter + 1,
            kmeans_iters,
            changed
        );
        if changed == 0 {
            break;
        }
    }

    let centroids_f32 = centroids.iter().map(|&v| v as f32).collect();
    let centroids_i16 = centroids
        .iter()
        .map(|&v| v.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16)
        .collect();
    (centroids_f32, centroids_i16)
}
