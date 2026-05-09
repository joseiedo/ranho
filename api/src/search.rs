use crate::types::Label;
use memmap2::Mmap;
use std::collections::BinaryHeap;
use std::fs::File;

const IVF_MAGIC: &[u8; 8] = b"RINHIVF1";
/// Fixed header size: magic(8) + K(4) + N(4) + dims(4)
const HEADER_SIZE: usize = 20;
const DIMS: usize = 14;
/// KNN k
const K: usize = 5;
/// Clusters probed on first pass (fast path).
const FAST_NPROBE: usize = 8;
/// Clusters probed when fraud_count ∈ {2,3} (decision boundary — re-probe with more clusters).
const FULL_NPROBE: usize = 24;
/// Dimension split for early-exit: after this many dims, check partial dist vs heap threshold.
const EARLY_EXIT_DIM: usize = 8;
const SENTINEL: i8 = -127;
/// Max squared distance for one i8 dimension (254² = 64516).
/// Applied when exactly one of query/reference carries the sentinel on dim 5 or 6.
const SENTINEL_PENALTY: i32 = 64516;

pub struct SearchIndex {
    mmap: Mmap,
    k_clusters: usize,
    n: usize,
    /// Parsed centroids — avoids byte-parsing on every query.
    centroids: Vec<[f32; DIMS]>,
    /// Byte offset into `mmap` where the (K+1) u32 cluster-start offsets begin.
    offsets_byte: usize,
    /// Byte offset into `mmap` where the flat N×15-byte record data begins.
    data_byte: usize,
}

impl SearchIndex {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }?;

        if mmap.len() < HEADER_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file too small",
            ));
        }
        if &mmap[0..8] != IVF_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid magic bytes",
            ));
        }

        let k_clusters = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
        let n = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let dims = u32::from_le_bytes(mmap[16..20].try_into().unwrap()) as usize;
        if dims != DIMS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unexpected dims in index header",
            ));
        }

        let centroids_byte = HEADER_SIZE;
        let offsets_byte = centroids_byte + k_clusters * DIMS * 4;
        let data_byte = offsets_byte + (k_clusters + 1) * 4;

        // Parse centroids once at load time.
        let mut centroids = Vec::with_capacity(k_clusters);
        for ci in 0..k_clusters {
            let base = centroids_byte + ci * DIMS * 4;
            let mut c = [0.0f32; DIMS];
            for d in 0..DIMS {
                let off = base + d * 4;
                c[d] = f32::from_le_bytes(mmap[off..off + 4].try_into().unwrap());
            }
            centroids.push(c);
        }

        Ok(Self {
            mmap,
            k_clusters,
            n,
            centroids,
            offsets_byte,
            data_byte,
        })
    }

    pub fn count(&self) -> usize {
        self.n
    }

    pub fn search(&self, query: &[i8; DIMS]) -> [Label; K] {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: AVX2 is guaranteed by RUSTFLAGS="-C target-cpu=haswell".
        return unsafe { self.search_avx2(query) };
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is mandatory on all AArch64 CPUs.
        return unsafe { self.search_neon(query) };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        return self.search_impl(query);
    }

    /// Hot path for x86_64: compiled with AVX2 so the compiler auto-vectorizes dist_scalar.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn search_avx2(&self, query: &[i8; DIMS]) -> [Label; K] {
        self.search_impl(query)
    }

    /// Hot path for aarch64: compiled with NEON so the compiler auto-vectorizes dist_scalar.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn search_neon(&self, query: &[i8; DIMS]) -> [Label; K] {
        self.search_impl(query)
    }

    fn search_impl(&self, query: &[i8; DIMS]) -> [Label; K] {
        let offsets = &self.mmap[self.offsets_byte..];
        let data = &self.mmap[self.data_byte..];

        // Phase 1: dequantize query and find the top-FULL_NPROBE closest centroids,
        // sorted so we can probe them in order (closest first).
        let mut query_f32 = [0.0f32; DIMS];
        for d in 0..DIMS {
            query_f32[d] = query[d] as f32 / 127.0;
        }

        let full_nprobe = FULL_NPROBE.min(self.k_clusters);
        let fast_nprobe = FAST_NPROBE.min(full_nprobe);

        let mut centroid_dists: Vec<(u32, usize)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(ci, c)| {
                let mut d = 0.0f32;
                for i in 0..DIMS {
                    let diff = query_f32[i] - c[i];
                    d += diff * diff;
                }
                (d.to_bits(), ci)
            })
            .collect();
        // Bring the top-full_nprobe to the front, then sort them so closest come first.
        centroid_dists.select_nth_unstable(full_nprobe - 1);
        centroid_dists[..full_nprobe].sort_unstable();

        // Phase 2: scan top-fast_nprobe clusters with early-exit distance pruning.
        let mut heap: BinaryHeap<(i32, u8)> = BinaryHeap::with_capacity(K + 1);
        scan_clusters(query, &centroid_dists[..fast_nprobe], offsets, data, &mut heap);

        // Adaptive: if fraud_count ∈ {2,3} the result sits on the decision boundary
        // (score 0.4 or 0.6). Probe the next batch of clusters for better accuracy.
        if full_nprobe > fast_nprobe && heap.len() == K {
            let fraud_in_heap = heap.iter().filter(|&&(_, l)| l == 1).count();
            if fraud_in_heap == 2 || fraud_in_heap == 3 {
                scan_clusters(
                    query,
                    &centroid_dists[fast_nprobe..full_nprobe],
                    offsets,
                    data,
                    &mut heap,
                );
            }
        }

        let mut result = [Label::Legit; K];
        for (slot, (_, label_byte)) in result.iter_mut().zip(heap.into_iter()) {
            if label_byte == 1 {
                *slot = Label::Fraud;
            }
        }
        result
    }
}

/// Scan a list of clusters, inserting candidates into `heap` with early-exit pruning.
/// Threshold is kept tight: updated whenever the heap changes.
#[inline]
fn scan_clusters(
    query: &[i8; DIMS],
    clusters: &[(u32, usize)],
    offsets: &[u8],
    data: &[u8],
    heap: &mut BinaryHeap<(i32, u8)>,
) {
    for &(_, ci) in clusters {
        let start = u32::from_le_bytes(offsets[ci * 4..ci * 4 + 4].try_into().unwrap()) as usize;
        let end =
            u32::from_le_bytes(offsets[(ci + 1) * 4..(ci + 1) * 4 + 4].try_into().unwrap())
                as usize;

        // threshold = worst distance in the top-K heap; i32::MAX when heap not full.
        let mut threshold = heap.peek().map_or(i32::MAX, |&(d, _)| d);

        for j in start..end {
            let base = j * (DIMS + 1);
            if let Some(dist) = dist_with_early_exit(query, &data[base..base + DIMS], threshold) {
                if heap.len() < K {
                    heap.push((dist, data[base + DIMS]));
                    if heap.len() == K {
                        threshold = heap.peek().unwrap().0;
                    }
                } else {
                    heap.pop();
                    heap.push((dist, data[base + DIMS]));
                    threshold = heap.peek().unwrap().0;
                }
            }
        }
    }
}

/// Full distance — used by tests via `dist_scalar`.
#[inline]
fn dist_scalar(query: &[i8; DIMS], record: &[u8]) -> i32 {
    dist_with_early_exit(query, record, i32::MAX).unwrap()
}

/// Compute squared Euclidean distance with sentinel handling and early exit.
/// After `EARLY_EXIT_DIM` dimensions, if the partial sum already meets or exceeds
/// `threshold`, returns `None` (vector is pruned). Dims 8-13 have no sentinels.
#[inline]
fn dist_with_early_exit(query: &[i8; DIMS], record: &[u8], threshold: i32) -> Option<i32> {
    let mut dist: i32 = 0;

    // First EARLY_EXIT_DIM dims (includes sentinel dims 5 and 6).
    for d in 0..EARLY_EXIT_DIM {
        let q = query[d];
        let r = record[d] as i8;
        dist += if (d == 5 || d == 6) && (q == SENTINEL || r == SENTINEL) {
            if q == SENTINEL && r == SENTINEL { 0 } else { SENTINEL_PENALTY }
        } else {
            let diff = q as i16 - r as i16;
            (diff * diff) as i32
        };
    }

    if dist >= threshold {
        return None;
    }

    // Remaining dims — no sentinels past dim 7.
    for d in EARLY_EXIT_DIM..DIMS {
        let diff = query[d] as i16 - record[d] as i8 as i16;
        dist += (diff * diff) as i32;
    }

    Some(dist)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn quantize(v: f32) -> i8 {
        (v * 127.0).round().clamp(-127.0, 127.0) as i8
    }

    /// Write a tiny IVF index with K=1 (single cluster = brute-force over all records).
    fn make_index(records: &[([f32; DIMS], u8)]) -> NamedTempFile {
        let n = records.len();
        let k: usize = 1;
        let mut f = NamedTempFile::new().unwrap();

        // Header
        f.write_all(IVF_MAGIC).unwrap();
        f.write_all(&(k as u32).to_le_bytes()).unwrap();
        f.write_all(&(n as u32).to_le_bytes()).unwrap();
        f.write_all(&(DIMS as u32).to_le_bytes()).unwrap();

        // One centroid: all zeros (doesn't matter — K=1 always probes it)
        for _ in 0..DIMS {
            f.write_all(&0.0f32.to_le_bytes()).unwrap();
        }

        // Offsets: cluster 0 starts at record 0, ends at record N
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&(n as u32).to_le_bytes()).unwrap();

        // Flat data
        for (vec, label) in records {
            for &v in vec {
                f.write_all(&[quantize(v) as u8]).unwrap();
            }
            f.write_all(&[*label]).unwrap();
        }
        f.flush().unwrap();
        f
    }

    #[test]
    fn header_validates_magic() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(b"BADMAGIC").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&14u32.to_le_bytes()).unwrap();
        f.flush().unwrap();
        let result = SearchIndex::open(f.path().to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn header_validates_dims() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(IVF_MAGIC).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap(); // K
        f.write_all(&1u32.to_le_bytes()).unwrap(); // N
        f.write_all(&13u32.to_le_bytes()).unwrap(); // wrong dims
        f.flush().unwrap();
        let result = SearchIndex::open(f.path().to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn search_returns_correct_labels_20_vectors() {
        // 15 legit near origin, 5 fraud far away.
        // Query at origin → nearest 5 should all be legit.
        let mut records: Vec<([f32; DIMS], u8)> = Vec::new();
        for i in 0..15usize {
            let mut v = [0.0f32; DIMS];
            v[0] = 0.01 * i as f32;
            records.push((v, 0));
        }
        for _ in 0..5 {
            records.push(([1.0f32; DIMS], 1));
        }

        let f = make_index(&records);
        let idx = SearchIndex::open(f.path().to_str().unwrap()).unwrap();
        assert_eq!(idx.count(), 20);

        let query = [quantize(0.0); DIMS];
        let neighbors = idx.search(&query);
        let fraud_count = neighbors.iter().filter(|&&l| l == Label::Fraud).count();
        assert_eq!(fraud_count, 0, "expected all legit neighbors");
    }

    #[test]
    fn search_returns_fraud_when_nearest() {
        // 5 fraud near origin, 15 legit far away.
        let mut records: Vec<([f32; DIMS], u8)> = Vec::new();
        for _ in 0..5 {
            records.push(([0.0f32; DIMS], 1));
        }
        for _ in 0..15 {
            records.push(([1.0f32; DIMS], 0));
        }

        let f = make_index(&records);
        let idx = SearchIndex::open(f.path().to_str().unwrap()).unwrap();

        let query = [quantize(0.0); DIMS];
        let neighbors = idx.search(&query);
        let fraud_count = neighbors.iter().filter(|&&l| l == Label::Fraud).count();
        assert_eq!(fraud_count, 5, "expected all fraud neighbors");
    }

    #[test]
    fn sentinel_both_contribute_zero() {
        let dist = dist_scalar(
            &{
                let mut q = [0i8; DIMS];
                q[5] = SENTINEL;
                q
            },
            &{
                let mut r = [0u8; DIMS];
                r[5] = SENTINEL as u8;
                r
            },
        );
        assert_eq!(dist, 0, "both sentinels → 0 distance");
    }

    #[test]
    fn sentinel_one_side_contributes_penalty() {
        let dist = dist_scalar(
            &{
                let mut q = [0i8; DIMS];
                q[5] = SENTINEL;
                q
            },
            &[0u8; DIMS],
        );
        assert_eq!(dist, SENTINEL_PENALTY);
    }
}
