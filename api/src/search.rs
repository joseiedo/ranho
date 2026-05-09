use crate::types::Label;
use memmap2::Mmap;
use std::fs::File;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// IVF binary file magic (8 bytes).
const IVF_MAGIC: &[u8; 8] = b"RINHIVF6";

// Header: magic(8) + k_clusters(4) + n(4) + dims(4) = 20 bytes (all u32 LE).
const HEADER_SIZE: usize = 20;

// Each reference vector has 14 features (see vectorizer.rs).
const DIMS: usize = 14;

// Stored vectors: 14 i16 values (28 bytes) + 4 zero-padding bytes = 32 bytes.
// Padding ensures 256-bit AVX2 loads are always aligned within the stride.
const STRIDE: usize = 32;

// Top-K nearest neighbors returned per query.
const K: usize = 5;

// Two-phase IVF probe strategy:
//   Phase 1 (fast): always scan the 8 nearest centroids.
//   Phase 2 (retry): conditionally scan up to 144 additional centroids
//     when fraud_count ∈ {2, 3} (ambiguous). Total centroids considered: 152.
const NPROBE_FAST: usize = 8;
const NPROBE_RETRY: usize = 144;
const NPROBE_SLOW: usize = NPROBE_FAST + NPROBE_RETRY;

// AVX2 operates on 256-bit (32 byte) registers = 16 × i16.
// The 14-dim query is zero-padded to 16 elements for SIMD loads.
const PADDED_DIMS: usize = 16;

// Memory-mapped IVF index. All vector data resides in the mmap'd file;
// metadata (centroids, radii, offsets) is eagerly loaded into Vecs for fast random access.
//
// Binary layout of the mapped file:
//   header(20) | centroids(k×14×2) | radii(k×4) | offsets((k+1)×4) | data(n×32) | labels(n×1)
pub struct SearchIndex {
    mmap: Mmap,
    k_clusters: usize,
    n: usize,
    centroids_i16: Vec<i16>,
    radii: Vec<i32>,
    offsets: Vec<u32>,
    data_byte: usize,
    labels_byte: usize,
}

impl SearchIndex {
    /// Memory-maps the IVF file and validates its header (magic, dims).
    /// Centroids, radii, and cluster offsets are eagerly parsed into Vecs
    /// so the hot path only needs slice indexing (no per-access byte parsing).
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

        // Byte offsets for each section (see binary layout above).
        let centroids_byte = HEADER_SIZE;
        let centroids_len = k_clusters * DIMS;
        let radii_byte = centroids_byte + centroids_len * 2;
        let offsets_byte = radii_byte + k_clusters * 4;
        let data_byte = offsets_byte + (k_clusters + 1) * 4;
        let labels_byte = data_byte + n * STRIDE;

        let mut centroids_i16 = Vec::with_capacity(centroids_len);
        for i in 0..centroids_len {
            let off = centroids_byte + i * 2;
            centroids_i16.push(i16::from_le_bytes(
                mmap[off..off + 2].try_into().unwrap(),
            ));
        }

        let mut radii = Vec::with_capacity(k_clusters);
        for i in 0..k_clusters {
            let off = radii_byte + i * 4;
            radii.push(i32::from_le_bytes(mmap[off..off + 4].try_into().unwrap()));
        }

        let mut offsets = Vec::with_capacity(k_clusters + 1);
        for i in 0..=k_clusters {
            let base = offsets_byte + i * 4;
            offsets.push(u32::from_le_bytes(mmap[base..base + 4].try_into().unwrap()));
        }

        Ok(Self {
            mmap,
            k_clusters,
            n,
            centroids_i16,
            radii,
            offsets,
            data_byte,
            labels_byte,
        })
    }

    pub fn count(&self) -> usize {
        self.n
    }

    /// Two-part warmup before serving traffic:
    /// 1. Touch every 64th byte of the data section and XOR offsets to force
    ///    page faults — pre-populates the OS page cache with the index.
    /// 2. Run 500 dummy searches with pseudo-random query vectors (LCG) to
    ///    warm up the CPU's instruction cache and branch predictor for the hot path.
    pub fn warmup(&self) {
        let mut sink: u64 = 0;
        for &o in &self.offsets {
            sink ^= o as u64;
        }
        for b in self.mmap[self.data_byte..].iter().step_by(64) {
            sink ^= *b as u64;
        }
        let _ = sink;

        // Linear congruential generator: x_{n+1} = x_n * 1664525 + 1013904223
        // Values clamped to [-10000, 10000] in i16 range (matching quantized feature range).
        let mut state = 0x12345678u32;
        for _ in 0..500 {
            let mut q = [0i16; DIMS];
            for v in q.iter_mut() {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *v = (((state >> 16) % 20_001) as i32 - 10_000) as i16;
            }
            let _ = self.search(&q);
        }
    }

    pub fn search(&self, query: &[i16; DIMS]) -> [Label; K] {
        self.search_impl(query)
    }

    pub fn search_with_vector(&self, _query_f32: &[f32; DIMS], query: &[i16; DIMS]) -> [Label; K] {
        self.search(query)
    }

    /// Two-phase IVF search with radius-based pruning.
    ///
    /// Phase 1 (fast):
    ///   1. Compute squared i64 distance from query to all k centroids.
    ///   2. Insertion-sort the top-NPROBE_SLOW (152) nearest centroids
    ///      into a fixed-size best array — no heap allocations.
    ///   3. Scan only the NPROBE_FAST (8) nearest clusters.
    ///
    /// Phase 2 (retry) — only if fraud_count ∈ {2, 3} from phase 1:
    ///   For each of the next NPROBE_RETRY (144) clusters:
    ///     a. Compute radius lower bound:
    ///        lb = max(0, isqrt(dist_to_centroid) - cluster_radius)²
    ///     b. Skip cluster entirely if lb >= worst_dist (can't beat current top-K).
    ///     c. Otherwise scan the cluster.
    ///
    /// All distance calculations use i64 integer arithmetic — no f32 in the hot path.
    /// Distance is always computed via AVX2 (target-cpu=haswell ensures availability).
    fn search_impl(&self, query: &[i16; DIMS]) -> [Label; K] {
        let vectors = &self.mmap[self.data_byte..self.labels_byte];
        let labels = &self.mmap[self.labels_byte..];

        let nprobe_slow = NPROBE_SLOW.min(self.k_clusters);
        let nprobe_fast = NPROBE_FAST.min(nprobe_slow);
        let mut best = [(u64::MAX, 0usize); NPROBE_SLOW];

        // Compute distance from query to every centroid (i64 squared Euclidean).
        // Keep the top-NPROBE_SLOW via insertion sort into a fixed-size array.
        for ci in 0..self.k_clusters {
            let base = ci * DIMS;
            let mut dist: i64 = 0;
            for d in 0..DIMS {
                let diff = query[d] as i64 - self.centroids_i16[base + d] as i64;
                dist += diff * diff;
            }
            if dist < 0 {
                continue;
            }
            let du = dist as u64;
            // Only insert if it beats the current worst in the top-NPROBE_SLOW.
            if du < best[nprobe_slow - 1].0 {
                best[nprobe_slow - 1] = (du, ci);
                let mut i = nprobe_slow - 1;
                while i > 0 && best[i].0 < best[i - 1].0 {
                    best.swap(i, i - 1);
                    i -= 1;
                }
            }
        }

        // Extract just the cluster indices from the sorted best array.
        let mut probed = [0usize; NPROBE_SLOW];
        for (slot, &(_, ci)) in probed[..nprobe_slow]
            .iter_mut()
            .zip(best[..nprobe_slow].iter())
        {
            *slot = ci;
        }

        // Top-K heap: unsorted array of (distance, label_byte).
        // worst_dist tracks the K-th best (largest) distance; worst_pos is its index.
        let mut top = [(i64::MAX, 0u8); K];
        let mut top_len = 0usize;
        let mut worst_dist = i64::MAX;
        let mut worst_pos = 0usize;

        let query_simd = pad_query_i16(query);

        // Phase 1: scan the NPROBE_FAST nearest clusters.
        self.scan_clusters(
            &query_simd,
            vectors,
            labels,
            &probed[..nprobe_fast],
            &mut top,
            &mut top_len,
            &mut worst_dist,
            &mut worst_pos,
        );

        let result = labels_to_result(&top);
        let fraud_count = result.iter().filter(|&&l| l == Label::Fraud).count();

        // Phase 2: ambiguous result (2 or 3 fraud among top-5) → probe more clusters.
        // Radius-based pruning: skip clusters that can't possibly beat the current worst.
        if fraud_count == 2 || fraud_count == 3 {
            let retry_end = nprobe_slow.min(nprobe_fast + NPROBE_RETRY);
            for &ci in &probed[nprobe_fast..retry_end] {
                if top_len == K {
                    let mut cdist: i64 = 0;
                    let base = ci * DIMS;
                    for d in 0..DIMS {
                        let diff = query[d] as i64 - self.centroids_i16[base + d] as i64;
                        cdist += diff * diff;
                    }
                    // Lower bound on distance to any vector in this cluster.
                    // If the query is inside the cluster radius, lower bound is 0
                    // (the cluster might contain a perfect match).
                    let lb = radius_lower_bound(cdist, self.radii[ci]);
                    if lb >= worst_dist {
                        continue;
                    }
                }
                self.scan_clusters(
                    &query_simd,
                    vectors,
                    labels,
                    &[ci],
                    &mut top,
                    &mut top_len,
                    &mut worst_dist,
                    &mut worst_pos,
                );
            }
            labels_to_result(&top)
        } else {
            result
        }
    }

    /// Scan one or more clusters via AVX2 distance, updating the top-K heap.
    ///
    /// Top-K maintenance: when we have fewer than K candidates, insert directly.
    /// Once full, replace the worst (largest distance) only when a closer vector is found,
    /// then re-scan to find the new worst.
    #[cfg(target_arch = "x86_64")]
    fn scan_clusters(
        &self,
        query_simd: &[i16; PADDED_DIMS],
        vectors: &[u8],
        labels: &[u8],
        clusters: &[usize],
        top: &mut [(i64, u8); K],
        top_len: &mut usize,
        worst_dist: &mut i64,
        worst_pos: &mut usize,
    ) {
        unsafe {
            self.scan_clusters_avx2(
                query_simd,
                vectors,
                labels,
                clusters,
                top,
                top_len,
                worst_dist,
                worst_pos,
            );
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[allow(unused_variables)]
    fn scan_clusters(
        &self,
        query_simd: &[i16; PADDED_DIMS],
        vectors: &[u8],
        labels: &[u8],
        clusters: &[usize],
        top: &mut [(i64, u8); K],
        top_len: &mut usize,
        worst_dist: &mut i64,
        worst_pos: &mut usize,
    ) {
        unimplemented!("AVX2 is required; build for x86_64");
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn scan_clusters_avx2(
        &self,
        query: &[i16; PADDED_DIMS],
        vectors: &[u8],
        labels: &[u8],
        clusters: &[usize],
        top: &mut [(i64, u8); K],
        top_len: &mut usize,
        worst_dist: &mut i64,
        worst_pos: &mut usize,
    ) {
        for &ci in clusters {
            let start = self.offsets[ci] as usize;
            let end = self.offsets[ci + 1] as usize;

            for j in start..end {
                let base = j * STRIDE;
                let dist = dist_avx2(query, &vectors[base..base + STRIDE]);

                if *top_len < K {
                    top[*top_len] = (dist, labels[j]);
                    *top_len += 1;
                    if *top_len == K {
                        *worst_pos = 0;
                        for i in 1..K {
                            if top[i].0 > top[*worst_pos].0 {
                                *worst_pos = i;
                            }
                        }
                        *worst_dist = top[*worst_pos].0;
                    }
                } else if dist < *worst_dist {
                    top[*worst_pos] = (dist, labels[j]);
                    *worst_pos = 0;
                    for i in 1..K {
                        if top[i].0 > top[*worst_pos].0 {
                            *worst_pos = i;
                        }
                    }
                    *worst_dist = top[*worst_pos].0;
                }
            }
        }
    }
}

// Convert the raw [(i64, u8); K] distance-label array to [Label; K].
// Label byte 0 = Legit, 1 = Fraud (matching the preprocessor encoding).
#[inline(always)]
fn labels_to_result(top: &[(i64, u8); K]) -> [Label; K] {
    let mut result = [Label::Legit; K];
    for (slot, &(_, label_byte)) in result.iter_mut().zip(top.iter()) {
        if label_byte == 1 {
            *slot = Label::Fraud;
        }
    }
    result
}

// Radius-based cluster pruning lower bound.
// Given the query's squared distance to the cluster centroid (dist_sq) and the
// cluster's radius (max distance from centroid to any member), the minimum possible
// distance from the query to any member of this cluster is:
//   lb = max(0, sqrt(dist_sq) - radius)²
// If lb >= worst_dist, the entire cluster can be skipped safely.
#[inline(always)]
fn radius_lower_bound(dist_sq: i64, radius: i32) -> i64 {
    if dist_sq <= 0 || radius <= 0 {
        return dist_sq;
    }
    let dist = isqrt(dist_sq as u64) as i64;
    let diff = (dist - radius as i64).max(0);
    diff * diff
}

// Integer square root via Newton's method (Heron's method).
// Avoids libm calls and f64 arithmetic. Converges in ≤ 5 iterations for u64 range.
// Used in both radius-based pruning (search path) and preprocessor radii computation.
#[inline(always)]
fn isqrt(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let mut x = n;
    let mut y = (x + 1) >> 1;
    while y < x {
        x = y;
        y = (x + n / x) >> 1;
    }
    x
}

// Zero-pad a 14-dim query to 16 elements so AVX2 can load it as one 256-bit register.
#[inline(always)]
fn pad_query_i16(query: &[i16; DIMS]) -> [i16; PADDED_DIMS] {
    let mut padded = [0i16; PADDED_DIMS];
    padded[..DIMS].copy_from_slice(query);
    padded
}

// AVX2 distance: squared Euclidean in a single SIMD pass.
//
// Algorithm per vector:
//   1. Load query (16 i16) and record (16 i16 from 32-byte stride) into __m256i.
//   2. _mm256_sub_epi16  → pairwise i16 differences.
//   3. _mm256_madd_epi16 → adjacent pairs multiplied and accumulated as i32
//      (computes d0²+d1², d2²+d3², ..., d14²+d15² — last pair is padding zeros).
//   4. Horizontal reduction via _mm_hadd_epi32 chain → single i32 → i64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dist_avx2(query: &[i16; PADDED_DIMS], record: &[u8]) -> i64 {
    let q = _mm256_loadu_si256(query.as_ptr().cast::<__m256i>());
    let r = _mm256_loadu_si256(record.as_ptr().cast::<__m256i>());
    let diff = _mm256_sub_epi16(q, r);
    let squares = _mm256_madd_epi16(diff, diff);

    let lo = _mm256_castsi256_si128(squares);
    let hi = _mm256_extracti128_si256(squares, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    let sum64 = _mm_hadd_epi32(sum128, sum128);
    let sum32 = _mm_hadd_epi32(sum64, sum64);

    _mm_cvtsi128_si32(sum32) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn quantize(v: f32) -> i16 {
        (v * 10_000.0)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16
    }

    fn make_index(records: &[([f32; DIMS], u8)]) -> NamedTempFile {
        let n = records.len();
        let k: usize = 1;
        let mut f = NamedTempFile::new().unwrap();

        f.write_all(IVF_MAGIC).unwrap();
        f.write_all(&(k as u32).to_le_bytes()).unwrap();
        f.write_all(&(n as u32).to_le_bytes()).unwrap();
        f.write_all(&(DIMS as u32).to_le_bytes()).unwrap();

        let centroid_f32 = [0.0f32; DIMS];
        for &v in &centroid_f32 {
            f.write_all(&quantize(v).to_le_bytes()).unwrap();
        }

        for _ in 0..k {
            f.write_all(&0i32.to_le_bytes()).unwrap();
        }

        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&(n as u32).to_le_bytes()).unwrap();

        for (vec, _) in records {
            for &v in vec {
                f.write_all(&quantize(v).to_le_bytes()).unwrap();
            }
            f.write_all(&[0u8; 4]).unwrap();
        }
        for (_, label) in records {
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
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&13u32.to_le_bytes()).unwrap();
        f.write_all(&[0u8; 28]).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&[0u8; 32]).unwrap();
        f.write_all(&[0u8]).unwrap();
        f.flush().unwrap();
        let result = SearchIndex::open(f.path().to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn search_returns_correct_labels_20_vectors() {
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
        let mut q = [0i16; DIMS];
        q[5] = quantize(-1.0);
        let q_padded = pad_query_i16(&q);
        let mut r = [0u8; STRIDE];
        r[10..12].copy_from_slice(&quantize(-1.0).to_le_bytes());
        unsafe {
            let dist = dist_avx2(&q_padded, &r);
            assert_eq!(dist, 0, "both sentinels → 0 distance");
        }
    }

    #[test]
    fn sentinel_one_side_contributes_penalty() {
        let mut q = [0i16; DIMS];
        q[5] = quantize(-1.0);
        let q_padded = pad_query_i16(&q);
        unsafe {
            let dist = dist_avx2(&q_padded, &[0u8; STRIDE]);
            let diff = quantize(-1.0) as i64;
            assert_eq!(dist, diff * diff);
        }
    }

    #[test]
    fn centroids_are_i16() {
        let records: Vec<([f32; DIMS], u8)> = vec![([0.5f32; DIMS], 0)];
        let f = make_index(&records);
        let idx = SearchIndex::open(f.path().to_str().unwrap()).unwrap();
        assert_eq!(idx.k_clusters, 1);
        for d in 0..DIMS {
            assert_eq!(idx.centroids_i16[d], quantize(0.0));
        }
    }
}
