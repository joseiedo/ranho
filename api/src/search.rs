use crate::types::Label;
use memmap2::Mmap;
use std::fs::File;

const IVF_MAGIC: &[u8; 8] = b"RINHIVF3";
const HEADER_SIZE: usize = 20;
const DIMS: usize = 14;
const STRIDE: usize = 32;
const K: usize = 5;
const NPROBE_FAST: usize = 8;
const NPROBE_SLOW: usize = 48;

pub struct SearchIndex {
    mmap: Mmap,
    k_clusters: usize,
    n: usize,
    centroids: Vec<[f32; DIMS]>,
    offsets: Vec<u32>,
    data_byte: usize,
    labels_byte: usize,
}

impl SearchIndex {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }?;

        #[cfg(target_os = "linux")]
        {
            let _ = mmap.advise(memmap2::Advice::HugePage);
            let _ = mmap.advise(memmap2::Advice::Random);
        }

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
        let labels_byte = data_byte + n * STRIDE;

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

        let mut offsets = Vec::with_capacity(k_clusters + 1);
        for i in 0..=k_clusters {
            let base = offsets_byte + i * 4;
            offsets.push(u32::from_le_bytes(mmap[base..base + 4].try_into().unwrap()));
        }

        Ok(Self {
            mmap,
            k_clusters,
            n,
            centroids,
            offsets,
            data_byte,
            labels_byte,
        })
    }

    pub fn count(&self) -> usize {
        self.n
    }

    pub fn warmup(&self) {
        let mut sink: u64 = 0;
        for &o in &self.offsets {
            sink ^= o as u64;
        }
        for b in self.mmap[self.data_byte..].iter().step_by(64) {
            sink ^= *b as u64;
        }
        let _ = sink;

        let mut state = 0x12345678u32;
        for _ in 0..500 {
            let mut q = [0i16; DIMS];
            for v in q.iter_mut() {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *v = (((state >> 16) % 20_001) as i32 - 10_000) as i16;
            }
            let _ = self.search(&q);
        }

        #[cfg(target_os = "linux")]
        let _ = self.mmap.lock();
    }

    pub fn search(&self, query: &[i16; DIMS]) -> [Label; K] {
        let mut query_f32 = [0.0f32; DIMS];
        for d in 0..DIMS {
            query_f32[d] = query[d] as f32 / 10_000.0;
        }
        self.search_impl(&query_f32, query)
    }

    pub fn search_with_vector(&self, query_f32: &[f32; DIMS], query: &[i16; DIMS]) -> [Label; K] {
        self.search_impl(query_f32, query)
    }

    fn search_impl(&self, query_f32: &[f32; DIMS], query: &[i16; DIMS]) -> [Label; K] {
        #[cfg(target_arch = "x86_64")]
        return unsafe { self.search_avx2(query_f32, query) };
        #[cfg(target_arch = "aarch64")]
        return unsafe { self.search_neon(query_f32, query) };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        return self.search_impl_inner(query_f32, query);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn search_avx2(&self, query_f32: &[f32; DIMS], query: &[i16; DIMS]) -> [Label; K] {
        // Zero-pad query to 16 i16s. Record bytes 28-31 are zero (preprocessor padding),
        // so positions 14-15 contribute 0 to the distance in both sides.
        let mut q_padded = [0i16; 16];
        q_padded[..DIMS].copy_from_slice(query);

        let vectors = &self.mmap[self.data_byte..self.labels_byte];
        let labels = &self.mmap[self.labels_byte..];

        let nprobe_slow = NPROBE_SLOW.min(self.k_clusters);
        let nprobe_fast = NPROBE_FAST.min(nprobe_slow);
        let mut best = [(u32::MAX, 0usize); NPROBE_SLOW];

        // Centroid search: float arithmetic auto-vectorizes cleanly with avx2 target_feature.
        for (ci, c) in self.centroids.iter().enumerate() {
            let mut d = 0.0f32;
            for i in 0..DIMS {
                let diff = query_f32[i] - c[i];
                d += diff * diff;
            }
            let db = d.to_bits();
            if db < best[nprobe_slow - 1].0 {
                best[nprobe_slow - 1] = (db, ci);
                let mut i = nprobe_slow - 1;
                while i > 0 && best[i].0 < best[i - 1].0 {
                    best.swap(i, i - 1);
                    i -= 1;
                }
            }
        }

        let mut probed = [0usize; NPROBE_SLOW];
        for (slot, &(_, ci)) in probed[..nprobe_slow]
            .iter_mut()
            .zip(best[..nprobe_slow].iter())
        {
            *slot = ci;
        }

        let mut top = [(i64::MAX, 0u8); K];
        let mut top_len = 0usize;
        let mut worst_dist = i64::MAX;
        let mut worst_pos = 0usize;

        scan_avx2(
            &q_padded,
            vectors,
            labels,
            &self.offsets,
            &probed[..nprobe_fast],
            &mut top,
            &mut top_len,
            &mut worst_dist,
            &mut worst_pos,
        );

        let fraud_count = top[..top_len].iter().filter(|&&(_, l)| l == 1).count();
        if top_len >= K && (fraud_count == 0 || fraud_count == 1 || fraud_count == 5) {
            return labels_to_result(&top);
        }

        scan_avx2(
            &q_padded,
            vectors,
            labels,
            &self.offsets,
            &probed[nprobe_fast..nprobe_slow],
            &mut top,
            &mut top_len,
            &mut worst_dist,
            &mut worst_pos,
        );

        labels_to_result(&top)
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn search_neon(&self, query_f32: &[f32; DIMS], query: &[i16; DIMS]) -> [Label; K] {
        self.search_impl_inner(query_f32, query)
    }

    fn search_impl_inner(&self, query_f32: &[f32; DIMS], query: &[i16; DIMS]) -> [Label; K] {
        let vectors = &self.mmap[self.data_byte..self.labels_byte];
        let labels = &self.mmap[self.labels_byte..];

        let nprobe_slow = NPROBE_SLOW.min(self.k_clusters);
        let nprobe_fast = NPROBE_FAST.min(nprobe_slow);
        let mut best = [(u32::MAX, 0usize); NPROBE_SLOW];

        for (ci, c) in self.centroids.iter().enumerate() {
            let mut d = 0.0f32;
            for i in 0..DIMS {
                let diff = query_f32[i] - c[i];
                d += diff * diff;
            }
            let db = d.to_bits();
            if db < best[nprobe_slow - 1].0 {
                best[nprobe_slow - 1] = (db, ci);
                let mut i = nprobe_slow - 1;
                while i > 0 && best[i].0 < best[i - 1].0 {
                    best.swap(i, i - 1);
                    i -= 1;
                }
            }
        }

        let mut probed = [0usize; NPROBE_SLOW];
        for (slot, &(_, ci)) in probed[..nprobe_slow]
            .iter_mut()
            .zip(best[..nprobe_slow].iter())
        {
            *slot = ci;
        }

        let mut top = [(i64::MAX, 0u8); K];
        let mut top_len = 0usize;
        let mut worst_dist = i64::MAX;
        let mut worst_pos = 0usize;

        self.scan_clusters(
            query,
            vectors,
            labels,
            &probed[..nprobe_fast],
            &mut top,
            &mut top_len,
            &mut worst_dist,
            &mut worst_pos,
        );

        let fraud_count = top[..top_len].iter().filter(|&&(_, l)| l == 1).count();

        if top_len >= K && (fraud_count == 0 || fraud_count == 1 || fraud_count == 5) {
            return labels_to_result(&top);
        }

        self.scan_clusters(
            query,
            vectors,
            labels,
            &probed[nprobe_fast..nprobe_slow],
            &mut top,
            &mut top_len,
            &mut worst_dist,
            &mut worst_pos,
        );

        labels_to_result(&top)
    }

    fn scan_clusters(
        &self,
        query: &[i16; DIMS],
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
                let dist = dist_scalar(query, &vectors[base..base + STRIDE]);

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

// AVX2 cluster scan — called only from search_avx2 which already holds the target_feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_avx2(
    q_padded: &[i16; 16],
    vectors: &[u8],
    labels: &[u8],
    offsets: &[u32],
    clusters: &[usize],
    top: &mut [(i64, u8); K],
    top_len: &mut usize,
    worst_dist: &mut i64,
    worst_pos: &mut usize,
) {
    for &ci in clusters {
        let start = offsets[ci] as usize;
        let end = offsets[ci + 1] as usize;

        for j in start..end {
            let base = j * STRIDE;
            let dist = dist_avx2(q_padded, vectors.get_unchecked(base..base + STRIDE));

            if *top_len < K {
                top[*top_len] = (dist, *labels.get_unchecked(j));
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
                top[*worst_pos] = (dist, *labels.get_unchecked(j));
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

// One AVX2 register (256-bit) holds exactly STRIDE=32 bytes = 16 i16s.
// Dimensions 14-15 are zero in both q_padded and the record padding bytes,
// so they contribute 0 to the sum.
//
// Overflow analysis:
//   max |diff| per dim = 20_000 (range [-10_000, 10_000])
//   madd output per pair: diff² + diff² ≤ 2 × 20_000² = 800_000_000 < i32::MAX ✓
//   after one hadd: ≤ 1_600_000_000 < i32::MAX ✓  (safe to extract as i32)
//   final i64 sum: ≤ 4 × 1_600_000_000 = 6_400_000_000 < i64::MAX ✓
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dist_avx2(q_padded: &[i16; 16], record: &[u8]) -> i64 {
    use std::arch::x86_64::*;

    let q = _mm256_loadu_si256(q_padded.as_ptr() as *const __m256i);
    let r = _mm256_loadu_si256(record.as_ptr() as *const __m256i);
    let diff = _mm256_sub_epi16(q, r);
    // vpmaddwd: multiplies adjacent i16 pairs and sums into i32 → 8 i32 results
    let sq = _mm256_madd_epi16(diff, diff);
    // One hadd collapses 8 i32 → 4 i32 partial sums (each ≤ 1.6B, safe in i32)
    let h = _mm256_hadd_epi32(sq, sq);
    let lo = _mm256_castsi256_si128(h);
    let hi = _mm256_extracti128_si256(h, 1);

    _mm_cvtsi128_si32(lo) as i64
        + _mm_extract_epi32::<1>(lo) as i64
        + _mm_cvtsi128_si32(hi) as i64
        + _mm_extract_epi32::<1>(hi) as i64
}

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

#[inline(always)]
fn dist_scalar(query: &[i16; DIMS], record: &[u8]) -> i64 {
    let mut dist: i64 = 0;
    for d in 0..DIMS {
        let off = d * 2;
        let rv = i16::from_le_bytes([record[off], record[off + 1]]);
        let diff = query[d] as i64 - rv as i64;
        dist += diff * diff;
    }
    dist
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

        for _ in 0..DIMS {
            f.write_all(&0.0f32.to_le_bytes()).unwrap();
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
        let mut r = [0u8; STRIDE];
        r[10..12].copy_from_slice(&quantize(-1.0).to_le_bytes());
        let dist = dist_scalar(&q, &r);
        assert_eq!(dist, 0, "both sentinels → 0 distance");
    }

    #[test]
    fn sentinel_one_side_contributes_penalty() {
        let mut q = [0i16; DIMS];
        q[5] = quantize(-1.0);
        let dist = dist_scalar(&q, &[0u8; STRIDE]);
        let diff = quantize(-1.0) as i64;
        assert_eq!(dist, diff * diff);
    }
}
