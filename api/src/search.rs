use crate::types::Label;
use memmap2::Mmap;
use std::fs::File;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const IVF_MAGIC: &[u8; 8] = b"RINHIVF5";
const HEADER_SIZE: usize = 20;
const DIMS: usize = 14;
const STRIDE: usize = 32;
const K: usize = 5;
const NPROBE_FAST: usize = 10;
const NPROBE_RETRY: usize = 142;
const NPROBE_SLOW: usize = NPROBE_FAST + NPROBE_RETRY;
const PADDED_DIMS: usize = 16;

pub struct SearchIndex {
    mmap: Mmap,
    k_clusters: usize,
    n: usize,
    centroids_i16: Vec<i16>,
    bbox_min: Vec<i16>,
    bbox_max: Vec<i16>,
    offsets: Vec<u32>,
    data_byte: usize,
    labels_byte: usize,
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
        let centroids_len = k_clusters * DIMS;
        let bbox_byte = centroids_byte + centroids_len * 2;
        let offsets_byte = bbox_byte + centroids_len * 2 * 2;
        let data_byte = offsets_byte + (k_clusters + 1) * 4;
        let labels_byte = data_byte + n * STRIDE;

        let mut centroids_i16 = Vec::with_capacity(centroids_len);
        for i in 0..centroids_len {
            let off = centroids_byte + i * 2;
            centroids_i16.push(i16::from_le_bytes(
                mmap[off..off + 2].try_into().unwrap(),
            ));
        }

        let mut bbox_min = Vec::with_capacity(centroids_len);
        let mut bbox_max = Vec::with_capacity(centroids_len);
        for i in 0..centroids_len {
            let off_min = bbox_byte + i * 2;
            let off_max = bbox_byte + centroids_len * 2 + i * 2;
            bbox_min.push(i16::from_le_bytes(mmap[off_min..off_min + 2].try_into().unwrap()));
            bbox_max.push(i16::from_le_bytes(mmap[off_max..off_max + 2].try_into().unwrap()));
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
            bbox_min,
            bbox_max,
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
    }

    pub fn search(&self, query: &[i16; DIMS]) -> [Label; K] {
        self.search_impl(query)
    }

    pub fn search_with_vector(&self, _query_f32: &[f32; DIMS], query: &[i16; DIMS]) -> [Label; K] {
        self.search(query)
    }

    fn search_impl(&self, query: &[i16; DIMS]) -> [Label; K] {
        let vectors = &self.mmap[self.data_byte..self.labels_byte];
        let labels = &self.mmap[self.labels_byte..];

        let nprobe_slow = NPROBE_SLOW.min(self.k_clusters);
        let nprobe_fast = NPROBE_FAST.min(nprobe_slow);
        let mut best = [(u64::MAX, 0usize); NPROBE_SLOW];

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
            if du < best[nprobe_slow - 1].0 {
                best[nprobe_slow - 1] = (du, ci);
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

        let query_simd = pad_query_i16(query);

        self.scan_clusters(
            query,
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

        if fraud_count == 2 || fraud_count == 3 {
            let retry_end = nprobe_slow.min(nprobe_fast + NPROBE_RETRY);
            for &ci in &probed[nprobe_fast..retry_end] {
                if top_len == K {
                    let lb = bbox_lower_bound(query, &self.bbox_min, &self.bbox_max, ci);
                    if lb >= worst_dist {
                        continue;
                    }
                }
                self.scan_clusters(
                    query,
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

    fn scan_clusters(
        &self,
        query: &[i16; DIMS],
        _query_simd: &[i16; PADDED_DIMS],
        vectors: &[u8],
        labels: &[u8],
        clusters: &[usize],
        top: &mut [(i64, u8); K],
        top_len: &mut usize,
        worst_dist: &mut i64,
        worst_pos: &mut usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                self.scan_clusters_avx2(
                    _query_simd,
                    vectors,
                    labels,
                    clusters,
                    top,
                    top_len,
                    worst_dist,
                    worst_pos,
                );
            }
            return;
        }

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
fn bbox_lower_bound(query: &[i16; DIMS], bbox_min: &[i16], bbox_max: &[i16], ci: usize) -> i64 {
    let base = ci * DIMS;
    let mut lb: i64 = 0;
    for d in 0..DIMS {
        let q = query[d] as i32;
        let lo = bbox_min[base + d] as i32;
        let hi = bbox_max[base + d] as i32;
        let diff = if q < lo {
            lo - q
        } else if q > hi {
            q - hi
        } else {
            0
        };
        lb += (diff as i64) * (diff as i64);
    }
    lb
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

#[inline(always)]
fn pad_query_i16(query: &[i16; DIMS]) -> [i16; PADDED_DIMS] {
    let mut padded = [0i16; PADDED_DIMS];
    padded[..DIMS].copy_from_slice(query);
    padded
}

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

        for _ in 0..DIMS {
            f.write_all(&0i16.to_le_bytes()).unwrap();
        }
        for _ in 0..DIMS {
            f.write_all(&0i16.to_le_bytes()).unwrap();
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
