use crate::types::Label;
use memmap2::Mmap;
use std::cell::RefCell;
use std::fs::File;

const IVF_MAGIC: &[u8; 8] = b"RINHIVF1";
/// Fixed header size: magic(8) + K(4) + N(4) + dims(4)
const HEADER_SIZE: usize = 20;
const DIMS: usize = 14;
/// KNN k
const K: usize = 5;
/// Clusters to probe on the fast path (clear-cut cases).
const NPROBE_FAST: usize = 5;
/// Clusters to probe when the fast result is on the boundary (fraud_count == 2 or 3).
const NPROBE_FULL: usize = 16;
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
    /// Pre-parsed cluster start offsets — avoids per-query mmap reads.
    offsets: Vec<u32>,
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

        // Pre-parse cluster offsets to avoid per-query mmap reads.
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
        })
    }

    pub fn count(&self) -> usize {
        self.n
    }

    /// Touch mmap pages and run random queries to bring the index into page cache
    /// before real traffic arrives.
    pub fn warmup(&self) {
        // Touch offsets (already parsed) and data region.
        let mut sink: u64 = 0;
        for &o in &self.offsets {
            sink ^= o as u64;
        }
        // Sample every 64th byte of the data region (one per cache line) to fault
        // in all pages without reading every byte.
        for b in self.mmap[self.data_byte..].iter().step_by(64) {
            sink ^= *b as u64;
        }
        let _ = sink;

        // Run 500 random queries to warm up the search code paths and TLB.
        let mut state = 0x12345678u32;
        for _ in 0..500 {
            let mut q = [0i8; DIMS];
            for v in q.iter_mut() {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *v = (state >> 24) as i8;
            }
            let _ = self.search(&q);
        }
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
        let data = &self.mmap[self.data_byte..];

        // Phase 1: dequantize query and find closest centroids (thread-local buf).
        let mut query_f32 = [0.0f32; DIMS];
        for d in 0..DIMS {
            query_f32[d] = query[d] as f32 / 127.0;
        }

        let nprobe_fast = NPROBE_FAST.min(self.k_clusters);
        let nprobe_full = NPROBE_FULL.min(self.k_clusters);
        let mut probed = [0usize; NPROBE_FULL];

        CENTROID_BUF.with_borrow_mut(|buf| {
            buf.clear();
            buf.extend(self.centroids.iter().enumerate().map(|(ci, c)| {
                let mut d = 0.0f32;
                for i in 0..DIMS {
                    let diff = query_f32[i] - c[i];
                    d += diff * diff;
                }
                (d.to_bits(), ci)
            }));
            buf.select_nth_unstable(nprobe_full - 1);
            for (slot, &(_, ci)) in probed.iter_mut().zip(buf[..nprobe_full].iter()) {
                *slot = ci;
            }
        });

        // Phase 2: fast scan (NPROBE_FAST clusters).
        let mut top = [(i32::MAX, 0u8); K];
        let mut top_len = 0usize;
        let mut worst_dist = i32::MAX;
        let mut worst_pos = 0usize;

        self.scan_clusters(query, data, &probed[..nprobe_fast],
            &mut top, &mut top_len, &mut worst_dist, &mut worst_pos);

        // Count fraud in fast result.
        let fraud_fast = top[..top_len].iter().filter(|&&(_, l)| l == 1).count();

        // Adaptive: only run full probe if result is on the approval boundary.
        if top_len == K && (fraud_fast == 2 || fraud_fast == 3) {
            self.scan_clusters(query, data, &probed[nprobe_fast..nprobe_full],
                &mut top, &mut top_len, &mut worst_dist, &mut worst_pos);
        }

        let mut result = [Label::Legit; K];
        for (slot, &(_, label_byte)) in result.iter_mut().zip(top[..top_len].iter()) {
            if label_byte == 1 {
                *slot = Label::Fraud;
            }
        }
        result
    }

    fn scan_clusters(
        &self,
        query: &[i8; DIMS],
        data: &[u8],
        clusters: &[usize],
        top: &mut [(i32, u8); K],
        top_len: &mut usize,
        worst_dist: &mut i32,
        worst_pos: &mut usize,
    ) {
        for &ci in clusters {
            let start = self.offsets[ci] as usize;
            let end = self.offsets[ci + 1] as usize;

            for j in start..end {
                let base = j * (DIMS + 1);
                let dist = dist_scalar(query, &data[base..base + DIMS]);

                if *top_len < K {
                    top[*top_len] = (dist, data[base + DIMS]);
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
                    top[*worst_pos] = (dist, data[base + DIMS]);
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

thread_local! {
    static CENTROID_BUF: RefCell<Vec<(u32, usize)>> = RefCell::new(Vec::with_capacity(2048));
}

#[inline(always)]
fn dist_scalar(query: &[i8; DIMS], record: &[u8]) -> i32 {
    // Dims 0–4 and 7–13: branch-free squared diff — auto-vectorized by AVX2/NEON.
    let mut dist: i32 = 0;
    for d in 0..5 {
        let diff = query[d] as i32 - (record[d] as i8) as i32;
        dist += diff * diff;
    }
    for d in 7..DIMS {
        let diff = query[d] as i32 - (record[d] as i8) as i32;
        dist += diff * diff;
    }
    // Dims 5–6: sentinel-aware (handled separately to keep the loops above branch-free).
    for d in 5..7 {
        let q = query[d];
        let r = record[d] as i8;
        dist += match (q == SENTINEL, r == SENTINEL) {
            (true, true) => 0,
            (true, false) | (false, true) => SENTINEL_PENALTY,
            (false, false) => {
                let diff = q as i32 - r as i32;
                diff * diff
            }
        };
    }
    dist
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
