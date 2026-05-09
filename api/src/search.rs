use crate::types::Label;
use memmap2::Mmap;
use std::collections::BinaryHeap;
use std::fs::File;

const MAGIC: &[u8; 8] = b"RINHA026";
const HEADER_SIZE: usize = 16;
const DIMS: usize = 14;
const K: usize = 5;
const SENTINEL: i8 = -127;
/// Max squared distance for one i8 dimension (254² = 64516).
/// Applied when exactly one of query/reference carries the sentinel on dim 5 or 6.
const SENTINEL_PENALTY: i32 = 64516;

pub struct SearchIndex {
    mmap: Mmap,
    count: usize,
}

impl SearchIndex {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }?;

        if &mmap[0..8] != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid magic bytes",
            ));
        }
        let count = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
        let dims = u32::from_le_bytes(mmap[12..16].try_into().unwrap());
        if dims as usize != DIMS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unexpected dims in index header",
            ));
        }

        Ok(Self { mmap, count })
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn search(&self, query: &[i8; DIMS]) -> [Label; K] {
        let data = &self.mmap[HEADER_SIZE..];
        // Each arch gets a #[target_feature]-annotated wrapper so the compiler can
        // auto-vectorize dist_scalar with the platform's best SIMD instruction set.
        // The single unsafe block per arch is the only unsafe surface.
        #[cfg(target_arch = "x86_64")]
        // SAFETY: AVX2 is guaranteed by RUSTFLAGS="-C target-cpu=haswell".
        return unsafe { search_avx2(data, self.count, query) };
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is mandatory on all AArch64 CPUs.
        return unsafe { search_neon(data, self.count, query) };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        return search_impl(data, self.count, query);
    }
}

/// Shared search implementation — compiled into each arch-specific wrapper.
fn search_impl(data: &[u8], count: usize, query: &[i8; DIMS]) -> [Label; K] {
    let mut heap: BinaryHeap<(i32, u8)> = BinaryHeap::with_capacity(K + 1);

    for i in 0..count {
        let base = i * (DIMS + 1);
        let dist = dist_scalar(query, &data[base..base + DIMS]);

        if heap.len() < K {
            heap.push((dist, data[base + DIMS]));
        } else if let Some(&(max_dist, _)) = heap.peek() {
            if dist < max_dist {
                heap.pop();
                heap.push((dist, data[base + DIMS]));
            }
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

/// Hot path for x86_64: compiled with AVX2 so the compiler auto-vectorizes dist_scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn search_avx2(data: &[u8], count: usize, query: &[i8; DIMS]) -> [Label; K] {
    search_impl(data, count, query)
}

/// Hot path for aarch64: compiled with NEON so the compiler auto-vectorizes dist_scalar.
/// NEON is mandatory on all AArch64 CPUs, so this is always safe to call.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn search_neon(data: &[u8], count: usize, query: &[i8; DIMS]) -> [Label; K] {
    search_impl(data, count, query)
}

#[inline]
fn dist_scalar(query: &[i8; DIMS], record: &[u8]) -> i32 {
    let mut dist: i32 = 0;
    for d in 0..DIMS {
        let q = query[d];
        let r = record[d] as i8;
        let contribution = if (d == 5 || d == 6) && (q == SENTINEL || r == SENTINEL) {
            if q == SENTINEL && r == SENTINEL {
                0
            } else {
                SENTINEL_PENALTY
            }
        } else {
            let diff = q as i16 - r as i16;
            (diff * diff) as i32
        };
        dist += contribution;
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

    /// Write a tiny in-memory index to a temp file and return the file.
    fn make_index(records: &[([f32; DIMS], u8)]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        let count = records.len() as u32;
        f.write_all(MAGIC).unwrap();
        f.write_all(&count.to_le_bytes()).unwrap();
        f.write_all(&(DIMS as u32).to_le_bytes()).unwrap();
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
        f.write_all(&10u32.to_le_bytes()).unwrap();
        f.write_all(&14u32.to_le_bytes()).unwrap();
        f.flush().unwrap();
        let result = SearchIndex::open(f.path().to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn header_validates_dims() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(MAGIC).unwrap();
        f.write_all(&10u32.to_le_bytes()).unwrap();
        f.write_all(&13u32.to_le_bytes()).unwrap(); // wrong dims
        f.flush().unwrap();
        let result = SearchIndex::open(f.path().to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn search_returns_correct_labels_20_vectors() {
        // 15 legit, 5 fraud; fraud vectors are far from origin, legit near origin.
        // Query at origin → nearest 5 should all be legit.
        let mut records: Vec<([f32; DIMS], u8)> = Vec::new();

        // 15 legit near origin
        for i in 0..15usize {
            let mut v = [0.0f32; DIMS];
            v[0] = 0.01 * i as f32;
            records.push((v, 0));
        }
        // 5 fraud far away
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
            records.push(([0.0f32; DIMS], 1)); // fraud at origin
        }
        for _ in 0..15 {
            records.push(([1.0f32; DIMS], 0)); // legit far away
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
        // query dim5=-127, ref dim5=-127 → contribution=0
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
        // query dim5=-127, ref dim5=0 → contribution=SENTINEL_PENALTY
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
