use crate::types::Label;
use memmap2::Mmap;
use std::fs::File;
use std::io::{Error, ErrorKind};
use std::time::Instant;

const IVF_MAGIC_V3: &[u8; 8] = b"RINHIVF3";
const IVF_MAGIC_V4: &[u8; 8] = b"RINHIVF4";
const HEADER_SIZE: usize = 20;
pub const DIMS: usize = 14;
const PADDED_DIMS: usize = 16;
const STRIDE: usize = PADDED_DIMS * 2;
pub const K: usize = 5;
const NPROBE_FAST: usize = 8;
const NPROBE_SLOW: usize = 64;
const QUANT_SCALE: f32 = 10_000.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IndexFormat {
    V3,
    V4,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct SearchMetrics {
    pub centroid_time_ns: u128,
    pub scan_time_ns: u128,
    pub scanned_vectors: usize,
    pub scanned_clusters: usize,
    pub pruned_clusters: usize,
}

impl SearchMetrics {
    pub fn total_time_ns(self) -> u128 {
        self.centroid_time_ns + self.scan_time_ns
    }
}

pub struct SearchIndex {
    mmap: Mmap,
    format: IndexFormat,
    k_clusters: usize,
    n: usize,
    centroids: Vec<[f32; DIMS]>,
    quantized_centroids: Vec<[f32; DIMS]>,
    offsets: Vec<u32>,
    radii: Vec<f32>,
    data_byte: usize,
    labels_byte: usize,
}

impl SearchIndex {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }?;

        if mmap.len() < HEADER_SIZE {
            return Err(Error::new(ErrorKind::InvalidData, "file too small"));
        }

        let format = match &mmap[0..8] {
            magic if magic == IVF_MAGIC_V3 => IndexFormat::V3,
            magic if magic == IVF_MAGIC_V4 => IndexFormat::V4,
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "invalid or unsupported magic bytes",
                ))
            }
        };

        let k_clusters = read_u32_at(&mmap, 8)? as usize;
        let n = read_u32_at(&mmap, 12)? as usize;
        let dims = read_u32_at(&mmap, 16)? as usize;
        if dims != DIMS {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "unexpected dims in index header",
            ));
        }

        let centroids_byte = HEADER_SIZE;
        let centroids_bytes = k_clusters
            .checked_mul(DIMS)
            .and_then(|x| x.checked_mul(4))
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "centroid section overflow"))?;
        let offsets_byte = centroids_byte + centroids_bytes;
        let offsets_bytes = (k_clusters + 1)
            .checked_mul(4)
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "offset section overflow"))?;
        let radii_byte = offsets_byte + offsets_bytes;
        let radii_bytes = match format {
            IndexFormat::V3 => 0,
            IndexFormat::V4 => k_clusters
                .checked_mul(4)
                .ok_or_else(|| Error::new(ErrorKind::InvalidData, "radius section overflow"))?,
        };
        let data_byte = radii_byte + radii_bytes;
        let data_bytes = n
            .checked_mul(STRIDE)
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "vector section overflow"))?;
        let labels_byte = data_byte + data_bytes;
        let labels_end = labels_byte
            .checked_add(n)
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "label section overflow"))?;

        if mmap.len() < labels_end {
            return Err(Error::new(ErrorKind::InvalidData, "truncated index file"));
        }

        let mut centroids = Vec::with_capacity(k_clusters);
        for ci in 0..k_clusters {
            let base = centroids_byte + ci * DIMS * 4;
            let mut c = [0.0f32; DIMS];
            for (d, slot) in c.iter_mut().enumerate() {
                *slot = read_f32_at(&mmap, base + d * 4)?;
            }
            centroids.push(c);
        }

        let quantized_centroids = centroids
            .iter()
            .map(|centroid| {
                let mut q = [0.0f32; DIMS];
                for d in 0..DIMS {
                    q[d] = centroid[d] * QUANT_SCALE;
                }
                q
            })
            .collect();

        let mut offsets = Vec::with_capacity(k_clusters + 1);
        for i in 0..=k_clusters {
            offsets.push(read_u32_at(&mmap, offsets_byte + i * 4)?);
        }
        if offsets.windows(2).any(|window| window[0] > window[1]) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "cluster offsets are not monotonic",
            ));
        }
        if offsets.last().copied().unwrap_or_default() as usize != n {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "cluster offsets do not match vector count",
            ));
        }

        let radii = match format {
            IndexFormat::V3 => vec![f32::INFINITY; k_clusters],
            IndexFormat::V4 => {
                let mut values = Vec::with_capacity(k_clusters);
                for ci in 0..k_clusters {
                    let radius = read_f32_at(&mmap, radii_byte + ci * 4)?;
                    if radius.is_sign_negative() {
                        return Err(Error::new(
                            ErrorKind::InvalidData,
                            "cluster radius must be non-negative",
                        ));
                    }
                    values.push(radius);
                }
                values
            }
        };

        Ok(Self {
            mmap,
            format,
            k_clusters,
            n,
            centroids,
            quantized_centroids,
            offsets,
            radii,
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
            for v in &mut q {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *v = (((state >> 16) % 20_001) as i32 - 10_000) as i16;
            }
            let _ = self.search(&q);
        }
    }

    pub fn vector_at(&self, idx: usize) -> Option<[i16; DIMS]> {
        if idx >= self.n {
            return None;
        }

        let record = load_record(&self.mmap[self.data_byte..self.labels_byte], idx);
        let mut vector = [0i16; DIMS];
        vector.copy_from_slice(&record[..DIMS]);
        Some(vector)
    }

    pub fn search(&self, query: &[i16; DIMS]) -> [Label; K] {
        self.search_with_metrics(query).0
    }

    pub fn search_with_vector(&self, query_f32: &[f32; DIMS], query: &[i16; DIMS]) -> [Label; K] {
        self.search_with_vector_and_metrics(query_f32, query).0
    }

    pub fn search_with_metrics(&self, query: &[i16; DIMS]) -> ([Label; K], SearchMetrics) {
        let mut query_f32 = [0.0f32; DIMS];
        for d in 0..DIMS {
            query_f32[d] = query[d] as f32 / QUANT_SCALE;
        }
        self.search_with_vector_and_metrics(&query_f32, query)
    }

    pub fn search_with_vector_and_metrics(
        &self,
        query_f32: &[f32; DIMS],
        query: &[i16; DIMS],
    ) -> ([Label; K], SearchMetrics) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                return unsafe { self.search_impl_avx2(query_f32, query) };
            }
        }

        self.search_impl_scalar(query_f32, query)
    }

    fn search_impl_scalar(
        &self,
        query_f32: &[f32; DIMS],
        query: &[i16; DIMS],
    ) -> ([Label; K], SearchMetrics) {
        self.search_impl_inner(
            query_f32,
            query,
            centroid_distances_top_scalar,
            centroid_distances_all_scalar,
            dist_scalar_record,
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn search_impl_avx2(
        &self,
        query_f32: &[f32; DIMS],
        query: &[i16; DIMS],
    ) -> ([Label; K], SearchMetrics) {
        self.search_impl_inner(
            query_f32,
            query,
            centroid_distances_top_avx2,
            centroid_distances_all_avx2,
            dist_avx2_record,
        )
    }

    fn search_impl_inner<TopCentroidFn, AllCentroidFn, DistFn>(
        &self,
        query_f32: &[f32; DIMS],
        query: &[i16; DIMS],
        top_centroid_fn: TopCentroidFn,
        all_centroid_fn: AllCentroidFn,
        dist_fn: DistFn,
    ) -> ([Label; K], SearchMetrics)
    where
        TopCentroidFn: Fn(&[[f32; DIMS]], &[f32; DIMS], usize) -> [(u32, usize); NPROBE_SLOW],
        AllCentroidFn: Fn(&[[f32; DIMS]], &[f32; DIMS]) -> Vec<(u32, usize)>,
        DistFn: Fn(&[i16; DIMS], &[u8], usize) -> i64,
    {
        let vectors = &self.mmap[self.data_byte..self.labels_byte];
        let labels = &self.mmap[self.labels_byte..];
        let mut metrics = SearchMetrics::default();

        if self.format == IndexFormat::V4 {
            let start = Instant::now();
            let ordered = all_centroid_fn(&self.centroids, query_f32);
            metrics.centroid_time_ns = start.elapsed().as_nanos();

            let mut top = [(i64::MAX, 0u8); K];
            let mut top_len = 0usize;
            let mut worst_dist = i64::MAX;
            let mut worst_pos = 0usize;

            let start = Instant::now();
            for &(bits, ci) in &ordered {
                let centroid_sq = f32::from_bits(bits);
                self.scan_clusters(
                    query,
                    vectors,
                    labels,
                    std::slice::from_ref(&ci),
                    std::slice::from_ref(&centroid_sq),
                    &dist_fn,
                    &mut top,
                    &mut top_len,
                    &mut worst_dist,
                    &mut worst_pos,
                    &mut metrics,
                );
            }
            metrics.scan_time_ns = start.elapsed().as_nanos();

            return (labels_to_result(&top), metrics);
        }

        let nprobe_slow = NPROBE_SLOW.min(self.k_clusters);
        let nprobe_fast = NPROBE_FAST.min(nprobe_slow);

        let start = Instant::now();
        let best = top_centroid_fn(&self.centroids, query_f32, nprobe_slow);
        metrics.centroid_time_ns = start.elapsed().as_nanos();

        let mut probed = [0usize; NPROBE_SLOW];
        let mut centroid_sq = [0.0f32; NPROBE_SLOW];
        for i in 0..nprobe_slow {
            probed[i] = best[i].1;
            centroid_sq[i] = f32::from_bits(best[i].0);
        }

        let mut top = [(i64::MAX, 0u8); K];
        let mut top_len = 0usize;
        let mut worst_dist = i64::MAX;
        let mut worst_pos = 0usize;

        let start = Instant::now();
        self.scan_clusters(
            query,
            vectors,
            labels,
            &probed[..nprobe_fast],
            &centroid_sq[..nprobe_fast],
            &dist_fn,
            &mut top,
            &mut top_len,
            &mut worst_dist,
            &mut worst_pos,
            &mut metrics,
        );

        let fraud_count = top[..top_len].iter().filter(|&&(_, label)| label == 1).count();
        if !(top_len >= K && (fraud_count == 0 || fraud_count == 1 || fraud_count == 5)) {
            self.scan_clusters(
                query,
                vectors,
                labels,
                &probed[nprobe_fast..nprobe_slow],
                &centroid_sq[nprobe_fast..nprobe_slow],
                &dist_fn,
                &mut top,
                &mut top_len,
                &mut worst_dist,
                &mut worst_pos,
                &mut metrics,
            );
        }
        metrics.scan_time_ns = start.elapsed().as_nanos();

        (labels_to_result(&top), metrics)
    }

    fn scan_clusters<DistFn>(
        &self,
        query: &[i16; DIMS],
        vectors: &[u8],
        labels: &[u8],
        clusters: &[usize],
        centroid_sq: &[f32],
        dist_fn: &DistFn,
        top: &mut [(i64, u8); K],
        top_len: &mut usize,
        worst_dist: &mut i64,
        worst_pos: &mut usize,
        metrics: &mut SearchMetrics,
    ) where
        DistFn: Fn(&[i16; DIMS], &[u8], usize) -> i64,
    {
        for (&ci, &centroid_distance_sq) in clusters.iter().zip(centroid_sq.iter()) {
            if *top_len >= K && self.can_prune_cluster(query, ci, centroid_distance_sq, *worst_dist) {
                metrics.pruned_clusters += 1;
                continue;
            }

            metrics.scanned_clusters += 1;
            let start = self.offsets[ci] as usize;
            let end = self.offsets[ci + 1] as usize;

            for j in start..end {
                let dist = dist_fn(query, vectors, j);
                metrics.scanned_vectors += 1;

                if *top_len < K {
                    top[*top_len] = (dist, labels[j]);
                    *top_len += 1;
                    if *top_len == K {
                        recompute_worst(top, worst_dist, worst_pos);
                    }
                } else if dist < *worst_dist {
                    top[*worst_pos] = (dist, labels[j]);
                    recompute_worst(top, worst_dist, worst_pos);
                }
            }
        }
    }

    fn can_prune_cluster(
        &self,
        query: &[i16; DIMS],
        ci: usize,
        centroid_distance_sq: f32,
        worst_dist: i64,
    ) -> bool {
        if self.format != IndexFormat::V4 || !worst_dist.is_positive() {
            return false;
        }

        let radius = self.radii[ci];
        if !radius.is_finite() {
            return false;
        }

        let mut exact_centroid_distance_sq = 0.0f32;
        let quantized_centroid = &self.quantized_centroids[ci];
        for d in 0..DIMS {
            let diff = query[d] as f32 - quantized_centroid[d];
            exact_centroid_distance_sq += diff * diff;
        }

        let lower_bound = (exact_centroid_distance_sq.sqrt() - radius).max(0.0);
        let lower_bound_sq = lower_bound * lower_bound;
        lower_bound_sq > worst_dist as f32 && centroid_distance_sq.is_finite()
    }
}

#[inline(always)]
fn recompute_worst(top: &[(i64, u8); K], worst_dist: &mut i64, worst_pos: &mut usize) {
    *worst_pos = 0;
    for i in 1..K {
        if top[i].0 > top[*worst_pos].0 {
            *worst_pos = i;
        }
    }
    *worst_dist = top[*worst_pos].0;
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
fn read_u32_at(bytes: &[u8], offset: usize) -> std::io::Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "truncated index file"))?;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

#[inline(always)]
fn read_f32_at(bytes: &[u8], offset: usize) -> std::io::Result<f32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "truncated index file"))?;
    Ok(f32::from_le_bytes(slice.try_into().unwrap()))
}

#[inline(always)]
fn load_record(records: &[u8], idx: usize) -> [i16; PADDED_DIMS] {
    let base = idx * STRIDE;
    unsafe {
        records[base..base + STRIDE]
            .as_ptr()
            .cast::<[i16; PADDED_DIMS]>()
            .read_unaligned()
    }
}

#[inline(always)]
fn centroid_distances_top_scalar(
    centroids: &[[f32; DIMS]],
    query_f32: &[f32; DIMS],
    nprobe_slow: usize,
) -> [(u32, usize); NPROBE_SLOW] {
    let mut best = [(u32::MAX, 0usize); NPROBE_SLOW];

    for (ci, centroid) in centroids.iter().enumerate() {
        let mut d = 0.0f32;
        for i in 0..DIMS {
            let diff = query_f32[i] - centroid[i];
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

    best
}

#[inline(always)]
fn centroid_distances_all_scalar(
    centroids: &[[f32; DIMS]],
    query_f32: &[f32; DIMS],
) -> Vec<(u32, usize)> {
    let mut ordered = Vec::with_capacity(centroids.len());
    for (ci, centroid) in centroids.iter().enumerate() {
        let mut d = 0.0f32;
        for i in 0..DIMS {
            let diff = query_f32[i] - centroid[i];
            d += diff * diff;
        }
        ordered.push((d.to_bits(), ci));
    }
    ordered.sort_unstable_by_key(|&(bits, _)| bits);
    ordered
}

#[inline(always)]
fn dist_scalar_record(query: &[i16; DIMS], records: &[u8], idx: usize) -> i64 {
    let record = load_record(records, idx);
    let mut dist: i64 = 0;
    for d in 0..DIMS {
        let diff = query[d] as i64 - record[d] as i64;
        dist += diff * diff;
    }
    dist
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn centroid_distances_top_avx2(
    centroids: &[[f32; DIMS]],
    query_f32: &[f32; DIMS],
    nprobe_slow: usize,
) -> [(u32, usize); NPROBE_SLOW] {
    use std::arch::x86_64::*;

    let mut best = [(u32::MAX, 0usize); NPROBE_SLOW];
    let q0 = _mm256_loadu_ps(query_f32.as_ptr());
    let q1 = _mm_loadu_ps(query_f32[8..].as_ptr());

    for (ci, centroid) in centroids.iter().enumerate() {
        let c0 = _mm256_loadu_ps(centroid.as_ptr());
        let c1 = _mm_loadu_ps(centroid[8..].as_ptr());

        let diff0 = _mm256_sub_ps(q0, c0);
        let diff1 = _mm_sub_ps(q1, c1);
        let sq0 = _mm256_mul_ps(diff0, diff0);
        let sq1 = _mm_mul_ps(diff1, diff1);

        let mut sum0 = [0.0f32; 8];
        let mut sum1 = [0.0f32; 4];
        _mm256_storeu_ps(sum0.as_mut_ptr(), sq0);
        _mm_storeu_ps(sum1.as_mut_ptr(), sq1);

        let d = sum0.iter().sum::<f32>() + sum1.iter().sum::<f32>();
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

    best
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn centroid_distances_all_avx2(
    centroids: &[[f32; DIMS]],
    query_f32: &[f32; DIMS],
) -> Vec<(u32, usize)> {
    use std::arch::x86_64::*;

    let mut ordered = Vec::with_capacity(centroids.len());
    let q0 = _mm256_loadu_ps(query_f32.as_ptr());
    let q1 = _mm_loadu_ps(query_f32[8..].as_ptr());

    for (ci, centroid) in centroids.iter().enumerate() {
        let c0 = _mm256_loadu_ps(centroid.as_ptr());
        let c1 = _mm_loadu_ps(centroid[8..].as_ptr());

        let diff0 = _mm256_sub_ps(q0, c0);
        let diff1 = _mm_sub_ps(q1, c1);
        let sq0 = _mm256_mul_ps(diff0, diff0);
        let sq1 = _mm_mul_ps(diff1, diff1);

        let mut sum0 = [0.0f32; 8];
        let mut sum1 = [0.0f32; 4];
        _mm256_storeu_ps(sum0.as_mut_ptr(), sq0);
        _mm_storeu_ps(sum1.as_mut_ptr(), sq1);

        ordered.push(((sum0.iter().sum::<f32>() + sum1.iter().sum::<f32>()).to_bits(), ci));
    }

    ordered.sort_unstable_by_key(|&(bits, _)| bits);
    ordered
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dist_avx2_record(query: &[i16; DIMS], records: &[u8], idx: usize) -> i64 {
    use std::arch::x86_64::*;

    let record = load_record(records, idx);
    let mut padded_query = [0i16; PADDED_DIMS];
    padded_query[..DIMS].copy_from_slice(query);

    let q = _mm256_loadu_si256(padded_query.as_ptr() as *const __m256i);
    let r = _mm256_loadu_si256(record.as_ptr() as *const __m256i);
    let q_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q));
    let q_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q, 1));
    let r_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r));
    let r_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r, 1));

    let d_lo = _mm256_sub_epi32(q_lo, r_lo);
    let d_hi = _mm256_sub_epi32(q_hi, r_hi);
    let sq_lo = _mm256_mullo_epi32(d_lo, d_lo);
    let sq_hi = _mm256_mullo_epi32(d_hi, d_hi);

    let mut acc_lo = [0i32; 8];
    let mut acc_hi = [0i32; 8];
    _mm256_storeu_si256(acc_lo.as_mut_ptr() as *mut __m256i, sq_lo);
    _mm256_storeu_si256(acc_hi.as_mut_ptr() as *mut __m256i, sq_hi);

    acc_lo.iter().map(|&x| x as i64).sum::<i64>() + acc_hi.iter().map(|&x| x as i64).sum::<i64>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn quantize(v: f32) -> i16 {
        (v * QUANT_SCALE)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16
    }

    fn write_index(records: &[([f32; DIMS], u8)], format: IndexFormat) -> NamedTempFile {
        let n = records.len();
        let k = 2usize.min(n.max(1));
        let mut f = NamedTempFile::new().unwrap();
        let magic = match format {
            IndexFormat::V3 => IVF_MAGIC_V3,
            IndexFormat::V4 => IVF_MAGIC_V4,
        };

        f.write_all(magic).unwrap();
        f.write_all(&(k as u32).to_le_bytes()).unwrap();
        f.write_all(&(n as u32).to_le_bytes()).unwrap();
        f.write_all(&(DIMS as u32).to_le_bytes()).unwrap();

        let centroid_a = records.first().map(|record| record.0).unwrap_or([0.0; DIMS]);
        let centroid_b = records.last().map(|record| record.0).unwrap_or([0.0; DIMS]);
        for centroid in [centroid_a, centroid_b].iter().take(k) {
            for &v in centroid {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }

        let split = n / k.max(1);
        let offsets = [0u32, split as u32, n as u32];
        for &offset in offsets.iter().take(k + 1) {
            f.write_all(&offset.to_le_bytes()).unwrap();
        }

        if format == IndexFormat::V4 {
            for ci in 0..k {
                let start = offsets[ci] as usize;
                let end = offsets[ci + 1] as usize;
                let centroid = if ci == 0 { centroid_a } else { centroid_b };
                let mut radius = 0.0f32;
                for (vector, _) in &records[start..end] {
                    let mut dist = 0.0f32;
                    for d in 0..DIMS {
                        let diff = quantize(vector[d]) as f32 - centroid[d] * QUANT_SCALE;
                        dist += diff * diff;
                    }
                    radius = radius.max(dist.sqrt());
                }
                f.write_all(&radius.to_le_bytes()).unwrap();
            }
        }

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

    fn brute_force(records: &[([f32; DIMS], u8)], query: &[i16; DIMS]) -> [Label; K] {
        let mut top = [(i64::MAX, 0u8); K];
        let mut len = 0usize;
        let mut worst = i64::MAX;
        let mut worst_pos = 0usize;

        for (vector, label) in records {
            let mut dist = 0i64;
            for d in 0..DIMS {
                let diff = query[d] as i64 - quantize(vector[d]) as i64;
                dist += diff * diff;
            }

            if len < K {
                top[len] = (dist, *label);
                len += 1;
                if len == K {
                    recompute_worst(&top, &mut worst, &mut worst_pos);
                }
            } else if dist < worst {
                top[worst_pos] = (dist, *label);
                recompute_worst(&top, &mut worst, &mut worst_pos);
            }
        }

        labels_to_result(&top)
    }

    fn fraud_count(labels: [Label; K]) -> usize {
        labels.into_iter().filter(|label| *label == Label::Fraud).count()
    }

    #[test]
    fn header_validates_magic() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(b"BADMAGIC").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&(DIMS as u32).to_le_bytes()).unwrap();
        f.flush().unwrap();
        assert!(SearchIndex::open(f.path().to_str().unwrap()).is_err());
    }

    #[test]
    fn header_validates_dims() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(IVF_MAGIC_V4).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&13u32.to_le_bytes()).unwrap();
        f.flush().unwrap();
        assert!(SearchIndex::open(f.path().to_str().unwrap()).is_err());
    }

    #[test]
    fn header_validates_truncation() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(IVF_MAGIC_V4).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&(DIMS as u32).to_le_bytes()).unwrap();
        f.write_all(&[0u8; 8]).unwrap();
        f.flush().unwrap();
        assert!(SearchIndex::open(f.path().to_str().unwrap()).is_err());
    }

    #[test]
    fn loads_v4_index_successfully() {
        let records = vec![([0.0f32; DIMS], 0), ([1.0f32; DIMS], 1)];
        let f = write_index(&records, IndexFormat::V4);
        let idx = SearchIndex::open(f.path().to_str().unwrap()).unwrap();
        assert_eq!(idx.count(), 2);
    }

    #[test]
    fn search_matches_bruteforce_for_v4() {
        let mut records = Vec::new();
        for i in 0..18usize {
            let mut v = [0.0f32; DIMS];
            v[0] = i as f32 * 0.01;
            v[1] = (18 - i) as f32 * 0.005;
            records.push((v, if i % 4 == 0 { 1 } else { 0 }));
        }

        let f = write_index(&records, IndexFormat::V4);
        let idx = SearchIndex::open(f.path().to_str().unwrap()).unwrap();

        for q in [0.0f32, 0.03, 0.09, 0.14] {
            let mut query = [0i16; DIMS];
            query[0] = quantize(q);
            query[1] = quantize(0.04);
            assert_eq!(
                fraud_count(idx.search(&query)),
                fraud_count(brute_force(&records, &query))
            );
        }
    }

    #[test]
    fn pruning_keeps_exact_neighbors() {
        let records = vec![
            ([0.0f32; DIMS], 0),
            ([0.001f32; DIMS], 0),
            ([0.002f32; DIMS], 0),
            ([0.003f32; DIMS], 1),
            ([0.004f32; DIMS], 0),
            ([1.0f32; DIMS], 1),
            ([1.001f32; DIMS], 1),
            ([1.002f32; DIMS], 1),
            ([1.003f32; DIMS], 1),
            ([1.004f32; DIMS], 1),
        ];

        let f = write_index(&records, IndexFormat::V4);
        let idx = SearchIndex::open(f.path().to_str().unwrap()).unwrap();
        let query = [quantize(0.0); DIMS];

        let (neighbors, metrics) = idx.search_with_metrics(&query);
        assert_eq!(
            fraud_count(neighbors),
            fraud_count(brute_force(&records, &query))
        );
        assert!(metrics.pruned_clusters <= 1);
    }

    #[test]
    fn vector_at_reads_quantized_values() {
        let mut vector = [0.0f32; DIMS];
        vector[0] = 0.1234;
        let f = write_index(&[(vector, 0)], IndexFormat::V4);
        let idx = SearchIndex::open(f.path().to_str().unwrap()).unwrap();
        let stored = idx.vector_at(0).unwrap();
        assert_eq!(stored[0], quantize(0.1234));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        let mut records = Vec::new();
        for i in 0..32usize {
            let mut v = [0.0f32; DIMS];
            v[0] = i as f32 * 0.0025;
            v[3] = (i % 7) as f32 * 0.01;
            records.push((v, (i % 2) as u8));
        }

        let f = write_index(&records, IndexFormat::V4);
        let idx = SearchIndex::open(f.path().to_str().unwrap()).unwrap();
        let mut query = [0i16; DIMS];
        query[0] = quantize(0.0175);
        query[3] = quantize(0.03);
        let mut query_f32 = [0.0f32; DIMS];
        for d in 0..DIMS {
            query_f32[d] = query[d] as f32 / QUANT_SCALE;
        }

        let scalar = idx.search_impl_scalar(&query_f32, &query).0;
        let simd = unsafe { idx.search_impl_avx2(&query_f32, &query).0 };
        assert_eq!(scalar, simd);
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

        let f = write_index(&records, IndexFormat::V4);
        let idx = SearchIndex::open(f.path().to_str().unwrap()).unwrap();
        assert_eq!(idx.count(), 20);

        let query = [quantize(0.0); DIMS];
        let neighbors = idx.search(&query);
        let fraud_count = neighbors.iter().filter(|&&l| l == Label::Fraud).count();
        assert_eq!(fraud_count, 0);
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

        let f = write_index(&records, IndexFormat::V4);
        let idx = SearchIndex::open(f.path().to_str().unwrap()).unwrap();

        let query = [quantize(0.0); DIMS];
        let neighbors = idx.search(&query);
        let fraud_count = neighbors.iter().filter(|&&l| l == Label::Fraud).count();
        assert_eq!(fraud_count, 5);
    }
}
