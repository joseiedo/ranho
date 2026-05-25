use crate::types::Label;
use memmap2::Mmap;
use std::fs::File;

const IVF_MAGIC: &[u8; 8] = b"RINHIVF4";
const HEADER_SIZE: usize = 24;
const DIMS: usize = 14;
const STRIDE: usize = 16;
const K: usize = 5;
const NPROBE: usize = 8;
const NPROBE_RETRY_EXTRA: usize = 100;

pub struct SearchIndex {
    n_clusters: usize,
    n: usize,
    centroids: Vec<f32>,
    bbox_min: Vec<i16>,
    bbox_max: Vec<i16>,
    offsets: Vec<u32>,
    dim_data: Vec<i16>,
    labels: Vec<u8>,
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

        let n = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
        let n_clusters = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let dims = u32::from_le_bytes(mmap[16..20].try_into().unwrap()) as usize;
        let stride = u32::from_le_bytes(mmap[20..24].try_into().unwrap()) as usize;

        if dims != DIMS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unexpected dims in index header",
            ));
        }
        if stride != STRIDE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unexpected stride in index header",
            ));
        }

        let centroid_bytes = n_clusters * STRIDE * std::mem::size_of::<f32>();
        let bbox_bytes = n_clusters * DIMS * std::mem::size_of::<i16>();
        let offsets_bytes = (n_clusters + 1) * std::mem::size_of::<u32>();
        let dim_data_bytes = DIMS * n * std::mem::size_of::<i16>();
        let labels_bytes = n;
        let expected_len = HEADER_SIZE
            + centroid_bytes
            + bbox_bytes
            + bbox_bytes
            + offsets_bytes
            + dim_data_bytes
            + labels_bytes;

        if mmap.len() < expected_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "truncated index file",
            ));
        }

        let mut cursor = HEADER_SIZE;
        let centroids = read_f32_slice(&mmap[cursor..cursor + centroid_bytes]);
        cursor += centroid_bytes;
        let bbox_min = read_i16_slice(&mmap[cursor..cursor + bbox_bytes]);
        cursor += bbox_bytes;
        let bbox_max = read_i16_slice(&mmap[cursor..cursor + bbox_bytes]);
        cursor += bbox_bytes;
        let offsets = read_u32_slice(&mmap[cursor..cursor + offsets_bytes]);
        cursor += offsets_bytes;
        let dim_data = read_i16_slice(&mmap[cursor..cursor + dim_data_bytes]);
        cursor += dim_data_bytes;
        let labels = mmap[cursor..cursor + labels_bytes].to_vec();

        Ok(Self {
            n_clusters,
            n,
            centroids,
            bbox_min,
            bbox_max,
            offsets,
            dim_data,
            labels,
        })
    }

    pub fn count(&self) -> usize {
        self.n
    }

    pub fn warmup(&self) {
        let mut sink: u64 = 0;
        for &offset in &self.offsets {
            sink ^= offset as u64;
        }
        for value in self.dim_data.iter().step_by(64) {
            sink ^= *value as u64;
        }
        for &label in self.labels.iter().step_by(64) {
            sink ^= label as u64;
        }

        let mut state = 0x1234_5678u32;
        for _ in 0..500 {
            let mut q = [0i16; DIMS];
            for v in &mut q {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *v = (((state >> 16) % 20_001) as i32 - 10_000) as i16;
            }
            let _ = self.search(&q);
        }

        let _ = sink;
    }

    pub fn search(&self, query: &[i16; DIMS]) -> [Label; K] {
        self.search_impl(query)
    }

    pub fn search_with_vector(&self, _query_f32: &[f32; DIMS], query: &[i16; DIMS]) -> [Label; K] {
        self.search_impl(query)
    }

    fn search_impl(&self, query: &[i16; DIMS]) -> [Label; K] {
        let nprobe = NPROBE.min(self.n_clusters);
        let max_probe = NPROBE_RETRY_EXTRA.min(self.n_clusters);

        let mut probe_dist = vec![f32::MAX; nprobe];
        let mut probe_idx = vec![0usize; nprobe];

        for c in 0..self.n_clusters {
            let dist = self.centroid_distance(query, c);
            if dist < probe_dist[nprobe - 1] {
                let mut pos = nprobe - 1;
                while pos > 0 && probe_dist[pos - 1] > dist {
                    probe_dist[pos] = probe_dist[pos - 1];
                    probe_idx[pos] = probe_idx[pos - 1];
                    pos -= 1;
                }
                probe_dist[pos] = dist;
                probe_idx[pos] = c;
            }
        }

        let mut top = [(i64::MAX, 0u8); K];
        let mut top_len = 0usize;
        let mut worst_dist = i64::MAX;
        let mut worst_pos = 0usize;

        self.scan_clusters(
            query,
            &probe_idx,
            &mut top,
            &mut top_len,
            &mut worst_dist,
            &mut worst_pos,
        );

        let fraud_count = top[..top_len]
            .iter()
            .filter(|&&(_, label)| label == 1)
            .count();
        if top_len >= K && (fraud_count == 2 || fraud_count == 3) && max_probe > nprobe {
            let mut visited = vec![false; self.n_clusters];
            for &cluster in &probe_idx {
                visited[cluster] = true;
            }

            let extra_count = max_probe - nprobe;
            let mut extra_dist = vec![f32::MAX; extra_count];
            let mut extra_idx = vec![0usize; extra_count];

            for (c, already_visited) in visited.iter().enumerate() {
                if *already_visited {
                    continue;
                }

                let dist = self.centroid_distance(query, c);
                if dist < extra_dist[extra_count - 1] {
                    let mut pos = extra_count - 1;
                    while pos > 0 && extra_dist[pos - 1] > dist {
                        extra_dist[pos] = extra_dist[pos - 1];
                        extra_idx[pos] = extra_idx[pos - 1];
                        pos -= 1;
                    }
                    extra_dist[pos] = dist;
                    extra_idx[pos] = c;
                }
            }

            self.scan_clusters(
                query,
                &extra_idx,
                &mut top,
                &mut top_len,
                &mut worst_dist,
                &mut worst_pos,
            );
        }

        labels_to_result(&top)
    }

    fn centroid_distance(&self, query: &[i16; DIMS], cluster: usize) -> f32 {
        let base = cluster * STRIDE;
        let mut dist = 0.0f32;
        for d in 0..DIMS {
            let diff = query[d] as f32 - self.centroids[base + d];
            dist += diff * diff;
        }
        dist
    }

    fn scan_clusters(
        &self,
        query: &[i16; DIMS],
        clusters: &[usize],
        top: &mut [(i64, u8); K],
        top_len: &mut usize,
        worst_dist: &mut i64,
        worst_pos: &mut usize,
    ) {
        for &cluster in clusters {
            if *top_len >= K && self.bbox_lower_bound(query, cluster) > *worst_dist {
                continue;
            }

            let start = self.offsets[cluster] as usize;
            let end = self.offsets[cluster + 1] as usize;

            for i in start..end {
                let dist = self.vector_distance(query, i, *worst_dist);

                if *top_len < K {
                    top[*top_len] = (dist, self.labels[i]);
                    *top_len += 1;
                    if *top_len == K {
                        *worst_pos = index_of_worst(top);
                        *worst_dist = top[*worst_pos].0;
                    }
                } else if dist < *worst_dist {
                    top[*worst_pos] = (dist, self.labels[i]);
                    *worst_pos = index_of_worst(top);
                    *worst_dist = top[*worst_pos].0;
                }
            }
        }
    }

    #[inline(always)]
    fn bbox_lower_bound(&self, query: &[i16; DIMS], cluster: usize) -> i64 {
        let base = cluster * DIMS;
        let mut lb = 0i64;
        for d in 0..DIMS {
            let q = query[d] as i32;
            let lo = self.bbox_min[base + d] as i32;
            let hi = self.bbox_max[base + d] as i32;
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
    fn vector_distance(&self, query: &[i16; DIMS], idx: usize, stop_at: i64) -> i64 {
        let mut dist = 0i64;
        for d in 0..DIMS {
            let rv = self.dim_data[d * self.n + idx] as i64;
            let diff = query[d] as i64 - rv;
            dist += diff * diff;
            if dist > stop_at {
                break;
            }
        }
        dist
    }
}

#[inline(always)]
fn index_of_worst(top: &[(i64, u8); K]) -> usize {
    let mut worst = 0usize;
    for i in 1..K {
        if top[i].0 > top[worst].0 {
            worst = i;
        }
    }
    worst
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

fn read_f32_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn read_i16_slice(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn read_u32_slice(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
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
        let clusters = 1usize;
        let mut f = NamedTempFile::new().unwrap();

        f.write_all(IVF_MAGIC).unwrap();
        f.write_all(&(n as u32).to_le_bytes()).unwrap();
        f.write_all(&(clusters as u32).to_le_bytes()).unwrap();
        f.write_all(&(DIMS as u32).to_le_bytes()).unwrap();
        f.write_all(&(STRIDE as u32).to_le_bytes()).unwrap();

        let mut centroid = [0.0f32; STRIDE];
        if n > 0 {
            for (vec, _) in records {
                for d in 0..DIMS {
                    centroid[d] += quantize(vec[d]) as f32;
                }
            }
            for value in centroid.iter_mut().take(DIMS) {
                *value /= n as f32;
            }
        }
        for &v in &centroid {
            f.write_all(&v.to_le_bytes()).unwrap();
        }

        let mut bbox_min = [0i16; DIMS];
        let mut bbox_max = [0i16; DIMS];
        if n > 0 {
            bbox_min.fill(i16::MAX);
            bbox_max.fill(i16::MIN);
            for (vec, _) in records {
                for d in 0..DIMS {
                    let q = quantize(vec[d]);
                    bbox_min[d] = bbox_min[d].min(q);
                    bbox_max[d] = bbox_max[d].max(q);
                }
            }
        }
        for &v in &bbox_min {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        for &v in &bbox_max {
            f.write_all(&v.to_le_bytes()).unwrap();
        }

        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&(n as u32).to_le_bytes()).unwrap();

        for d in 0..DIMS {
            for (vec, _) in records {
                f.write_all(&quantize(vec[d]).to_le_bytes()).unwrap();
            }
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
        f.write_all(&(DIMS as u32).to_le_bytes()).unwrap();
        f.write_all(&(STRIDE as u32).to_le_bytes()).unwrap();
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
        f.write_all(&(STRIDE as u32).to_le_bytes()).unwrap();
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
}
