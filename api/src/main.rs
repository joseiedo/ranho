use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use api::search::SearchIndex;
use api::types::Label;
use api::vectorizer::Vectorizer;
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::UnixListener;
use std::{collections::HashMap, sync::Arc};

// Full HTTP/1.1 responses (headers + body) for each fraud count (0–5).
// Content-Length values:
//   true  body: {"approved":true,"fraud_score":0.X}  = 35 bytes
//   false body: {"approved":false,"fraud_score":X.X} = 36 bytes
static FRAUD_HTTP: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];

static READY_HTTP: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";

struct AppState {
    vectorizer: Vectorizer,
    index: Option<SearchIndex>,
}

// Returns the byte offset just past the final \n of the blank line (\r\n\r\n).
#[inline]
fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

// Scans header lines starting at `start` for Content-Length.
fn parse_content_length(buf: &[u8], start: usize) -> usize {
    let mut pos = start;
    while pos < buf.len() {
        let end = match buf[pos..].iter().position(|&b| b == b'\r') {
            Some(e) => pos + e,
            None => break,
        };
        let line = &buf[pos..end];
        if line.is_empty() {
            break;
        }
        if line.len() > 15 && line[..15].eq_ignore_ascii_case(b"content-length:") {
            return line[15..]
                .trim_ascii()
                .iter()
                .fold(0usize, |acc, &d| {
                    if d.is_ascii_digit() {
                        acc * 10 + (d - b'0') as usize
                    } else {
                        acc
                    }
                });
        }
        pos = end + 2; // skip \r\n
    }
    0
}

fn process_fraud(body: &[u8], state: &AppState) -> &'static [u8] {
    let Some(index) = &state.index else {
        return FRAUD_HTTP[0];
    };
    let payload = match serde_json::from_slice(body) {
        Ok(p) => p,
        Err(_) => return FRAUD_HTTP[0],
    };
    let vector = state.vectorizer.vectorize(&payload);
    let quantized = Vectorizer::quantize(&vector);
    let neighbors = index.search_with_vector(&vector, &quantized);
    let fraud_count = neighbors.iter().filter(|&&l| l == Label::Fraud).count();
    FRAUD_HTTP[fraud_count]
}

async fn handle_conn(mut stream: monoio::net::UnixStream, state: Arc<AppState>) {
    // `buf` accumulates raw bytes from the socket. We shift consumed bytes out
    // after each request so the next request always starts at offset 0.
    let mut buf: Vec<u8> = Vec::with_capacity(8192);
    // Reused write buffer — avoids re-allocating per response.
    let mut wbuf: Vec<u8> = Vec::with_capacity(256);

    loop {
        // ── Phase 1: read until we have the complete header block ─────────────
        let header_end = loop {
            if let Some(end) = find_header_end(&buf) {
                break end;
            }
            if buf.len() == buf.capacity() {
                // Headers exceed buffer — malformed or oversized request.
                return;
            }
            let (res, b) = stream.read(buf).await;
            buf = b;
            match res {
                Ok(0) | Err(_) => return,
                Ok(_) => {}
            }
        };

        // ── Phase 2: identify route and parse Content-Length ─────────────────
        let first_crlf = match buf[..header_end].iter().position(|&b| b == b'\r') {
            Some(p) => p,
            None => return,
        };
        let first_line = &buf[..first_crlf];
        let is_fraud = first_line.starts_with(b"POST /fraud-score");
        let is_ready = !is_fraud && first_line.starts_with(b"GET /ready");

        let content_length = if is_fraud {
            parse_content_length(&buf, first_crlf + 2)
        } else {
            0
        };

        // ── Phase 3: read body if needed ──────────────────────────────────────
        let body_end = header_end + content_length;
        while buf.len() < body_end {
            let needed = body_end.saturating_sub(buf.capacity());
            if needed > 0 {
                buf.reserve(needed);
            }
            let (res, b) = stream.read(buf).await;
            buf = b;
            match res {
                Ok(0) | Err(_) => return,
                Ok(_) => {}
            }
        }

        // ── Phase 4: handle request ───────────────────────────────────────────
        let response: &[u8] = if is_ready {
            READY_HTTP
        } else if is_fraud {
            process_fraud(&buf[header_end..body_end], &state)
        } else {
            FRAUD_HTTP[0]
        };

        // ── Phase 5: write response ───────────────────────────────────────────
        wbuf.clear();
        wbuf.extend_from_slice(response);
        let (res, wb) = stream.write_all(wbuf).await;
        wbuf = wb;
        if res.is_err() {
            return;
        }

        // ── Phase 6: slide leftover pipelined bytes to front ─────────────────
        let remaining = buf.len() - body_end;
        if remaining > 0 {
            buf.copy_within(body_end.., 0);
        }
        // SAFETY: we just moved `remaining` valid bytes to [0..remaining].
        unsafe { buf.set_len(remaining) };
    }
}

fn main() {
    let mcc_risk: HashMap<String, f32> = [
        ("5411", 0.15),
        ("5812", 0.30),
        ("5912", 0.20),
        ("5944", 0.45),
        ("7801", 0.80),
        ("7802", 0.75),
        ("7995", 0.85),
        ("4511", 0.35),
        ("5311", 0.25),
        ("5999", 0.50),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();

    let index_path =
        std::env::var("INDEX_PATH").unwrap_or_else(|_| "./resources/index.bin".to_string());

    let index = SearchIndex::open(&index_path).ok();
    if let Some(ref idx) = index {
        idx.warmup();
    }

    let state = Arc::new(AppState {
        vectorizer: Vectorizer::new(mcc_risk),
        index,
    });

    let socket_path =
        std::env::var("SOCKET_PATH").unwrap_or_else(|_| "/tmp/api.sock".to_string());
    let _ = std::fs::remove_file(&socket_path);

    #[cfg(target_os = "linux")]
    let mut rt = monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
        .entries(1024)
        .build()
        .expect("io_uring runtime failed — kernel >= 5.9 required");

    #[cfg(not(target_os = "linux"))]
    let mut rt = monoio::RuntimeBuilder::<monoio::LegacyDriver>::new()
        .build()
        .expect("legacy runtime failed");

    rt.block_on(async move {
        use std::os::unix::fs::PermissionsExt;
        let listener = UnixListener::bind(&socket_path).expect("bind failed");
        std::fs::set_permissions(
            &socket_path,
            std::fs::Permissions::from_mode(0o777),
        )
        .unwrap();

        loop {
            let Ok((stream, _)) = listener.accept().await else {
                continue;
            };
            monoio::spawn(handle_conn(stream, state.clone()));
        }
    });
}
