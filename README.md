# Rinha de Backend 2026 — Fraud Detection

Detecção de fraude via KNN nos 3 milhões de vetores de referência.

## Estratégia

**Vetorização:** 14 dimensões definidas pela spec, quantizadas em `i16` (escala ×10.000). Sentinela `-1.0` para campos ausentes mapeia naturalmente para `-10.000`.

**Índice:** IVF (Inverted File Index) pré-construído pelo `preprocessor`. Os centroides são treinados em espaço quantizado `i16`, depois o índice é gravado com centroides `float32`, offsets cumulativos, bounding boxes por cluster e vetores clusterizados em layout column-major.

**Busca IVF:**
- Proba os 8 clusters mais próximos pelo centróide.
- Faz pruning por bounding box antes de varrer o cluster.
- Escaneia os vetores quantizados em layout column-major.
- Se o top-5 parcial ficar ambíguo (2 ou 3 fraudes), expande a busca para até 100 clusters.

**Scoring:** `fraud_score = fraud_count / 5`. Aprovado se `< 0.6`.

**Respostas pré-computadas:** só existem 6 saídas possíveis, então os JSONs são `&[u8]` estáticos — zero serialização por request.

**Infra:** Axum + Unix socket, 1 thread Tokio, mimalloc. Budget: 1 CPU / 350 MB total (nginx + 2 instâncias).

## Build do índice

O dataset de referência usado neste repositório fica em `./spec/resources/references.json.gz`. Para reconstruir o índice local:

```bash
cargo run --release -p preprocessor -- \
  ./spec/resources/references.json.gz \
  ./resources/index.bin
```

## Benchmark

Benchmark local do search path com o índice reconstruído:

- `search_with_vector`: `12,708.86 qps`, `78,685 ns/query`
- `search`: `12,732.46 qps`, `78,539 ns/query`
