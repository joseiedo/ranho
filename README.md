# Rinha de Backend 2026 — Fraud Detection

Detecção de fraude via KNN nos 3 milhões de vetores de referência.

## Estratégia

**Vetorização:** 14 dimensões definidas pela spec, quantizadas em `i16` (escala ×10.000). Sentinela `-1.0` para campos ausentes mapeia naturalmente para `-10.000`.

**Índice:** IVF (Inverted File Index) pré-construído pelo `preprocessor` com K-means++. Vetores armazenados em arquivo binário mapeado em memória (`mmap`).

**Busca em duas fases:**
- Fase 1 — proba 8 clusters mais próximos. Se o resultado for inequívoco (0, 1 ou 5 vizinhos fraud com K=5 completos), retorna imediatamente.
- Fase 2 — proba até 64 clusters para casos ambíguos (2, 3 ou 4 fraud).

**Scoring:** `fraud_score = fraud_count / 5`. Aprovado se `< 0.6`.

**Respostas pré-computadas:** só existem 6 saídas possíveis, então os JSONs são `&[u8]` estáticos — zero serialização por request.

**Infra:** Axum + Unix socket, 1 thread Tokio, mimalloc. Budget: 1 CPU / 350 MB total (nginx + 2 instâncias).
