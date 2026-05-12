# Optimization Progress

Baseline: **4190.44** (p99=2.63ms, failure=0.23%, FP=60, FN=64)

> Nota: ambiente Mac (Docker VM) não reflete as mudanças de latência — p99 local é dominado por overhead de virtualização. As medições reais serão na submissão Linux.

---

## Fase 1 — Ganhos grátis (sem tradeoff de acurácia)

- [x] Respostas pré-computadas — elimina serde por request (6 possíveis outcomes)
- [x] `cpuset` por instância no docker-compose — fixa cores, preserva L1/L2
- [x] LUT para hour/dow no vectorizer — troca divisão float por array lookup

## Fase 2 — Tradeoff latência vs acurácia

- [x] Remover adaptive full probe — custo fixo por request
- [x] Reduzir `NPROBE_FAST` 5 → 3 — menos clusters, menos scan (acurácia mantida nos testes)
- [x] `worker_threads = 1` por instância — sem contention em 0.45 CPU

## Fase 3 — Mudanças estruturais

- [x] Stride 16 bytes (14 + 2 padding) + labels separados — registrador AVX2 alinhado, sem cache line splits
- [x] Eliminar `thread_local` Vec para centróides — array fixo na stack, sem heap/RefCell
- [x] Rebalancear nginx 0.2 → 0.1 CPU → cada API sobe para 0.45 CPU
- [ ] `MADV_RANDOM` no mmap pós-warmup — elimina prefetch inútil do kernel

---

## Resultados

| Fase | Score | p99 | Failure | FP | FN | Notas |
|------|-------|-----|---------|----|----|-------|
| Baseline | 4190.44 | 2.63ms | 0.23% | 60 | 64 | — |
| Pós F1+F2+F3 | ? | ? | ? | ? | ? | Medir na submissão |
