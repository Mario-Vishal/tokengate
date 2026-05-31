# ContextPilot V2 — Neural Context Optimization (design)

Last updated: **2026-05-31** · Status: **approved direction; task breakdown pending user review (no code yet)**

This document is the design reference for ContextPilot's mature form: a **neural**
context optimization engine. It supersedes the "pure, zero-dependency core" stance of
V1 (see ADR-010). V1's deterministic pieces are retained only as *signals/components*
inside the neural pipeline, not as the engine and not as a runtime fallback.

---

## One-sentence goal

> ContextPilot uses **neural retrieval signals and cross-encoder reranking** to
> understand which context matters, then uses **non-generative extractive compression**,
> **MMR diversity**, and **token-aware budgeting** to minimize what the final LLM sees —
> with a full audit trail explaining every decision.

Hard rules:
- **Embeddings + reranker are required, first-class core components.** Configurable
  (swap models), never optional. The engine does not run without them.
- **A generative LLM is used exactly once, downstream, for the final answer.** There is
  **no generative/LLM compression** in the optimization path — compression is
  extractive only (keeps original sentences, never rewrites).
- ContextPilot does **not** retrieve from LanceDB or own the index — the app does that
  and passes candidate blocks (with their stored vectors) in.

## Approved decisions (2026-05-31)

| # | Decision |
|---|----------|
| Direction | Neural-first, required; real ML deps; no product fallback. (ADR-010) |
| Embeddings | **Reuse** caller-provided block vectors when present; the required model computes query embeddings, sentence embeddings (for compression), and reranking. (ADR-011) |
| Python | Pin the library to **Python 3.12** for ML wheel support. (ADR-012) |
| Default models | Embeddings **BAAI/bge-m3** (dense, dim 1024); reranker **BAAI/bge-reranker-v2-m3** (cross-encoder). Configurable. (ADR-013) |

## The pipeline

```
query + candidate blocks (each may carry a precomputed BGE-M3 vector)
   ↓
1. Normalize candidate blocks
2. Semantic scoring        (cosine: query embedding vs block vectors)
3. Keyword scoring         (lexical overlap — a supporting signal)
4. Hybrid ranking          (multi-signal weighted score)
5. Neural reranking        (bge-reranker-v2-m3 cross-encoder, query+chunk)
6. Semantic deduplication  (embedding cosine clustering, keep best representative)
7. Extractive compression  (sentence embeddings + keyword/entity/heading scoring)
8. MMR diversity selection (relevance − redundancy)
9. Token-aware budgeting   (value-per-token "best set under budget")
10. Prompt assembly
11. Audit report           (models_used, rerank scores, per-signal reasons)
   ↓
optimized prompt → (Beacon) → single local LLM call for the answer
```

## Engine architecture (neural-first, required)

ContextPilot ships the neural capability as **required core**, structured as:

1. **Model interfaces (protocols)** — `EmbeddingModel`, `Reranker`. These exist for
   *configurability and testing*, not optionality: the engine always has a real model
   behind them. Unit tests may substitute deterministic fakes for speed (a test
   concern, never a product mode).

   ```python
   class EmbeddingModel(Protocol):
       dim: int
       def embed(self, texts: list[str]) -> list[Sequence[float]]: ...

   class Reranker(Protocol):
       def rerank(self, query: str, texts: list[str]) -> list[float]: ...
   ```

2. **Default implementations (shipped, required deps)** — `BGEM3Embedder` and
   `BGEReranker`, backed by sentence-transformers / FlagEmbedding + PyTorch. GPU-aware
   (CUDA if available, else CPU). These load on construction and are the out-of-box
   engine.

3. **Vector reuse** — when a `ContextBlock` carries a precomputed vector (from Beacon's
   LanceDB), stages 2/6/8 use it directly. Query embeddings, sentence embeddings (for
   compression), and reranking are always computed by the model.

Dependencies become core (no longer zero-dep): `torch`, `sentence-transformers` and/or
`FlagEmbedding`, `numpy`. Managed under Python 3.12 via uv.

## Stage detail

### 2–4 Semantic + keyword + hybrid ranking
`final_score` blends multiple signals (weights configurable; missing signals contribute
0 and weights renormalize):
```
final_score = w_sem·semantic + w_kw·keyword + w_rec·recency
            + w_src·source_priority + w_tok·token_efficiency
```
- `semantic` = cosine(query_vec, block_vec) (block_vec reused if provided).
- `recency` from `metadata.modified_time`; `source_priority` from a configurable source
  weight map; `token_efficiency` rewards value per token.

### 5 Neural reranking
Cross-encoder scores (query, chunk) jointly → `rerank_score`. Flow: app sends ~top-50 →
rerank → keep top ~10–15 (`rerank_top_n` configurable) into later stages. Query-aware,
non-generative (no generation token cost).

### 6 Semantic deduplication
Pairwise cosine over block vectors; if ≥ threshold, keep the best representative
(higher rerank/score, shorter, newer, better source, richer metadata) and drop the rest
with recorded reasons. Major token saver.

### 7 Extractive compression (no LLM)
Sentence/paragraph scoring:
```
sentence_score = semantic_sim(query, sentence) + keyword_overlap
               + entity_match_bonus + heading_bonus − redundancy_penalty
```
Keep highest-value original sentences until `target_tokens`. Records original/compressed
tokens, `compression_method="embedding_extractive"`, and `source_block_id`.

### 8 MMR diversity selection
`mmr = λ·relevance(query) − (1−λ)·max_sim(already_selected)` (λ configurable). Prevents
selecting many near-identical chunks; favors a diverse evidence set.

### 9 Token-aware budgeting
"Best set under budget" rather than top-k: maximize total value s.t. `Σtokens ≤ budget`
(knapsack-style; greedy-by-value/token with required-first reservation per ADR-009);
compress high-value large blocks before dropping.

### 11 Audit
Adds `models_used` (embedding_model, reranker; final_llm reported by the app),
per-block `rerank_score`, and multi-signal `reason` lists. Counts and token totals must
reconcile.

## Public API (V2)

```python
pilot = ContextPilot(
    max_prompt_tokens=4096,
    strategy="quality",
    config=OptimizerConfig(...),   # weights, thresholds, mmr_lambda, rerank_top_n
    # embedding_model / reranker default to the shipped BGE implementations;
    # injectable for tests or to swap models — but always present.
)
result = pilot.optimize(query, blocks)
```

## Testing strategy

- Unit-test each stage with deterministic **fake** `EmbeddingModel`/`Reranker` (toy
  vectors) so the suite stays fast — a *test* substitution, not a product fallback.
- Opt-in integration tests load the real BGE models (skipped if not present), mirroring
  the existing tiktoken-skip pattern.

## Out of this design (still later)
- Structured/LLM JSON compression (explicitly excluded from the main path).
- TurboVec/Qdrant backends (retrieval lives in the app).
