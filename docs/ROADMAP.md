# ContextPilot — ROADMAP

Last updated: **2026-05-30**

Phases are sequential. A phase is "done" only when its code **and** tests pass and
`PROJECT_MEMORY.md` is updated.

> **STATUS (2026-05-30): Phases 0–8 COMPLETE — library V1 (v0.1.0) shipped.**
> 114 tests passing, mypy strict + ruff clean. V2 (below) not started.

## Phase 0 — Planning ✅ (in progress → done at end of this session)
- Repo init, license, gitignore, README.
- All planning docs created.
- Sequential TODO list created.

## Phase 1 — Package skeleton
- `pyproject.toml` (PEP 621, uv-managed), `src/contextpilot/` tree.
- `utils/errors.py`, `utils/logging.py`, `utils/hashing.py`.
- Empty/typed module stubs so imports resolve.
- `uv sync` works; `import contextpilot` works; empty pytest run is green.

## Phase 2 — Core models
- `core/block.py` — `ContextBlock` with validation + id/hash derivation.
- `core/result.py` — `OptimizationResult`, `AuditReport`, `BlockDecision`.
- `core/config.py` — `OptimizerConfig` + `balanced` preset.
- Tests: block validation, model serialization.

## Phase 3 — Token counting & budgeting
- `budgeting/token_counter.py` — protocol + heuristic default (+ optional tiktoken).
- `budgeting/budgeter.py` — greedy selection, required-first, under max tokens.
- Tests: token counts monotonic/sane; budget never exceeded; required always kept.

## Phase 4 — Deduplication & ranking
- `deduplication/exact.py`.
- `ranking/keyword_ranker.py`, `score_normalizer.py`, `hybrid_ranker.py`.
- Tests: exact dedup; keyword scoring; hybrid scoring; normalization bounds.

## Phase 5 — Compression
- `compression/extractive.py` (sentence selection by query relevance).
- Tests: compressed output ≤ original tokens; preserves top sentences; respects
  `compressible=False`.

## Phase 6 — Prompt builder & audit
- `prompts/prompt_builder.py` — stable/cacheable sections first, deterministic.
- `audit/audit_report.py` — assemble full audit from pipeline state.
- Tests: ordering rule; audit counts and token math correct.

## Phase 7 — Optimizer (end-to-end)
- `core/optimizer.py` — `ContextPilot.optimize()` wiring all stages.
- `__init__.py` public exports.
- Tests: end-to-end; dropped/compressed recorded in audit; budget respected;
  required blocks always present.

## Phase 8 — V1 hardening & docs
- `LIBRARY_API.md` finalized against real signatures.
- README usage verified runnable.
- Coverage pass; ruff/mypy clean.
- **Definition of Done (V1)** met → tag `v0.1.0`.

---

## V2 — Neural context optimization engine (approved 2026-05-31)

> **STATUS (2026-05-31): V2 COMPLETE — neural engine shipped as v0.2.0.** CP-014..CP-024
> done; 178 tests + 4 opt-in real-model tests (verified on GPU); mypy strict + ruff clean.
> Full design: [`V2_DESIGN.md`](V2_DESIGN.md). ADR-010..017. Neural is required core. No
> generative LLM in the path (extractive only).

Sequential phases (each: code + tests green + PROJECT_MEMORY updated):

- **V2-P1 Environment & deps** — pin Python 3.12; add torch + sentence-transformers/
  FlagEmbedding + numpy; uv 3.12 venv; verify installs. (CP-014)
- **V2-P2 Model layer** — `EmbeddingModel`/`Reranker` protocols; default `BGEM3Embedder`
  + `BGEReranker` (GPU-aware); fake models for tests. (CP-015)
- **V2-P3 Vectors on blocks** — `ContextBlock.vector`; reuse-or-compute; dim validation.
  (CP-016)
- **V2-P4 Semantic + hybrid scoring** — cosine semantic scoring from vectors; expand
  hybrid ranking (semantic, keyword, recency, source_priority, token_efficiency). (CP-017)
- **V2-P5 Neural reranking** — cross-encoder rerank stage; `rerank_top_n`. (CP-018)
- **V2-P6 Semantic deduplication** — embedding-cosine dedup, best-representative. (CP-019)
- **V2-P7 Embedding extractive compression** — sentence embeddings + lexical/entity/
  heading scoring to a token target. (CP-020)
- **V2-P8 MMR diversity selection** — λ relevance vs redundancy. (CP-021)
- **V2-P9 Value-per-token budgeting** — knapsack-style "best set under budget". (CP-022)
- **V2-P10 Optimizer rewire + audit** — full neural pipeline end-to-end; audit gains
  `models_used`, `rerank_score`, multi-signal reasons; strategy presets become real. (CP-023)
- **V2-P11 Hardening & benchmarks** — integration tests with real models; baseline-vs-
  neural benchmark; docs; tag `v0.2.0`. (CP-024)

### Deferred beyond V2
- BGE-M3 sparse + multi-vector (ColBERT) hybrid scoring.
- Optimization cache; cache-aware prompt sectioning.
- Qdrant/TurboVec (retrieval lives in the app, not the library).
