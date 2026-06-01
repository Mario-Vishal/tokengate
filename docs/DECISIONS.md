# ContextPilot — DECISIONS (ADR log)

Architecture Decision Records. Append-only; supersede rather than delete.

Format: ID · Date · Status · Context · Decision · Consequences.

---

### ADR-001 — `src/` layout, importable as `contextpilot`
- **Date:** 2026-05-30 · **Status:** Accepted
- **Context:** Need a clean, testable package that can't be accidentally imported
  from the repo root before install.
- **Decision:** Use the `src/` layout. Package is `src/contextpilot/`, imported as
  `import contextpilot`.
- **Consequences:** Tests run against the installed (editable) package, catching
  packaging mistakes early. Slightly more config in `pyproject.toml`.

### ADR-002 — uv + pyproject (PEP 621) + MIT
- **Date:** 2026-05-30 · **Status:** Accepted
- **Context:** User chose uv and MIT. Need standard, modern packaging.
- **Decision:** Manage env/deps with **uv**; declare metadata in `pyproject.toml`
  via PEP 621; license **MIT**.
- **Consequences:** Fast, reproducible installs. Contributors need uv (documented
  in README). pip install still works from the same metadata.

### ADR-003 — Pluggable token counting; heuristic default
- **Date:** 2026-05-30 · **Status:** Accepted
- **Context:** Accurate token counts depend on the target model's tokenizer.
  Forcing `tiktoken` adds a heavy native dep and may lack wheels on Python 3.14.
- **Decision:** Define a `TokenCounter` protocol. Ship a dependency-free
  `HeuristicTokenCounter` as default; offer `TiktokenCounter` behind the optional
  extra `contextpilot[tiktoken]`. Applications may inject their own.
- **Consequences:** Zero required runtime deps for the core. Default counts are
  approximate; exactness is opt-in.

### ADR-004 — Pure core: no I/O, no LLM, no network
- **Date:** 2026-05-30 · **Status:** SUPERSEDED by ADR-010 (2026-05-31)
- **Context:** The library must be reusable and app-agnostic.
- **Decision:** The core performs no network calls, no user-data disk I/O, and
  never calls an LLM. It only transforms inputs into results.
- **Consequences:** Trivially unit-testable and deterministic. LLM-based features
  (e.g. structured compression) must be implemented as optional adapters in V2 that
  the *caller* drives.

### ADR-005 — Semantic scores are caller-provided
- **Date:** 2026-05-30 · **Status:** AMENDED by ADR-011 (2026-05-31)
- **Context:** Embeddings require a model and infra that belong to the application.
- **Decision:** Each `ContextBlock` carries an optional `semantic_score` supplied by
  the caller (e.g. retrieval similarity). The library computes `keyword_score` and
  combines both into `final_score`. The library computes no embeddings in V1.
- **Consequences:** Clean separation from retrieval. Works even with keyword-only
  signals when no semantic score is provided.

### ADR-006 — Every optimize() returns a complete, serializable audit
- **Date:** 2026-05-30 · **Status:** Accepted
- **Context:** Explainability is a core product value.
- **Decision:** `optimize()` always returns an `AuditReport` with per-block
  decisions, reasons, and token math; all result models are serializable.
- **Consequences:** Slightly more bookkeeping in the pipeline; big payoff for
  debugging, UI display (Beacon AuditPage), and benchmarking.

### ADR-007 — Naming: library `contextpilot`, app `Beacon`
- **Date:** 2026-05-30 · **Status:** Accepted
- **Context:** User wants the desktop app to have a standalone identity distinct
  from the library, not `contextpilot-desktop`.
- **Decision:** Library/package = `contextpilot`. Desktop app product name =
  **Beacon**, repo `beacon`, backend package `beacon`, bundle id
  `app.beacon.desktop`.
- **Consequences:** Clear brand separation. Beacon depends on `contextpilot`; never
  the reverse.

### ADR-009 — Required blocks are always included, even past the budget
- **Date:** 2026-05-30 · **Status:** Accepted
- **Context:** A caller marks a block `required=True` to guarantee it reaches the LLM
  (e.g. a system instruction). The token budget could in principle be smaller than the
  required blocks alone.
- **Decision:** The budgeter always includes every `required` block, even if their
  combined tokens exceed `max_prompt_tokens`. The overage is reported honestly in the
  audit (`final_prompt_tokens` may exceed the budget; `tokens_saved` can be small or
  negative) rather than silently dropping required content. Optional blocks are then
  fitted into whatever budget remains.
- **Consequences:** Predictable guarantee for callers; no silent loss of mandated
  context. Callers who over-mark `required` can blow the budget — surfaced via the
  audit so it's diagnosable. (A future `strict_budget` option could raise `BudgetError`
  instead; deferred.)

### ADR-008 — Single `balanced` strategy in V1, config shape reserves the rest
- **Date:** 2026-05-30 · **Status:** Accepted
- **Context:** Multiple strategy presets are a V2 feature, but the API shouldn't
  change later to accommodate them.
- **Decision:** `OptimizerConfig.strategy` accepts `"balanced"` in V1 (others
  reserved). The config carries the knobs the future presets will set.
- **Consequences:** Forward-compatible public API; presets become data, not API
  changes.

---

## V2 — Neural engine (approved 2026-05-31)

### ADR-010 — Neural-first engine; embeddings + reranker are required core
- **Date:** 2026-05-31 · **Status:** Accepted · **Supersedes:** ADR-004
- **Context:** The user's mature vision requires real neural sophistication, not
  deterministic heuristics with optional add-ons. "Nothing is optional."
- **Decision:** ContextPilot is a **neural** context-optimization engine. Embedding
  similarity and **cross-encoder reranking are required, first-class core components**
  (configurable models, never optional, no runtime fallback). The library takes real ML
  dependencies (PyTorch + sentence-transformers / FlagEmbedding + numpy). V1's
  deterministic pieces (keyword scoring, exact dedup, extractive selection) survive only
  as *signals/components* within the neural pipeline. The single generative LLM call
  remains downstream in the app; **no generative compression** in the path.
- **Consequences:** Far more capable and differentiated. Heavier install, GPU-aware,
  needs a constrained Python (ADR-012). Determinism via fixed model weights + seeds;
  unit tests use fake models (test-only, not a product mode). Reverses the
  zero-dependency stance of ADR-004.

### ADR-011 — Reuse caller-provided block vectors; model owns query/sentence/rerank
- **Date:** 2026-05-31 · **Status:** Accepted · **Amends:** ADR-005
- **Context:** Beacon's LanceDB already stores BGE-M3 vectors per chunk. Re-embedding
  every candidate per query wastes compute.
- **Decision:** When a `ContextBlock` carries a precomputed vector, semantic scoring,
  semantic dedup, and MMR reuse it. The required embedding model always computes the
  **query** embedding, **sentence-level** embeddings (for extractive compression), and
  the **reranker** runs the cross-encoder. `semantic_score` may still be supplied, but a
  model is always present (not "caller-provided or nothing" as in ADR-005).
- **Consequences:** Efficient and sophisticated. The block vector dim must match the
  model (validated; mismatch → recompute or error). Vectors are an input contract
  between Beacon indexing and ContextPilot.

### ADR-012 — Pin the library to Python 3.12
- **Date:** 2026-05-31 · **Status:** Accepted
- **Context:** PyTorch / sentence-transformers / FlagEmbedding wheels are unreliable on
  Python 3.14 (the local default). The pure V1 had no such constraint.
- **Decision:** The `contextpilot` repo targets **Python 3.12**; uv manages a 3.12 venv.
  `requires-python` updated accordingly.
- **Consequences:** Reliable ML installs. Beacon's backend should align to 3.12 too
  (resolves PD-3). Contributors need 3.12 available (uv can fetch it).

### ADR-013 — Default models: BGE-M3 dense + bge-reranker-v2-m3
- **Date:** 2026-05-31 · **Status:** Accepted
- **Context:** Need strong, multilingual, locally-runnable defaults that pair well.
- **Decision:** Default embeddings **`BAAI/bge-m3`** (dense, dim 1024); default reranker
  **`BAAI/bge-reranker-v2-m3`** (cross-encoder). Both configurable. BGE-M3's
  sparse/multi-vector (ColBERT) modes are a later enhancement; V2 starts dense.
- **Consequences:** Consistent with Beacon's embedding choice (ADR-108). Models are
  downloaded on first use (document size/cache); offline use needs a pre-pull.

### ADR-014 — Compression stays extractive; no generative LLM in the path
- **Date:** 2026-05-31 · **Status:** Accepted (reaffirms a hard product rule)
- **Decision:** Compression selects original sentences/paragraphs via embedding +
  lexical + entity/heading scoring; it never rewrites text and never calls a generative
  LLM. The only generative call is the final answer, made downstream by the app.
- **Consequences:** No hallucination risk from compression; real token/compute savings;
  excludes LLM-JSON "structured compression" from the main path.

### ADR-015 — MMR diversity selection
- **Date:** 2026-05-31 · **Status:** Accepted
- **Decision:** Selection uses Maximal Marginal Relevance
  (`λ·relevance − (1−λ)·max_sim(selected)`, λ configurable) so the chosen set is
  relevant *and* non-redundant.
- **Consequences:** Richer evidence sets; one more configurable knob; needs vectors for
  the similarity term.

### ADR-017 — GPU torch via CUDA 12.8 build (RTX 5070 Ti / Blackwell)
- **Date:** 2026-05-31 · **Status:** Accepted
- **Context:** The dev machine has an NVIDIA RTX 5070 Ti Laptop GPU (12GB, Blackwell,
  compute sm_120, driver 596.36 / CUDA 13.2). The default PyPI torch is a CPU build and
  won't use the GPU; Blackwell needs a CUDA 12.8+ build.
- **Decision:** Install torch from the PyTorch **cu128** index via uv
  (`[[tool.uv.index]]` + `[tool.uv.sources]`). Models run on GPU when available; torch
  falls back to CPU automatically when CUDA is absent (GPU is preferred, not required).
- **Consequences:** Larger download (~2.6GB) and the lockfile pins a `+cu128` build. The
  cu128 index's latest is torch **2.11.0+cu128** (vs 2.12 on CPU PyPI) — acceptable.
  Verified: `cuda.is_available()` True, capability (12,0), GPU matmul runs. Beacon's
  embedding/OCR/Ollama GPU usage should align with this (CUDA 12.8 build) too.

### ADR-018 — Reranker drives selection; relevance floor prunes noise
- **Date:** 2026-05-31 · **Status:** Accepted · **Builds on:** ADR-015/016
- **Context:** A 20-block real-model run exposed that `rerank_score` was used only for the
  top-N cutoff, not for selection. The hybrid `final_score` (which includes
  token_efficiency) let tiny low-relevance blocks (receipts, boilerplate) outrank and
  crowd out the large, highly-relevant JD — which was then *dropped* for lack of budget
  instead of compressed.
- **Decision:** (a) After reranking, blend the **min-max-normalized `rerank_score`** into
  `final_score` with a dominant weight (`rerank_weight`, default 0.7) so the cross-encoder
  drives MMR + budgeting. (b) Add a **`relevance_floor`** (default 0.15) that drops
  optional blocks below it *before* budgeting, so cheap noise can't consume budget; this
  frees space so high-value large blocks compress-to-fit rather than drop.
- **Consequences:** Selection now reflects the strongest neural signal. The floor is
  set-relative (rerank is min-max normalized) — in an all-relevant set it may drop the
  lowest; default is modest and per-preset (speed 0.2, quality 0.10). Verified on the
  20-block demo: relevant JD compressed+included, receipts/menus/boilerplate dropped.

### ADR-019 — Compression is relevance-driven, not target-driven
- **Date:** 2026-05-31 · **Status:** Accepted · **Supersedes part of:** ADR-014 mechanics
- **Context:** Compression originally trimmed a block down to the *remaining token budget*
  (a target). The user noted this is artificial — cutting relevant content to hit a number
  is wrong; compression should remove *irrelevant* content and let the size fall out of
  what's relevant.
- **Decision:** Extractive compression keeps sentences whose score is ≥
  ``compression_keep_ratio × best_sentence_score`` (relative to the block's own best
  sentence) and drops the rest — **no token target**. An all-relevant block keeps all its
  sentences; a boilerplate-heavy block keeps only the on-topic ones. The token budget is a
  **hard bound only**: if the relevance-pruned block still exceeds remaining budget, the
  block is **dropped** (never truncated to fit). Still extractive, still no LLM (ADR-014).
- **Consequences:** Compression output is content-determined and model/length-robust
  (relative gate, not an absolute cosine threshold). Some large-but-fully-relevant blocks
  may be dropped rather than partially included — an intentional, honest trade recorded in
  the audit. `compression_keep_ratio` (default 0.5) tunes aggressiveness.

### ADR-016 — Token-aware budgeting as value-per-token optimization
- **Date:** 2026-05-31 · **Status:** Accepted · **Builds on:** ADR-009
- **Decision:** Budgeting picks the best *set* under the token budget (knapsack-style,
  greedy by value/token with required-first reservation), compressing high-value large
  blocks before dropping — not naive top-k stuffing.
- **Consequences:** Better information density per token; more complex selection logic;
  required-over-budget behavior (ADR-009) still holds.

### ADR-020 — Per-stage trace in the audit (observability)
- **Date:** 2026-05-31 · **Status:** Accepted
- **Context:** Consumers (Beacon's "ContextPilot Insights" dashboard, and any library
  user) need to *see* the optimization funnel — what each stage took in, emitted, dropped,
  and how long it took — not just the final block decisions. The block-level `decisions`
  already explain individual fates but don't expose the stage-by-stage shape or timings.
- **Decision:** `AuditReport` gains an optional `stages: list[StageRecord]`
  (stage name, `blocks_in/out`, `tokens_in/out`, `dropped`, `duration_ms`), emitted by a
  tiny `StageTracer` the optimizer feeds one row per stage (exact_dedup, embed_rank,
  rerank, semantic_dedup, mmr, budget). Tracing is **on by default** (cheap — timing +
  cached token sums) and disabled via `ContextPilot(..., trace=False)`. `tokens_out` for
  the budget stage uses the post-compression `used_tokens` so the row reflects real
  shrinkage. The field is additive/backward-compatible (empty list when disabled).
- **Consequences:** Any caller can render or persist the funnel; negligible overhead;
  one more public dataclass (`StageRecord`). Stage names are a soft contract — adding a
  stage appends a row rather than breaking shape.
