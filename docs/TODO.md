# ContextPilot — TODO (sequential)

Work tasks **in order**. Do not jump ahead unless the current task is `blocked`.
Statuses: `pending` · `in-progress` · `done` · `blocked`.

Legend per task: **Repo** = which repo · **Files** = expected changes ·
**Accept** = acceptance criteria · **Tests** = tests required · **Docs** = doc updates.

---

## CP-000 — Planning docs & repo skeleton
- **Status:** done
- **Description:** Initialize repo, license, gitignore, README, and all seven docs.
- **Repo:** contextpilot
- **Files:** `LICENSE`, `.gitignore`, `README.md`, `docs/*.md`
- **Accept:** all planning docs exist; git repo on `main`.
- **Tests:** n/a
- **Docs:** this file + PROJECT_MEMORY created.

## CP-001 — Package skeleton + pyproject (uv)
- **Status:** done
- **Description:** Create `pyproject.toml` (PEP 621, src layout, optional `tiktoken`
  extra, dev deps pytest/ruff/mypy). Create `src/contextpilot/` tree with
  `__init__.py` and empty typed module files per ARCHITECTURE. Add `py.typed`.
- **Repo:** contextpilot
- **Files:** `pyproject.toml`, `src/contextpilot/__init__.py`, all subpackage
  `__init__.py` + stub modules, `src/contextpilot/py.typed`, `tests/__init__.py`
- **Accept:** `uv sync` succeeds; `uv run python -c "import contextpilot"` works;
  `uv run pytest` runs (0 tests) green.
- **Tests:** none (skeleton) — add a trivial import test.
- **Docs:** PROJECT_MEMORY (status, deps, commands).

## CP-002 — utils: errors, logging, hashing
- **Status:** done
- **Description:** Exception hierarchy rooted at `ContextPilotError`
  (`InvalidBlockError`, `BudgetError`, `OptimizationError`). Structured logger
  factory. Stable content hashing.
- **Repo:** contextpilot
- **Files:** `utils/errors.py`, `utils/logging.py`, `utils/hashing.py`
- **Accept:** errors importable from `contextpilot`; `hash_content` stable &
  deterministic; logger emits structured records.
- **Tests:** `tests/test_utils.py` — hashing determinism, error hierarchy.
- **Docs:** PROJECT_MEMORY; DECISIONS if logging format chosen.

## CP-003 — Core model: ContextBlock
- **Status:** done
- **Description:** Implement `ContextBlock` with all fields, validation (non-empty
  content, score ranges), auto `block_id` from content hash when omitted, and a
  `token_count` accessor that uses an injected `TokenCounter` when None.
- **Repo:** contextpilot
- **Files:** `core/block.py`
- **Accept:** invalid blocks raise `InvalidBlockError`; ids stable; round-trips
  to/from dict.
- **Tests:** `tests/test_block.py` — validation, id derivation, serialization.
- **Docs:** PROJECT_MEMORY; LIBRARY_API.

## CP-004 — Core models: results & audit
- **Status:** done
- **Description:** Implement `OptimizationResult`, `AuditReport`, `BlockDecision`
  with serialization. Audit holds counts + token math + per-block decisions.
- **Repo:** contextpilot
- **Files:** `core/result.py`, `audit/audit_report.py`
- **Accept:** models serialize; `tokens_saved_percent` computed safely (no div/0).
- **Tests:** `tests/test_result.py` — construction, serialization, percent math.
- **Docs:** PROJECT_MEMORY; LIBRARY_API.

## CP-005 — Config & strategy
- **Status:** done
- **Description:** `OptimizerConfig` (max_prompt_tokens, strategy, weights, toggles,
  safety margin). `balanced` preset. Validate values.
- **Repo:** contextpilot
- **Files:** `core/config.py`
- **Accept:** invalid config raises; `balanced` returns sane defaults.
- **Tests:** `tests/test_config.py`.
- **Docs:** DECISIONS (weights), LIBRARY_API.

## CP-006 — Token counting
- **Status:** done
- **Description:** `TokenCounter` protocol; `HeuristicTokenCounter` default;
  optional `TiktokenCounter` (guarded import). Safety-margin helper.
- **Repo:** contextpilot
- **Files:** `budgeting/token_counter.py`
- **Accept:** heuristic returns positive ints, monotonic with length; tiktoken path
  skipped gracefully when not installed.
- **Tests:** `tests/test_token_counter.py`.
- **Docs:** PROJECT_MEMORY; LIBRARY_API.

## CP-007 — Exact deduplication
- **Status:** done
- **Description:** Remove exact-content duplicate blocks (hash-based), keeping the
  highest-scored / first occurrence; record dropped duplicates for audit.
- **Repo:** contextpilot
- **Files:** `deduplication/exact.py`
- **Accept:** duplicates removed; one representative kept deterministically.
- **Tests:** `tests/test_deduplication.py`.
- **Docs:** PROJECT_MEMORY.

## CP-008 — Ranking (keyword, normalize, hybrid)
- **Status:** done
- **Description:** Keyword overlap scoring vs query; min-max normalization to [0,1];
  hybrid combine of semantic + keyword via config weights → `final_score`.
- **Repo:** contextpilot
- **Files:** `ranking/keyword_ranker.py`, `ranking/score_normalizer.py`,
  `ranking/hybrid_ranker.py`
- **Accept:** scores in [0,1]; blocks with no semantic score still rank by keyword;
  weighting respected.
- **Tests:** `tests/test_ranking.py`.
- **Docs:** PROJECT_MEMORY; LIBRARY_API.

## CP-009 — Extractive compression
- **Status:** done
- **Description:** Sentence-selection compression of large `compressible` blocks,
  keeping query-relevant sentences under a target ratio/token target.
- **Repo:** contextpilot
- **Files:** `compression/extractive.py`
- **Accept:** output tokens ≤ original; `compressible=False` untouched; never empties
  content.
- **Tests:** `tests/test_compression.py`.
- **Docs:** PROJECT_MEMORY; LIBRARY_API.

## CP-010 — Greedy budgeter
- **Status:** done (see ADR-009 for required-over-budget behavior)
- **Description:** Select blocks (ranked, required-first) under `max_prompt_tokens`
  using the token counter + safety margin; trigger compression for high-value large
  blocks before dropping; record drops.
- **Repo:** contextpilot
- **Files:** `budgeting/budgeter.py`
- **Accept:** total selected tokens ≤ budget; required blocks always included even
  over nominal budget (documented behavior); deterministic.
- **Tests:** `tests/test_budgeter.py` — budget respected; required kept; drops
  recorded.
- **Docs:** PROJECT_MEMORY; DECISIONS (required-over-budget behavior).

## CP-011 — Prompt builder
- **Status:** done
- **Description:** Assemble `final_prompt` from selected blocks: cacheable/stable
  sections first, then query-relevant blocks, then the query. Deterministic format.
- **Repo:** contextpilot
- **Files:** `prompts/prompt_builder.py`
- **Accept:** cacheable blocks ordered before non-cacheable; output deterministic;
  query present.
- **Tests:** `tests/test_prompt_builder.py`.
- **Docs:** PROJECT_MEMORY; LIBRARY_API.

## CP-012 — Optimizer (end-to-end) + public API
- **Status:** done
- **Description:** `ContextPilot.optimize()` wiring dedup → rank → compress →
  budget → build → audit, returning `OptimizationResult`. Export public symbols in
  `__init__.py`.
- **Repo:** contextpilot
- **Files:** `core/optimizer.py`, `src/contextpilot/__init__.py`
- **Accept:** README example runs; audit counts reconcile; budget respected;
  required always included; dropped/compressed recorded.
- **Tests:** `tests/test_optimizer.py` — end-to-end + edge cases (empty blocks, all
  required, everything over budget).
- **Docs:** PROJECT_MEMORY; LIBRARY_API finalized.

## CP-013 — V1 hardening
- **Status:** done — mypy(strict)+ruff clean, 114 tests pass, README example verified, tag v0.1.0
- **Description:** ruff + mypy clean; coverage review; verify README usage; tag
  `v0.1.0`. Confirm Definition of Done (V1).
- **Repo:** contextpilot
- **Files:** config tweaks as needed
- **Accept:** all V1 DoD items checked in PROJECT_MEMORY.
- **Tests:** full suite green.
- **Docs:** PROJECT_MEMORY, ROADMAP (Phase 8 done).

---

---

# V2 — Neural engine (CP-014 … CP-024)

> Direction approved 2026-05-31 (ADR-010..016, [`V2_DESIGN.md`](V2_DESIGN.md)).
> **GATE: do not begin CP-014 until the user signs off on this breakdown.** Sequential.
> Neural is required core (not optional). No generative LLM in the path.

## CP-014 — Environment & ML dependencies (Python 3.12)
- **Status:** done — Py 3.12.13, torch 2.12 (CPU), sentence-transformers 5.5.1; 117 tests pass
- **Description:** Set `requires-python = ">=3.12,<3.13"`; create uv 3.12 venv; add core
  deps `torch`, `sentence-transformers` and/or `FlagEmbedding`, `numpy`. Verify they
  install and import; document download/cache behavior of models.
- **Repo:** contextpilot
- **Files:** `pyproject.toml`, `uv.lock`, PROJECT_MEMORY
- **Accept:** `uv sync` on 3.12 succeeds; `import torch`, embedder/reranker libs import.
- **Tests:** import smoke test (skipped if libs absent in CI).
- **Docs:** PROJECT_MEMORY (deps, commands), ERROR_LOG if wheel issues, DECISIONS (ADR-012).

## CP-015 — Model layer (protocols + BGE defaults + fakes)
- **Status:** done — protocols + BGE adapters + fakes; real models verified on GPU (1024-dim, reranker orders correctly)
- **Description:** `models/` package: `EmbeddingModel` + `Reranker` protocols;
  `BGEM3Embedder` (BAAI/bge-m3, dense, GPU-aware), `BGEReranker`
  (BAAI/bge-reranker-v2-m3); deterministic `FakeEmbeddingModel`/`FakeReranker` for tests.
- **Repo:** contextpilot
- **Files:** `models/__init__.py`, `models/base.py`, `models/bge.py`, `models/fakes.py`
- **Accept:** fakes work offline; real models load when present; `dim` exposed; GPU used
  if available else CPU.
- **Tests:** fakes; cosine sanity; real-model test opt-in/skipped.
- **Docs:** PROJECT_MEMORY; LIBRARY_API (model interfaces); DECISIONS (ADR-013).

## CP-016 — Block vectors (reuse or compute)
- **Status:** done — ContextBlock.vector + ensure_block_vectors (reuse/compute/dim-mismatch) + tests
- **Description:** Add `vector: Sequence[float] | None` to `ContextBlock`; helper to
  ensure vectors (reuse if present, else embed via model); validate dim vs model.
- **Repo:** contextpilot
- **Files:** `core/block.py`, `models/` helper, tests
- **Accept:** provided vectors reused; missing ones computed; dim mismatch raises/recomputes.
- **Tests:** reuse path, compute path, dim-mismatch. **Docs:** PROJECT_MEMORY; DECISIONS (ADR-011).

## CP-017 — Semantic scoring + expanded hybrid ranking
- **Status:** done — cosine semantic scorer + recency/source/token-eff signals + multi-signal hybrid (per-block renormalized)
- **Description:** Cosine semantic score (query vec vs block vec, numpy). Expand hybrid
  ranking with recency, source_priority, token_efficiency signals + configurable weights
  with renormalization for missing signals.
- **Repo:** contextpilot
- **Files:** `ranking/semantic_scorer.py`, `ranking/hybrid_ranker.py`, `core/config.py`
- **Accept:** semantic scores in [0,1]; weights renormalize; deterministic. **Tests:**
  scoring + weighting. **Docs:** PROJECT_MEMORY; LIBRARY_API.

## CP-018 — Neural reranking stage
- **Status:** done — cross-encoder rerank stage, rerank_score field, rerank_top_n cutoff + tests
- **Description:** `ranking/reranker_stage.py` — run cross-encoder over (query, chunk);
  set `rerank_score`; keep top `rerank_top_n`.
- **Repo:** contextpilot · **Files:** `ranking/reranker_stage.py`, `core/config.py`
- **Accept:** reranks a candidate set; top-n cutoff honored; fake reranker in tests.
- **Tests:** ordering by rerank; cutoff. **Docs:** PROJECT_MEMORY; LIBRARY_API.

## CP-019 — Semantic deduplication
- **Status:** done — embedding-cosine dedup, best-first representative, reasons recorded + tests
- **Description:** `deduplication/semantic.py` — cosine threshold dedup over block
  vectors; keep best representative (rerank/score/length/recency/source); record reasons.
- **Repo:** contextpilot · **Files:** `deduplication/semantic.py`
- **Accept:** near-duplicates collapsed; best kept; reasons recorded. **Tests:**
  paraphrase pair collapses; distinct survive. **Docs:** PROJECT_MEMORY; DECISIONS (already ADR'd).

## CP-020 — Embedding extractive compression
- **Status:** done — embedding sentence scoring (semantic+keyword+entity/heading) + MMR-style redundancy selection; threaded model through budgeter
- **Description:** Upgrade `compression/extractive.py` to score sentences by
  `semantic_sim + keyword_overlap + entity_bonus + heading_bonus − redundancy`; keep
  original sentences to `target_tokens`; still no LLM.
- **Repo:** contextpilot · **Files:** `compression/extractive.py`
- **Accept:** keeps query-relevant sentences; ≤ target; never empties; records method.
- **Tests:** relevance selection, token target, non-empty. **Docs:** PROJECT_MEMORY; DECISIONS (ADR-014).

## CP-021 — MMR diversity selection
- **Status:** done — selection/mmr.py (relevance−redundancy greedy), enable_mmr/mmr_lambda config + tests
- **Description:** `selection/mmr.py` — `λ·relevance − (1−λ)·max_sim(selected)`; λ configurable.
- **Repo:** contextpilot · **Files:** `selection/mmr.py`, `core/config.py`
- **Accept:** diverse set chosen over near-duplicates; λ extremes behave (1=pure relevance).
- **Tests:** diversity vs redundancy. **Docs:** PROJECT_MEMORY; DECISIONS (ADR-015).

## CP-022 — Value-per-token budgeting
- **Status:** done — optional selection by final_score/tokens density (best set under budget); prompt order preserved
- **Description:** Upgrade budgeter to value/token "best set under budget" (knapsack-style
  greedy), required-first reservation (ADR-009), compress-before-drop.
- **Repo:** contextpilot · **Files:** `budgeting/budgeter.py`
- **Accept:** higher total value within budget than naive top-k on a fixture; invariants hold.
- **Tests:** value-density selection; budget respected; required kept. **Docs:** PROJECT_MEMORY; DECISIONS (ADR-016).

## CP-023 — Optimizer rewire (neural pipeline) + richer audit
- **Status:** done — full neural pipeline wired; models_used + rerank_score in audit; 4 real strategy presets
- **Description:** Rewire `ContextPilot.optimize()` to the full neural order (normalize →
  semantic+keyword → hybrid → rerank → semantic dedup → compression → MMR → budget →
  prompt → audit). Audit gains `models_used`, per-block `rerank_score`, multi-signal
  reasons. Strategy presets (speed/balanced/quality/max_compression) become real.
- **Repo:** contextpilot · **Files:** `core/optimizer.py`, `core/result.py`, `core/config.py`, `audit/audit_report.py`
- **Accept:** end-to-end with fake models; audit reconciles; presets differ measurably.
- **Tests:** end-to-end neural pipeline (fakes); audit fields. **Docs:** PROJECT_MEMORY; LIBRARY_API rewrite.

## CP-024 — V2 hardening & benchmarks
- **Status:** done — real-model integration test (GPU, passing), baseline-vs-neural benchmark, tag v0.2.0
- **Description:** Opt-in integration tests with real BGE models; baseline (V1-style) vs
  neural benchmark on a fixture; mypy/ruff clean; finalize docs; tag `v0.2.0`.
- **Repo:** contextpilot · **Files:** tests, benchmark script, docs
- **Accept:** integration green when models present; benchmark shows neural uplift; V2 DoD met.
- **Tests:** full suite + integration. **Docs:** PROJECT_MEMORY, ROADMAP, LIBRARY_API, V2_DESIGN.
