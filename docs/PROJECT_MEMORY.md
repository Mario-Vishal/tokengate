# ContextPilot — PROJECT MEMORY

> **This is the single source of truth for the ContextPilot library.**
> Update it after every meaningful work session. Newest "Session log" entry on top.

Last updated: **2026-05-30** (CP-001 done)

---

## Product goal

ContextPilot is a **reusable, app-agnostic Python library** that optimizes the
context sent to an LLM. Given a user query and many candidate context blocks, it
decides what to **include, drop, compress, rank, and budget**, builds a final
prompt, and emits a full **audit report** explaining every decision.

It is the optimization **engine**. The first application built on top of it is the
**Beacon** desktop app (separate repo). ContextPilot must never depend on Beacon,
Tauri, React, LanceDB, Ollama, or any local-folder logic.

## Current scope (V1)

- `ContextBlock` model (the unit everything becomes).
- `OptimizationResult`, `AuditReport`, `BlockDecision` models.
- Token counter (pluggable; heuristic default, optional `tiktoken`).
- Exact deduplication.
- Keyword scoring.
- Hybrid scoring (provided semantic score + keyword score).
- Greedy token budgeter (required blocks always included).
- Extractive compression.
- Prompt builder (cacheable/stable sections first).
- `ContextPilot` optimizer orchestrating the full pipeline.
- Unit tests for all of the above.
- Installable package via `uv` / `pip`.

## Out of scope (deferred to V2+)

- Semantic / near-duplicate deduplication (embeddings-based).
- Reranker adapters (e.g. BGE).
- Structured LLM-based compression (JSON schema via an LLM).
- Strategy presets beyond a minimal `balanced` default (speed/quality/max_compression).
- Cache-aware prompt sectioning beyond simple ordering.
- Optimization cache, benchmark utilities.
- Any LLM calls, vector stores, network I/O, or file I/O for user data.

## Architecture decisions (summary — full records in DECISIONS.md)

- **ADR-001** `src/` layout, package importable as `contextpilot`.
- **ADR-002** uv for env/deps; `pyproject.toml` (PEP 621); MIT license.
- **ADR-003** Token counting is pluggable via a `TokenCounter` protocol; default is a
  dependency-free heuristic, with optional `tiktoken` extra.
- **ADR-004** Core is pure: no network, no disk I/O for user data, no LLM calls.
- **ADR-005** Semantic scores are *provided by the caller* on each `ContextBlock`;
  the library never computes embeddings in V1.
- **ADR-006** Every optimize() call returns a complete, serializable audit.
- **ADR-007** Naming: library = `contextpilot`; desktop app = `Beacon` (repo `beacon`).

## Current implementation status

**LIBRARY V1 COMPLETE (v0.1.0).** Full pipeline implemented + tested (114 passing,
mypy strict + ruff clean). Definition of Done met (see below). Ready to begin Beacon.

### V1 Definition of Done
- [x] Package installs locally (`uv sync`).
- [x] ContextBlock, OptimizationResult, AuditReport, BlockDecision implemented.
- [x] Deduplication (exact) implemented.
- [x] Ranking (keyword/normalize/hybrid) implemented.
- [x] Budgeting (greedy, required-first, compress-to-fit) implemented.
- [x] Prompt builder implemented.
- [x] Optimizer (`ContextPilot.optimize`) implemented.
- [x] Audit report implemented.
- [x] Unit tests passing (114 passed, 1 skipped = optional tiktoken path).
- [x] README + LIBRARY_API updated and example verified.
- [x] PROJECT_MEMORY updated.


- [x] Repos initialized locally (`contextpilot`, `beacon`), git `main` branch.
- [x] LICENSE (MIT), `.gitignore`, README created.
- [x] Planning docs created (this file, ARCHITECTURE, ROADMAP, DECISIONS, ERROR_LOG, TODO, LIBRARY_API).
- [x] **CP-001:** `pyproject.toml` (uv, src layout, hatchling) + full package skeleton
      with typed stubs + `py.typed`. `uv sync` green on Python 3.14; `import contextpilot`
      works; 2 smoke tests pass.
- [x] CP-002 utils (errors/logging/hashing) + tests.
- [x] CP-003 `ContextBlock` model + tests.
- [x] CP-004 result/audit models (`OptimizationResult`/`AuditReport`/`BlockDecision`)
      + `build_audit_report` + tests.
- [x] CP-005 `OptimizerConfig` + strategy preset + `ConfigurationError` + tests.
      Suite at 48 passing, ruff clean.
- [x] CP-006 token counting (heuristic + optional tiktoken) + tests.
- [x] CP-007 exact dedup + tests.
- [x] CP-008 ranking (keyword/normalize/hybrid) + tests.
- [x] CP-009 extractive compression + tests.
- [x] CP-010 greedy budgeter + tests (ADR-009).
- [x] CP-011 prompt builder + tests.
- [x] CP-012 optimizer (end-to-end) + public API + tests.
- [x] CP-013 hardening: mypy strict + ruff clean, 114 passing, tag v0.1.0.

## Completed work

- 2026-05-30: Project kickoff. Confirmed scope, naming, tooling. Created repo
  skeletons and full planning doc set for the library.
- 2026-05-30: **CP-001** — package skeleton + `pyproject.toml`. Verified `uv sync`,
  import, and pytest (2 passing) on Python 3.14.0.
- 2026-05-30: **CP-002** — utils: error hierarchy (`ContextPilotError` + subclasses,
  exported at top level), deterministic `hashing` (sha256, short ids), library-safe
  structured `logging` (`get_logger`/`log_event`, NullHandler). +8 tests.
- 2026-05-30: **CP-003** — `ContextBlock` dataclass with validation (non-empty
  content, scores in [0,1], non-negative tokens), stable auto `block_id` from content
  hash, `ensure_token_count(counter)`, `to_dict`/`from_dict`/`copy`. Exported at top
  level. +11 tests. Suite: **29 passing**.

## User feedback and requirements

### 2026-05-31 — CP-016 done: block vectors (reuse or compute)
- `ContextBlock.vector: np.ndarray | None` (coerced to 1-D float32; validated; excluded
  from eq/repr to avoid ndarray-eq ambiguity; serialized as list in `to_dict`, round-trips
  via `from_dict`; preserved by `copy`).
- `models/vectors.py`: `ensure_block_vectors(blocks, model)` reuses present
  correctly-sized vectors, batch-computes missing ones (and recomputes dim-mismatched,
  logged), caches back, returns `(n, dim)` matrix in order; `embed_query(query, model)`.
- `tests/test_vectors.py`. ruff + mypy(strict) clean; **138 passed, 2 skipped**.
- Next: CP-017 (semantic scoring + expanded hybrid ranking).

### 2026-05-31 — CP-015 done: model layer (protocols + BGE + fakes)
- `models/base.py`: `EmbeddingModel`/`Reranker` runtime-checkable protocols (embeddings
  L2-normalized → cosine==dot); `resolve_device(auto|cpu_only|force_gpu)`.
- `models/bge.py`: `BGEM3Embedder` (BAAI/bge-m3 dense, dim 1024) + `BGEReranker`
  (BAAI/bge-reranker-v2-m3) via sentence-transformers; GPU-aware; lazy heavy imports;
  weights download on first construct. Version-robust embedding-dim lookup.
- `models/fakes.py`: `FakeEmbeddingModel` (stable md5-hashing BoW, similarity reflects
  word overlap) + `FakeReranker` (lexical overlap) — test-only, deterministic, offline.
- mypy override for sentence_transformers/FlagEmbedding (no stubs).
- `tests/test_models.py`: fakes (fast) + opt-in real-model test gated by
  `CONTEXTPILOT_TEST_REAL_MODELS=1`.
- **Verified real models on the RTX 5070 Ti**: BGE-M3 dim 1024, reranker scores relevant
  > irrelevant (downloaded ~4.5GB to HF cache `C:\Users\mario\.cache\huggingface`).
- ruff + mypy(strict) clean; **127 passed, 2 skipped** (tiktoken + real models).
- Next: CP-016 (block vectors — reuse or compute).

### 2026-05-31 — CP-014 done: Python 3.12 + ML deps
- Pinned `requires-python = ">=3.12,<3.13"`, `.python-version` 3.12; mypy python_version
  3.12. Added required core deps: `numpy`, `torch`, `sentence-transformers` (+ optional
  `flag` extra for future BGE-M3 sparse/ColBERT).
- `uv sync` on **Python 3.12.13** OK: torch **2.12.0+cpu**, sentence-transformers 5.5.1,
  transformers 5.9, numpy 2.4.6 (also pulled scikit-learn/scipy). **torch is the CPU
  build** — GPU optional; for CUDA on this Windows box we'd install the CUDA torch wheel
  later (not needed for correctness).
- V1 suite still green on 3.12; added `tests/test_env.py`.
- **GPU (ADR-017):** dev box has **RTX 5070 Ti Laptop (12GB, Blackwell sm_120, driver
  596.36 / CUDA 13.2)**. Switched torch to the **cu128** index via uv sources →
  **torch 2.11.0+cu128**. Verified `cuda.is_available()` True, capability (12,0), real
  GPU matmul runs. CPU fallback automatic when CUDA absent. **119 passed, 1 skipped.**
- Resolves PD-3: **Beacon backend should also pin Python 3.12** and prefer a CUDA 12.8
  build for GPU paths (embeddings/OCR), with CPU fallback.
- Git history in both repos rewritten to drop the Claude co-author trailer; force-pushed.
  Future commits omit it (see [[no-claude-coauthor]]).
- Next: CP-015 (model layer — protocols + BGE defaults + fakes).

### 2026-05-31 — Mature vision: neural context optimization (V2 direction)
User paused Beacon and specified the "mature" ContextPilot: a **neural** context
optimization engine. Full spec captured in [`V2_DESIGN.md`](V2_DESIGN.md). Essentials:
- Pipeline: normalize → semantic+keyword scoring → **neural reranking** → **semantic
  dedup** → extractive compression → **MMR diversity** → token-aware budgeting → prompt
  → audit.
- **Embedding + reranker models select/score context; a generative LLM is used only
  once downstream for the final answer. NO generative/LLM compression in the main path
  — extractive only.**
- Hybrid ranking expands to multi-signal (semantic, keyword, recency, source_priority,
  token_efficiency). Budgeting becomes value-per-token "best set under budget".
- Audit gains `models_used`, `rerank_score`, richer reasons.
- Retrieval (LanceDB top-50) stays in the app, before ContextPilot — unchanged.

**User correction (2026-05-31):** the sophisticated techniques are **NOT optional** —
neural is the engine. Approved direction: embeddings + cross-encoder reranker are
**required, first-class core** (configurable, never optional); ContextPilot takes real
ML deps (PyTorch + BGE models); **no deterministic product fallback**. Approved
decisions: reuse caller-provided block vectors (model computes query/sentence
embeddings + reranking); **pin library to Python 3.12**; defaults **BGE-M3 dense +
bge-reranker-v2-m3**. This **supersedes ADR-004** (pure core) and **amends ADR-005**
(caller-provided scores → reuse vectors, but model always present). Full design in
[`V2_DESIGN.md`](V2_DESIGN.md). **Next: present sequential V2 task breakdown for user
review BEFORE writing any code (user requires explicit approval to proceed).**

### 2026-05-30 — original requirements

- Two **separate** repos; library must be app-agnostic, app must import the library
  (no duplicated core logic).
- Repos are **private** on GitHub (to be created after user installs `gh` + auths).
- Desktop app gets a **standalone name** distinct from the library → **Beacon**.
- Tooling: **uv**. License: **MIT**.
- Keep modules small and testable; tests for every meaningful module.
- Solve TODOs **sequentially**; don't jump ahead unless blocked.
- Don't write product code until planning docs exist; library V1 must work + pass
  tests before Beacon implementation begins.

## Errors faced / root causes / fixes

None yet. See [`ERROR_LOG.md`](ERROR_LOG.md).

## Tests added

None yet. Planned set listed in [`TODO.md`](TODO.md) and ROADMAP.

## Important findings

- Local environment: **Python 3.14.5**, Node v24.16 (Windows 11). Python 3.14 is
  very new — watch for wheel availability on optional deps (`tiktoken`). The core
  library is intentionally dependency-light to avoid this risk.

## Open questions

- Exact `tiktoken` encoding to default to when the extra is installed (likely
  `cl100k_base`) — to be confirmed when wiring the token counter.
- Final copyright holder name for LICENSE (currently "ContextPilot Authors").

## Next tasks

See [`TODO.md`](TODO.md). Immediate next: **CP-001** — create `pyproject.toml` and
the package skeleton under `src/contextpilot/`.

## Commands used

```powershell
git init ; git symbolic-ref HEAD refs/heads/main     # per repo
# planned:
uv init / uv sync / uv run pytest
```

## Dependencies added

- Runtime: none yet (target: zero required runtime deps for the core).
- Optional extras (planned): `tiktoken` (accurate token counting).
- Dev (installed via uv group): `pytest` 9.0.3, `ruff` 0.15.15, `mypy` 2.1.0.

## Demo status

No demo yet. First demo target: `pilot.optimize(...)` on a handful of in-memory
blocks producing a prompt + audit (end of library V1).

---

## Session log

### 2026-05-30 — Kickoff & planning
- Confirmed with user: 2 private repos, MIT, uv, desktop app named **Beacon**.
- Initialized local git repos for `contextpilot` and `beacon` (branch `main`).
- Wrote LICENSE, `.gitignore`, README, and all seven library planning docs.
- GitHub remotes deferred until user installs `gh` and authenticates.
- Next: scaffold package + `pyproject.toml` (CP-001).

### 2026-05-30 — CP-001 package skeleton
- Added `pyproject.toml` (uv-managed, src layout, hatchling build, optional `tiktoken`
  extra, dev group pytest/ruff/mypy). Created full `src/contextpilot/` tree with typed
  stub modules tagged by their implementing task, `py.typed`, and `tests/`.
- Verified on Python 3.14.0: `uv sync` ok (no native-wheel issues — core has 0 runtime
  deps), `import contextpilot` ok, `uv run pytest` → 2 passed.
- Committed. Next: CP-002 (utils: errors/logging/hashing).

### 2026-05-30 — CP-011 prompt builder + CP-012 optimizer + CP-013 hardening → V1 DONE
- CP-011: `prompts/prompt_builder.py` `build_prompt()` — cacheable/stable sections
  first, numbered context section with source annotations, query last; deterministic.
  `tests/test_prompt_builder.py`.
- CP-012: `core/optimizer.py` `ContextPilot.optimize()` wiring dedup→rank→budget→
  prompt→audit, with input validation and `OptimizationError` wrapping; exported at
  top level. Also reworked the budgeter to **reserve required-block tokens up front**
  (so optional selection accounts for them) and emit `prompt_blocks` in ranked order.
  `tests/test_optimizer.py` (end-to-end + edges: empty, dups, over-budget, required,
  serialization, determinism).
- Verified the documented README/LIBRARY_API example runs. Noted that `tokens_saved`
  can be negative for tiny inputs under a large budget (scaffolding overhead) — honest,
  documented in LIBRARY_API.
- CP-013: added mypy override for optional `tiktoken`. **mypy(strict) + ruff clean;
  `uv run pytest` → 114 passed, 1 skipped.** Tagged **v0.1.0**.
- **ContextPilot library V1 Definition of Done met.** Next: begin Beacon (BK-001).

### 2026-05-30 — CP-009 compression + CP-010 budgeter
- CP-009: `compression/extractive.py` — `split_sentences`, `compress_text`
  (query-relevant sentence selection under a token target; never empties; returns
  original when it already fits or is single-sentence), `compress_block` (respects
  `compressible=False`; tags `metadata.compressed`/`original_token_count`).
  `tests/test_compression.py`.
- CP-010: `budgeting/budgeter.py` — `budget_blocks()` → `BudgetOutcome`
  (included/compressed/dropped + decisions + used_tokens). Required-first (always
  kept, **ADR-009**), optional fit-or-compress-to-fit-or-drop, greedy continues after
  a drop. Compressed blocks live in `compressed` (decision=compressed), not
  double-counted in `included`; prompt = included + compressed. `tests/test_budgeter.py`.
- ruff clean; `uv run pytest` → **95 passed, 1 skipped**. Next: CP-011 prompt builder.

### 2026-05-30 — CP-007 exact dedup + CP-008 ranking
- CP-007: `deduplication/exact.py` `deduplicate_exact()` → `(kept, dropped)`;
  collapses byte-identical content, keeps best representative (required > higher score
  > first occurrence), preserves first-occurrence order. `tests/test_deduplication.py`.
- CP-008: `ranking/keyword_ranker.py` (tokenize, distinct-coverage raw keyword score),
  `ranking/score_normalizer.py` (`normalize_min_max`, safe all-equal/empty),
  `ranking/hybrid_ranker.py` (`combine_scores` weighted avg + `rank_blocks` → sets
  `keyword_score`/`final_score`, returns stable-sorted by `final_score`). Keyword-only
  ranking works when no semantic score. `tests/test_ranking.py`.
- ruff clean; `uv run pytest` → **78 passed, 1 skipped**. Next: CP-009 (compression).

### 2026-05-30 — CP-006 token counting
- `budgeting/token_counter.py`: `TokenCounter` runtime-checkable Protocol;
  `HeuristicTokenCounter` (dependency-free, `max(ceil(chars/4), word_count)`,
  conservative + monotonic); `TiktokenCounter` (optional extra, clear ImportError when
  missing); `resolve_counter()` default helper. Exported `TokenCounter` +
  `HeuristicTokenCounter` at top level. `tests/test_token_counter.py`.
- GitHub: created+pushed private repos (account Mario-Vishal). `gh` lives at
  `C:\Program Files\GitHub CLI\gh.exe` (not on session PATH).
- `uv run pytest` → **56 passed, 1 skipped** (tiktoken not installed). Next: CP-007.

### 2026-05-30 — CP-004 result/audit models + CP-005 config
- CP-004: `core/result.py` (`BlockDecision`, `AuditReport`, `OptimizationResult` +
  decision constants, all serializable) and `audit/audit_report.py`
  (`build_audit_report` — derives counts + tokens_saved/percent, safe on 0 tokens,
  reports negative savings honestly). Models exported top-level. `tests/test_result.py`.
- CP-005: `core/config.py` `OptimizerConfig` (max_prompt_tokens, strategy, weights,
  toggles, safety_margin, optional token_counter) with validation, `effective_budget`
  property, `for_strategy()` preset factory; added `ConfigurationError` to the error
  hierarchy. `tests/test_config.py`.
- ruff clean; `uv run pytest` → **48 passed**. Next: CP-006 (token counting).

### 2026-05-30 — CP-002 utils + CP-003 ContextBlock
- CP-002: `utils/errors.py` (ContextPilotError → InvalidBlockError/BudgetError/
  OptimizationError, exported top-level), `utils/hashing.py` (deterministic sha256 +
  `block_id_from_content`), `utils/logging.py` (`get_logger` w/ NullHandler,
  `log_event` structured fields). `tests/test_utils.py`.
- CP-003: `core/block.py` `ContextBlock` dataclass — validation, auto stable id,
  `ensure_token_count`, dict round-trip + `copy`. Exported top-level.
  `tests/test_block.py`.
- `uv run pytest` → **29 passed**. Next: CP-004 (result + audit models).
