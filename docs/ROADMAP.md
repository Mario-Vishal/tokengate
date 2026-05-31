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

## V2 (do not start until V1 is stable)
- Semantic / near-duplicate deduplication.
- Reranker adapter (e.g. BGE) at the ranking stage.
- Structured LLM compression schema (JSON via an LLM).
- Strategy presets: `speed`, `balanced`, `quality`, `max_compression`.
- Cache-aware prompt sectioning (beyond simple ordering).
- Optimization cache.
- Benchmark utilities + reproducible benchmark suite.
