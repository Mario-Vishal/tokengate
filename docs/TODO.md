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
- **Status:** pending
- **Description:** Assemble `final_prompt` from selected blocks: cacheable/stable
  sections first, then query-relevant blocks, then the query. Deterministic format.
- **Repo:** contextpilot
- **Files:** `prompts/prompt_builder.py`
- **Accept:** cacheable blocks ordered before non-cacheable; output deterministic;
  query present.
- **Tests:** `tests/test_prompt_builder.py`.
- **Docs:** PROJECT_MEMORY; LIBRARY_API.

## CP-012 — Optimizer (end-to-end) + public API
- **Status:** pending
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
- **Status:** pending
- **Description:** ruff + mypy clean; coverage review; verify README usage; tag
  `v0.1.0`. Confirm Definition of Done (V1).
- **Repo:** contextpilot
- **Files:** config tweaks as needed
- **Accept:** all V1 DoD items checked in PROJECT_MEMORY.
- **Tests:** full suite green.
- **Docs:** PROJECT_MEMORY, ROADMAP (Phase 8 done).

---

### Backlog (V2 — do not start until V1 stable)
- Semantic/near-duplicate dedup · reranker adapter · structured LLM compression ·
  strategy presets · cache-aware sectioning · optimization cache · benchmark suite.
