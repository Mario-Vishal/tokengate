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

**Phase 1 — Package skeleton complete. Implementation of core begins next (CP-002).**

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
- [x] CP-006 token counting (heuristic + optional tiktoken) + tests. Suite **56 passing**.
- [ ] CP-007 exact dedup → next.
- [ ] Remaining pipeline modules (dedup, ranking, compression, budgeter, prompt) + optimizer.

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
