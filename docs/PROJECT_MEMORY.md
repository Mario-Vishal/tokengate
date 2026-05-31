# ContextPilot — PROJECT MEMORY

> **This is the single source of truth for the ContextPilot library.**
> Update it after every meaningful work session. Newest "Session log" entry on top.

Last updated: **2026-05-30**

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

**Phase 0 — Planning. No product code yet.**

- [x] Repos initialized locally (`contextpilot`, `beacon`), git `main` branch.
- [x] LICENSE (MIT), `.gitignore`, README created.
- [x] Planning docs created (this file, ARCHITECTURE, ROADMAP, DECISIONS, ERROR_LOG, TODO, LIBRARY_API).
- [ ] `pyproject.toml` + package skeleton.
- [ ] Core models, pipeline modules, optimizer.
- [ ] Tests.

## Completed work

- 2026-05-30: Project kickoff. Confirmed scope, naming, tooling. Created repo
  skeletons and full planning doc set for the library.

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
- Dev (planned): `pytest`, `ruff`, `mypy`.

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
