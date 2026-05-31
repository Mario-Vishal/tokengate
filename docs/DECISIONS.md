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
- **Date:** 2026-05-30 · **Status:** Accepted
- **Context:** The library must be reusable and app-agnostic.
- **Decision:** The core performs no network calls, no user-data disk I/O, and
  never calls an LLM. It only transforms inputs into results.
- **Consequences:** Trivially unit-testable and deterministic. LLM-based features
  (e.g. structured compression) must be implemented as optional adapters in V2 that
  the *caller* drives.

### ADR-005 — Semantic scores are caller-provided
- **Date:** 2026-05-30 · **Status:** Accepted
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
