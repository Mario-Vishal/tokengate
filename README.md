# ContextPilot

**ContextPilot** is a reusable, app-agnostic context-optimization library for LLM
applications. Given a user query and many candidate context blocks, it decides
what to **include, drop, compress, rank, and budget** before a prompt is sent to
an LLM — and produces a full **audit report** explaining every decision.

> Status: **Planning / pre-implementation.** See [`docs/PROJECT_MEMORY.md`](docs/PROJECT_MEMORY.md)
> for the single source of truth on current state, and [`docs/TODO.md`](docs/TODO.md)
> for the sequential task list.

## Why

LLM context windows are finite and expensive. Naively stuffing every retrieved
chunk into a prompt wastes tokens, degrades answer quality, and gives you no
visibility into *why* the model saw what it saw. ContextPilot turns context
assembly into an explicit, testable, auditable pipeline.

## The idea

```python
from contextpilot import ContextPilot, ContextBlock

pilot = ContextPilot(max_prompt_tokens=4096, strategy="balanced")

result = pilot.optimize(
    query="Find documents related to job search and summarize them",
    blocks=candidate_blocks,   # list[ContextBlock]
)

print(result.final_prompt)   # ready to send to any LLM
print(result.audit)          # what was included / compressed / dropped, and why
```

ContextPilot does **not** call an LLM, talk to a vector store, or know about any
particular application. It is a pure optimization core. Applications (such as the
**Beacon** desktop app) provide the candidate blocks and consume `final_prompt`.

## Pipeline (V1)

1. **Deduplicate** candidate blocks (exact).
2. **Score** blocks (keyword + provided semantic score → hybrid).
3. **Rank** by utility.
4. **Compress** high-value large blocks if needed (extractive).
5. **Select** blocks under the token budget (greedy, required-first).
6. **Build** the final prompt (cacheable/stable sections first).
7. **Audit** — produce a per-block decision report.

## Documentation

| Doc | Purpose |
|-----|---------|
| [`docs/PROJECT_MEMORY.md`](docs/PROJECT_MEMORY.md) | Living source of truth — updated every work session |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module layout and data flow |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | V1 / V2 phases |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Architecture Decision Records |
| [`docs/TODO.md`](docs/TODO.md) | Sequential, trackable task list |
| [`docs/LIBRARY_API.md`](docs/LIBRARY_API.md) | Public API reference |
| [`docs/ERROR_LOG.md`](docs/ERROR_LOG.md) | Errors, root causes, fixes |

## Development

This project uses [**uv**](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
uv sync            # create venv + install deps
uv run pytest      # run the test suite
```

## License

[MIT](LICENSE)
