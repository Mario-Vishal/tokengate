# ContextPilot — LIBRARY API (reference)

> **Status: implemented.** V1 surface below is current. **V2 (neural engine)** additions
> are summarized in the box right under here; full design in `V2_DESIGN.md`.

> ### V2 neural engine (CP-014 … CP-023)
> ContextPilot is now a neural engine (ADR-010). Key API additions:
> - `ContextPilot(..., embedding_model=None, reranker=None)` — required models with lazy
>   **BGE-M3** / **bge-reranker-v2-m3** defaults; inject your own (or fakes) to override.
>   Construction is cheap; weights download on first `optimize()`.
> - `ContextBlock` gains `vector` (reused if present, else computed) and `rerank_score`.
> - `OptimizerConfig` gains multi-signal weights (`semantic/keyword/recency/
>   source_priority/token_efficiency_weight`), `source_priorities`, `rerank_top_n`,
>   `enable_semantic_dedup` + `semantic_dedup_threshold`, `enable_mmr` + `mmr_lambda`.
> - Strategy presets are real: `speed` / `balanced` / `quality` / `max_compression`.
> - `AuditReport.models_used` and per-`BlockDecision` `rerank_score`.
> - Pipeline: exact dedup → embed → semantic+hybrid → rerank → semantic dedup → MMR →
>   value/token budget → prompt → audit. No generative LLM.
> - **Compression is relevance-driven (ADR-019), not target-driven:** it drops boilerplate
>   (keeps sentences ≥ `compression_keep_ratio × best`); there is no token target. If a
>   relevance-pruned block still doesn't fit the budget it is dropped, not truncated.

> **Note on `tokens_saved`.** Savings are `total_candidate_tokens - final_prompt_tokens`.
> If the candidate context comfortably fits the budget, nothing is dropped and the
> small prompt scaffolding (`Context:`, `[1]`, `Question:`) can make `tokens_saved`
> slightly **negative** — that is honest, not a bug. Savings go positive once the
> candidate context exceeds the budget and blocks are compressed/dropped.

> **Note on `tokens_saved`.** Savings are `total_candidate_tokens - final_prompt_tokens`.
> If the candidate context comfortably fits the budget, nothing is dropped and the
> small prompt scaffolding (`Context:`, `[1]`, `Question:`) can make `tokens_saved`
> slightly **negative** — that is honest, not a bug. Savings go positive once the
> candidate context exceeds the budget and blocks are compressed/dropped.

## Public exports

```python
from contextpilot import (
    ContextPilot,
    ContextBlock,
    OptimizationResult,
    AuditReport,
    BlockDecision,
    OptimizerConfig,
    # errors
    ContextPilotError,
    InvalidBlockError,
    BudgetError,
    OptimizationError,
    # token counting
    TokenCounter,
    HeuristicTokenCounter,
)
```

## `ContextBlock`

```python
ContextBlock(
    content: str,
    *,
    block_id: str | None = None,        # derived from content hash if None
    block_type: str = "document",
    source_id: str | None = None,
    token_count: int | None = None,     # computed lazily via TokenCounter
    metadata: dict | None = None,
    semantic_score: float | None = None,  # caller-provided, [0,1] recommended
    keyword_score: float | None = None,   # set by library
    final_score: float | None = None,     # set by library
    required: bool = False,
    cacheable: bool = False,
    compressible: bool = True,
)
```
- Raises `InvalidBlockError` on empty content or out-of-range scores.
- `to_dict()` / `from_dict()` for serialization.

## `OptimizerConfig`

```python
OptimizerConfig(
    max_prompt_tokens: int = 4096,
    strategy: str = "balanced",          # V1: "balanced" only
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4,
    enable_dedup: bool = True,
    enable_compression: bool = True,
    safety_margin: float = 0.05,         # reserve % of budget
    token_counter: TokenCounter | None = None,  # default: HeuristicTokenCounter
)
```

## `ContextPilot`

```python
pilot = ContextPilot(
    max_prompt_tokens: int = 4096,
    strategy: str = "balanced",
    config: OptimizerConfig | None = None,   # overrides the convenience args
    token_counter: TokenCounter | None = None,
)

result: OptimizationResult = pilot.optimize(
    query: str,
    blocks: list[ContextBlock],
)
```

Pipeline executed by `optimize()`: dedup → score/rank → compress (if needed) →
budget → build prompt → audit. Pure and deterministic for a given input + config.

## `OptimizationResult`

```python
result.query: str
result.final_prompt: str
result.included_blocks: list[ContextBlock]
result.compressed_blocks: list[ContextBlock]
result.dropped_blocks: list[ContextBlock]
result.audit: AuditReport
result.to_dict() -> dict
```

## `AuditReport`

```python
audit.total_candidate_blocks: int
audit.total_candidate_tokens: int
audit.final_prompt_tokens: int
audit.tokens_saved: int
audit.tokens_saved_percent: float
audit.included_count: int
audit.compressed_count: int
audit.dropped_count: int
audit.decisions: list[BlockDecision]
audit.to_dict() -> dict
```

## `BlockDecision`

```python
decision.block_id: str
decision.decision: str           # "included" | "compressed" | "dropped"
decision.reason: str
decision.original_tokens: int
decision.final_tokens: int
decision.score: float | None
```

## `TokenCounter` (protocol)

```python
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

HeuristicTokenCounter()          # dependency-free default
# optional, requires: pip install "contextpilot[tiktoken]"
TiktokenCounter(encoding="cl100k_base")
```

## End-to-end example

```python
from contextpilot import ContextPilot, ContextBlock

blocks = [
    ContextBlock(content="System: answer concisely.", block_type="system",
                 required=True, cacheable=True, compressible=False),
    ContextBlock(content="Resume mentions Python, FastAPI, job search 2026...",
                 source_id="file:resume.pdf", semantic_score=0.82),
    ContextBlock(content="Unrelated grocery list...",
                 source_id="file:notes.txt", semantic_score=0.10),
]

pilot = ContextPilot(max_prompt_tokens=2048, strategy="balanced")
result = pilot.optimize(
    query="Find documents related to job search and summarize them",
    blocks=blocks,
)

print(result.final_prompt)
print(f"saved {result.audit.tokens_saved} tokens "
      f"({result.audit.tokens_saved_percent:.1f}%)")
for d in result.audit.decisions:
    print(d.block_id, d.decision, "-", d.reason)
```
