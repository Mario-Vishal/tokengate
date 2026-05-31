# ContextPilot — ARCHITECTURE

Last updated: **2026-05-30**

## Design principles

1. **App-agnostic.** The core knows nothing about folders, vector stores, LLMs,
   or UIs. It transforms blocks → optimized prompt + audit.
2. **Pure.** No network, no disk I/O for user data, no hidden global state. Inputs
   in, results out. This makes the whole library trivially testable.
3. **Caller provides semantics.** Embeddings/semantic scores are computed by the
   application and attached to each `ContextBlock`. The library combines them with
   its own keyword signal.
4. **Everything is auditable.** Every `optimize()` produces a serializable audit
   that explains each block's fate and the token math.
5. **Small, composable modules.** Each pipeline stage is independent and unit-tested.

## Package layout

```
src/contextpilot/
  __init__.py            # public exports: ContextPilot, ContextBlock, results, errors
  core/
    block.py             # ContextBlock model + validation
    result.py            # OptimizationResult, AuditReport, BlockDecision
    config.py            # OptimizerConfig, strategy presets
    optimizer.py         # ContextPilot orchestrator (public entrypoint)
  deduplication/
    exact.py             # exact-content dedup (hash-based)            [V1]
    near_duplicate.py    # semantic / near-dup dedup                   [V2]
  ranking/
    keyword_ranker.py    # keyword-overlap scoring vs query           [V1]
    score_normalizer.py  # normalize heterogeneous scores to [0,1]    [V1]
    hybrid_ranker.py     # combine semantic + keyword -> final_score  [V1]
  compression/
    extractive.py        # sentence-selection extractive compression  [V1]
    structured_schema.py # schema for LLM-based compression           [V2]
  budgeting/
    token_counter.py     # TokenCounter protocol + default heuristic  [V1]
    budgeter.py          # greedy budgeted selection (required-first) [V1]
  prompts/
    prompt_builder.py    # assemble final prompt; stable sections 1st [V1]
  audit/
    audit_report.py      # build AuditReport from pipeline state       [V1]
  utils/
    hashing.py           # stable content hashing
    logging.py           # structured logger factory
    errors.py            # library exception hierarchy
```

## Data flow

```
                 query: str
                 blocks: list[ContextBlock]
                        |
                        v
        +-------------------------------+
        |        ContextPilot           |
        |        .optimize()            |
        +-------------------------------+
                        |
   (1) deduplication.exact  -> drop exact duplicates
                        |
   (2) ranking.keyword_ranker -> keyword_score per block
   (3) ranking.score_normalizer -> normalize semantic + keyword
   (4) ranking.hybrid_ranker  -> final_score per block
                        |
   (5) sort by final_score (required blocks pinned)
                        |
   (6) compression.extractive -> shrink high-value large blocks (if needed)
                        |
   (7) budgeting.budgeter (uses token_counter) -> select under max_prompt_tokens
                        |
   (8) prompts.prompt_builder -> final_prompt (stable/cacheable sections first)
                        |
   (9) audit.audit_report -> AuditReport with per-block BlockDecision
                        v
              OptimizationResult
```

## Core models

### `ContextBlock`
| field | type | notes |
|-------|------|-------|
| `block_id` | str | stable id; auto-derived from content hash if omitted |
| `content` | str | the text |
| `block_type` | str | e.g. `document`, `chunk`, `system`, `instruction` |
| `source_id` | str \| None | provenance (file/source id) |
| `token_count` | int \| None | lazily computed via TokenCounter if None |
| `metadata` | dict | free-form (page, path, etc.) |
| `semantic_score` | float \| None | provided by caller (retrieval similarity) |
| `keyword_score` | float \| None | computed by library |
| `final_score` | float \| None | computed by library |
| `required` | bool | always included, never dropped |
| `cacheable` | bool | stable content → ordered first in prompt |
| `compressible` | bool | may be extractively compressed |

### `OptimizationResult`
`query`, `final_prompt`, `included_blocks`, `compressed_blocks`,
`dropped_blocks`, `audit`.

### `AuditReport`
`total_candidate_blocks`, `total_candidate_tokens`, `final_prompt_tokens`,
`tokens_saved`, `tokens_saved_percent`, `included_count`, `compressed_count`,
`dropped_count`, `decisions: list[BlockDecision]`.

### `BlockDecision`
`block_id`, `decision` (`included|compressed|dropped`), `reason`,
`original_tokens`, `final_tokens`, `score`.

## Token counting

`TokenCounter` is a Protocol with `count(text: str) -> int`. The default
`HeuristicTokenCounter` is dependency-free (approximates ~4 chars/token with word
awareness). An optional `TiktokenCounter` (extra `contextpilot[tiktoken]`) gives
exact counts. The optimizer accepts any `TokenCounter`, so applications can inject
the tokenizer that matches their target model.

## Configuration & strategies

`OptimizerConfig` holds `max_prompt_tokens`, `strategy`, dedup/compression toggles,
and scoring weights. V1 ships a single functional `balanced` strategy; the config
shape reserves room for `speed`, `quality`, `max_compression` (V2) without API
changes.

## Error model

All raised errors derive from `ContextPilotError`. V1 notable subclasses:
`InvalidBlockError`, `BudgetError`, `OptimizationError`. Errors are explicit and
never swallow data needed for the audit.

## Extension points (forward-looking)

- **Rankers**: `hybrid_ranker` weights are configurable; a reranker adapter slots
  in at stage (4) in V2.
- **Compression**: `extractive` is the V1 implementation behind a common interface;
  `structured_schema` (LLM compression) is an alternate implementation for V2.
- **Dedup**: `near_duplicate` adds a stage between (1) and (2) in V2.
- **Token counters**: any `TokenCounter` implementation is accepted.
