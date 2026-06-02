"""Greedy, required-first token budgeter (CP-010).

Decides each block's fate against the token budget:

- **required** blocks are always included and their tokens are *reserved up front*, so
  optional selection accounts for them. If required blocks alone exceed the budget they
  are still kept (the caller asked explicitly); the overage is reported honestly in the
  audit rather than silently dropped (ADR-009).
- **optional** blocks are taken by **value-per-token** (``final_score / tokens``), not
  raw rank, so a small high-value block beats a large slightly-more-relevant one
  ("best set under budget", ADR-016). One that doesn't fit is **relevance-compressed**
  (boilerplate dropped, no token target — ADR-019); it's kept only if the pruned block
  fits, otherwise the whole block is dropped (relevant content is never truncated to fit).
- Below ``relevance_floor`` optional blocks are dropped before budgeting.

Greedy continues after a drop — a smaller, denser block may still fit. Every block
yields a :class:`BlockDecision` so the audit can explain the outcome. ``prompt_blocks``
preserves the input (ranked/MMR) order with compressed copies substituted in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tokengate.budgeting.token_counter import TokenCounter, resolve_counter
from tokengate.compression.extractive import compress_block
from tokengate.core.block import TokenBlock
from tokengate.core.result import (
    DECISION_COMPRESSED,
    DECISION_DROPPED,
    DECISION_INCLUDED,
    BlockDecision,
)
from tokengate.models.base import EmbeddingModel
from tokengate.utils.errors import BudgetError


def _value_density(block: TokenBlock, counter: TokenCounter) -> float:
    """Utility per token: ``final_score / token_count`` (the value-per-token metric)."""
    value = block.final_score if block.final_score is not None else 0.0
    return value / max(block.ensure_token_count(counter), 1)


@dataclass
class BudgetOutcome:
    """Result of budgeting: the kept blocks (by fate) and the per-block decisions.

    Both ``included`` (kept as-is) and ``compressed`` (kept in shortened form) end up in
    the final prompt; ``dropped`` do not. ``prompt_blocks`` is those in-prompt blocks in
    ranked order (compressed copies substituted in place) — use it to build the prompt.
    ``used_tokens`` is the sum across both in-prompt groups.
    """

    included: list[TokenBlock] = field(default_factory=list)
    compressed: list[TokenBlock] = field(default_factory=list)
    dropped: list[TokenBlock] = field(default_factory=list)
    prompt_blocks: list[TokenBlock] = field(default_factory=list)
    decisions: list[BlockDecision] = field(default_factory=list)
    used_tokens: int = 0


def budget_blocks(
    ranked_blocks: list[TokenBlock],
    *,
    query: str,
    budget_tokens: int,
    counter: TokenCounter | None = None,
    enable_compression: bool = True,
    embedding_model: EmbeddingModel | None = None,
    relevance_floor: float = 0.0,
    compression_keep_ratio: float = 0.5,
    compression_sentence_dedup_threshold: float = 0.95,
) -> BudgetOutcome:
    """Select blocks under ``budget_tokens`` (required reserved first, compress-to-fit).

    ``relevance_floor`` drops optional blocks whose ``final_score`` is below it *before*
    budgeting (ADR-018), so cheap low-relevance noise can't crowd out relevant content
    and leave no room to compress high-value large blocks. 0.0 disables it.

    Compress-to-fit needs ``embedding_model`` (extractive compression scores sentences
    with embeddings). Without one, oversized optional blocks are dropped rather than
    compressed; the engine (optimizer) always supplies a model.
    """
    if budget_tokens <= 0:
        raise BudgetError(f"budget_tokens must be positive, got {budget_tokens}")

    counter = resolve_counter(counter)
    outcome = BudgetOutcome()

    # Per-block result, keyed by object identity (robust to duplicate block_ids), so we
    # can emit prompt_blocks / decisions in ranked order after fitting.
    # Maps id(block) -> (fate, block_to_place_or_None, decision).
    result: dict[int, tuple[str, TokenBlock | None, BlockDecision]] = {}

    # (1) Reserve all required blocks first.
    for block in ranked_blocks:
        if not block.required:
            continue
        tokens = block.ensure_token_count(counter)
        outcome.used_tokens += tokens
        result[id(block)] = (
            DECISION_INCLUDED,
            block,
            BlockDecision(block.block_id, DECISION_INCLUDED, "required block",
                          tokens, tokens, block.final_score,
                          rerank_score=block.rerank_score,
                          source_id=block.source_id,
                          content_preview=block.content[:300]),
        )

    # (2) Fit optional blocks by value-per-token ("best set under budget", ADR-016):
    # process highest value-density first so a small high-value block beats a large
    # slightly-more-relevant one. (Prompt order is restored to input order in step 3.)
    optional = [b for b in ranked_blocks if not b.required]
    by_density = sorted(
        optional,
        key=lambda b: (_value_density(b, counter), b.final_score or 0.0),
        reverse=True,
    )
    for block in by_density:
        tokens = block.ensure_token_count(counter)
        score = block.final_score

        # Relevance floor: prune low-relevance blocks before they consume budget.
        if relevance_floor > 0.0 and (score if score is not None else 0.0) < relevance_floor:
            result[id(block)] = (
                DECISION_DROPPED,
                None,
                BlockDecision(block.block_id, DECISION_DROPPED,
                              f"below relevance floor ({relevance_floor:.2f})",
                              tokens, 0, score, rerank_score=block.rerank_score,
                              source_id=block.source_id,
                              content_preview=block.content[:300]),
            )
            continue

        remaining = budget_tokens - outcome.used_tokens

        if tokens <= remaining:
            outcome.used_tokens += tokens
            result[id(block)] = (
                DECISION_INCLUDED,
                block,
                BlockDecision(block.block_id, DECISION_INCLUDED, "fits within budget",
                              tokens, tokens, score, rerank_score=block.rerank_score,
                              source_id=block.source_id,
                              content_preview=block.content[:300]),
            )
            continue

        # Relevance-driven compression (ADR-019): drop boilerplate; size is
        # content-determined (no token target). Keep only if the pruned block fits;
        # otherwise drop the whole block — never truncate relevant content to a number.
        if (
            enable_compression
            and embedding_model is not None
            and block.compressible
            and remaining > 0
        ):
            compressed = compress_block(
                block, query, embedding_model, counter=counter,
                keep_ratio=compression_keep_ratio,
                sentence_dedup_threshold=compression_sentence_dedup_threshold,
            )
            new_tokens = compressed.ensure_token_count(counter)
            if compressed is not block and new_tokens <= remaining:
                outcome.used_tokens += new_tokens
                result[id(block)] = (
                    DECISION_COMPRESSED,
                    compressed,
                    BlockDecision(block.block_id, DECISION_COMPRESSED,
                                  "compressed (boilerplate dropped) to fit budget",
                                  tokens, new_tokens, score,
                                  rerank_score=block.rerank_score,
                                  source_id=block.source_id,
                                  content_preview=compressed.content[:300]),
                )
                continue

        result[id(block)] = (
            DECISION_DROPPED,
            None,
            BlockDecision(block.block_id, DECISION_DROPPED,
                          "exceeds remaining token budget", tokens, 0, score,
                          rerank_score=block.rerank_score,
                          source_id=block.source_id,
                          content_preview=block.content[:300]),
        )

    # (3) Emit in ranked order so prompt_blocks/decisions are deterministic.
    for block in ranked_blocks:
        fate, placed, decision = result[id(block)]
        outcome.decisions.append(decision)
        if fate == DECISION_INCLUDED:
            outcome.included.append(block)
            outcome.prompt_blocks.append(block)
        elif fate == DECISION_COMPRESSED:
            assert placed is not None
            outcome.compressed.append(placed)
            outcome.prompt_blocks.append(placed)
        else:
            outcome.dropped.append(block)

    return outcome


__all__ = ["BudgetOutcome", "budget_blocks"]
