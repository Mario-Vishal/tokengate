"""Greedy, required-first token budgeter (CP-010).

Decides each block's fate against the token budget:

- **required** blocks are always included and their tokens are *reserved up front*, so
  optional selection accounts for them. If required blocks alone exceed the budget they
  are still kept (the caller asked explicitly); the overage is reported honestly in the
  audit rather than silently dropped (ADR-009).
- **optional** blocks are taken in ranked order while they fit the remaining space; one
  that doesn't fit is first *compressed into the remaining space* (if compression is
  enabled and the block is compressible) and kept if that makes it fit, else dropped.

Greedy continues after a drop — a smaller, lower-ranked block may still fit. Every block
yields a :class:`BlockDecision` so the audit can explain the outcome. ``prompt_blocks``
preserves the input (ranked) order with compressed copies substituted in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contextpilot.budgeting.token_counter import TokenCounter, resolve_counter
from contextpilot.compression.extractive import compress_block
from contextpilot.core.block import ContextBlock
from contextpilot.core.result import (
    DECISION_COMPRESSED,
    DECISION_DROPPED,
    DECISION_INCLUDED,
    BlockDecision,
)
from contextpilot.models.base import EmbeddingModel
from contextpilot.utils.errors import BudgetError


@dataclass
class BudgetOutcome:
    """Result of budgeting: the kept blocks (by fate) and the per-block decisions.

    Both ``included`` (kept as-is) and ``compressed`` (kept in shortened form) end up in
    the final prompt; ``dropped`` do not. ``prompt_blocks`` is those in-prompt blocks in
    ranked order (compressed copies substituted in place) — use it to build the prompt.
    ``used_tokens`` is the sum across both in-prompt groups.
    """

    included: list[ContextBlock] = field(default_factory=list)
    compressed: list[ContextBlock] = field(default_factory=list)
    dropped: list[ContextBlock] = field(default_factory=list)
    prompt_blocks: list[ContextBlock] = field(default_factory=list)
    decisions: list[BlockDecision] = field(default_factory=list)
    used_tokens: int = 0


def budget_blocks(
    ranked_blocks: list[ContextBlock],
    *,
    query: str,
    budget_tokens: int,
    counter: TokenCounter | None = None,
    enable_compression: bool = True,
    embedding_model: EmbeddingModel | None = None,
) -> BudgetOutcome:
    """Select blocks under ``budget_tokens`` (required reserved first, compress-to-fit).

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
    result: dict[int, tuple[str, ContextBlock | None, BlockDecision]] = {}

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
                          tokens, tokens, block.final_score),
        )

    # (2) Fit optional blocks into the remaining space, in ranked order.
    for block in ranked_blocks:
        if block.required:
            continue
        tokens = block.ensure_token_count(counter)
        score = block.final_score
        remaining = budget_tokens - outcome.used_tokens

        if tokens <= remaining:
            outcome.used_tokens += tokens
            result[id(block)] = (
                DECISION_INCLUDED,
                block,
                BlockDecision(block.block_id, DECISION_INCLUDED, "fits within budget",
                              tokens, tokens, score),
            )
            continue

        if (
            enable_compression
            and embedding_model is not None
            and block.compressible
            and remaining > 0
        ):
            compressed = compress_block(
                block, query, embedding_model, target_tokens=remaining, counter=counter
            )
            new_tokens = compressed.ensure_token_count(counter)
            if compressed is not block and new_tokens <= remaining < tokens:
                outcome.used_tokens += new_tokens
                result[id(block)] = (
                    DECISION_COMPRESSED,
                    compressed,
                    BlockDecision(block.block_id, DECISION_COMPRESSED,
                                  "compressed to fit remaining budget",
                                  tokens, new_tokens, score),
                )
                continue

        result[id(block)] = (
            DECISION_DROPPED,
            None,
            BlockDecision(block.block_id, DECISION_DROPPED,
                          "exceeds remaining token budget", tokens, 0, score),
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
