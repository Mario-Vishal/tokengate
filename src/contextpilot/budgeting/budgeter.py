"""Greedy, required-first token budgeter (CP-010).

Walks blocks in ranked order and decides each one's fate against the token budget:

- **required** blocks are always included, even if they push past the budget (the
  caller asked for them explicitly; we keep them and report the overage honestly).
- **optional** blocks are included while they fit; one that doesn't fit is first
  *compressed to the remaining space* (if compression is enabled and the block is
  compressible) and included if that makes it fit; otherwise it is dropped.

Greedy continues after a drop — a smaller, lower-ranked block may still fit. Every
block yields a :class:`BlockDecision` so the audit can explain the outcome.
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
from contextpilot.utils.errors import BudgetError


@dataclass
class BudgetOutcome:
    """Result of budgeting: the kept blocks (by fate) and the per-block decisions.

    Both ``included`` (kept as-is) and ``compressed`` (kept in shortened form) end up in
    the final prompt; ``dropped`` do not. ``used_tokens`` is the sum across both
    in-prompt groups.
    """

    included: list[ContextBlock] = field(default_factory=list)
    compressed: list[ContextBlock] = field(default_factory=list)
    dropped: list[ContextBlock] = field(default_factory=list)
    decisions: list[BlockDecision] = field(default_factory=list)
    used_tokens: int = 0


def budget_blocks(
    ranked_blocks: list[ContextBlock],
    *,
    query: str,
    budget_tokens: int,
    counter: TokenCounter | None = None,
    enable_compression: bool = True,
) -> BudgetOutcome:
    """Select blocks under ``budget_tokens`` (required-first, compress-to-fit)."""
    if budget_tokens <= 0:
        raise BudgetError(f"budget_tokens must be positive, got {budget_tokens}")

    counter = resolve_counter(counter)
    outcome = BudgetOutcome()

    for block in ranked_blocks:
        tokens = block.ensure_token_count(counter)
        score = block.final_score

        if block.required:
            outcome.included.append(block)
            outcome.used_tokens += tokens
            outcome.decisions.append(
                BlockDecision(block.block_id, DECISION_INCLUDED, "required block",
                              tokens, tokens, score)
            )
            continue

        remaining = budget_tokens - outcome.used_tokens

        if tokens <= remaining:
            outcome.included.append(block)
            outcome.used_tokens += tokens
            outcome.decisions.append(
                BlockDecision(block.block_id, DECISION_INCLUDED, "fits within budget",
                              tokens, tokens, score)
            )
            continue

        # Doesn't fit: try compressing into the remaining space.
        if enable_compression and block.compressible and remaining > 0:
            compressed = compress_block(
                block, query, target_tokens=remaining, counter=counter
            )
            new_tokens = compressed.ensure_token_count(counter)
            if compressed is not block and new_tokens <= remaining < tokens:
                outcome.compressed.append(compressed)
                outcome.used_tokens += new_tokens
                outcome.decisions.append(
                    BlockDecision(
                        block.block_id, DECISION_COMPRESSED,
                        "compressed to fit remaining budget", tokens, new_tokens, score,
                    )
                )
                continue

        outcome.dropped.append(block)
        outcome.decisions.append(
            BlockDecision(block.block_id, DECISION_DROPPED,
                          "exceeds remaining token budget", tokens, 0, score)
        )

    return outcome


__all__ = ["BudgetOutcome", "budget_blocks"]
