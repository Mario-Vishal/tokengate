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
from tokengate.compression.base import Compressor
from tokengate.compression.extractive import ExtractiveCompressor
from tokengate.core.block import TokenBlock
from tokengate.core.result import (
    DECISION_COMPRESSED,
    DECISION_DROPPED,
    DECISION_INCLUDED,
    BlockDecision,
    Decision,
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
    min_rerank_score: float = 0.0,
    compress_always: bool = False,
    compressor: Compressor | None = None,
    compression_keep_ratio: float = 0.5,
    compression_sentence_dedup_threshold: float = 0.95,
) -> BudgetOutcome:
    """Select blocks under ``budget_tokens`` (required reserved first, compress-to-fit).

    ``relevance_floor`` drops optional blocks whose ``final_score`` is below it *before*
    budgeting (ADR-018), so cheap low-relevance noise can't crowd out relevant content
    and leave no room to compress high-value large blocks. 0.0 disables it.

    Compression needs ``embedding_model`` (extractive compression scores sentences with
    embeddings). Without one, oversized optional blocks are dropped rather than compressed;
    the engine (optimizer) always supplies a model.

    ``compress_always`` makes compression the primary token-saving lever (CP-029): every
    compressible optional block is relevance-compressed up front — boilerplate and
    off-query sentences dropped — *before* the fit check, so kept relevant content costs
    minimal tokens. When ``False`` (the legacy default) compression is compress-to-fit: a
    block is only compressed if it overflows the remaining budget, so blocks that already
    fit go in verbatim. Either way compression is relevance-driven with no token target
    (ADR-019); a block that still doesn't fit after compression is dropped, never truncated.
    """
    if budget_tokens <= 0:
        raise BudgetError(f"budget_tokens must be positive, got {budget_tokens}")

    counter = resolve_counter(counter)
    outcome = BudgetOutcome()

    # Default to the embedding extractive compressor when the caller didn't inject one
    # (keeps the legacy signature working; the optimizer injects the configured backend).
    if compressor is None and enable_compression and embedding_model is not None:
        compressor = ExtractiveCompressor(
            embedding_model, counter=counter,
            sentence_dedup_threshold=compression_sentence_dedup_threshold,
        )

    # Per-block result, keyed by object identity (robust to duplicate block_ids), so we
    # can emit prompt_blocks / decisions in ranked order after fitting.
    # Maps id(block) -> (fate, block_to_place_or_None, decision).
    result: dict[int, tuple[Decision, TokenBlock | None, BlockDecision]] = {}

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

        # Hard rerank floor: cross-encoder says this block is not relevant — drop unconditionally.
        raw_rerank = block.rerank_score if block.rerank_score is not None else 0.0
        if min_rerank_score > 0.0 and raw_rerank < min_rerank_score:
            result[id(block)] = (
                DECISION_DROPPED,
                None,
                BlockDecision(block.block_id, DECISION_DROPPED,
                              f"below min rerank score ({min_rerank_score:.3f})",
                              tokens, 0, score, rerank_score=block.rerank_score,
                              source_id=block.source_id,
                              content_preview=block.content[:300]),
            )
            continue

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

        # Relevance-driven compression (ADR-019): drop boilerplate / off-query sentences;
        # size is content-determined (no token target). With ``compress_always`` every
        # compressible block is squeezed up front (compression is the main token lever);
        # otherwise only an oversized block is compressed (compress-to-fit). A block that
        # still doesn't fit is dropped — relevant content is never truncated to a number.
        place = block
        fate = DECISION_INCLUDED
        reason = "fits within budget"
        if (
            enable_compression
            and compressor is not None
            and block.compressible
            and remaining > 0
            and (compress_always or tokens > remaining)
        ):
            compressed = compressor.compress_block(
                block, query, keep_ratio=compression_keep_ratio,
            )
            if compressed is not block:
                place = compressed
                fate = DECISION_COMPRESSED
                reason = "compressed (low-value text dropped)"

        place_tokens = place.ensure_token_count(counter)
        if place_tokens <= remaining:
            outcome.used_tokens += place_tokens
            result[id(block)] = (
                fate,
                place,
                BlockDecision(block.block_id, fate, reason,
                              tokens, place_tokens, score,
                              rerank_score=block.rerank_score,
                              source_id=block.source_id,
                              content_preview=place.content[:300]),
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

    # (2b) Fallback: if every optional block was filtered out, force-include the top-1
    # by raw rerank score so the LLM always has at least some grounding and can't
    # hallucinate from training data on an empty context.
    nothing_included = not any(
        fate == DECISION_INCLUDED or fate == DECISION_COMPRESSED
        for fate, _, _ in result.values()
    )
    if nothing_included and optional:
        best = max(optional, key=lambda b: b.rerank_score if b.rerank_score is not None else 0.0)
        tokens = best.ensure_token_count(counter)
        outcome.used_tokens += tokens
        result[id(best)] = (
            DECISION_INCLUDED,
            best,
            BlockDecision(best.block_id, DECISION_INCLUDED,
                          "fallback: top-1 by rerank score (all others filtered)",
                          tokens, tokens, best.final_score,
                          rerank_score=best.rerank_score,
                          source_id=best.source_id,
                          content_preview=best.content[:300]),
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
