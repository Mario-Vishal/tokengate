"""Neural reranking stage (CP-018).

A cross-encoder scores (query, chunk) pairs *jointly* — more accurate than comparing
separately-encoded embeddings — to judge whether a chunk actually helps answer the query.
The typical flow: the app retrieves a broad top-N from LanceDB, this stage reranks them,
and only the top ``top_n`` proceed to dedup / compression / MMR / budgeting. Reranking is
query-aware but non-generative (no generation token cost).
"""

from __future__ import annotations

from contextpilot.core.block import ContextBlock
from contextpilot.models.base import Reranker
from contextpilot.utils.logging import get_logger, log_event

_logger = get_logger("ranking.reranker")


def _rerank_key(block: ContextBlock) -> float:
    return block.rerank_score if block.rerank_score is not None else float("-inf")


def rerank_blocks(
    query: str,
    blocks: list[ContextBlock],
    reranker: Reranker,
    *,
    top_n: int | None = 15,
) -> list[ContextBlock]:
    """Score blocks with the cross-encoder, sort by ``rerank_score`` desc, keep ``top_n``.

    Sets ``rerank_score`` (raw, unbounded) on every input block. ``top_n=None`` keeps all.
    Stable for ties. Returns a new list; input order is not mutated beyond the scores.
    """
    if not blocks:
        return []

    scores = reranker.rerank(query, [b.content for b in blocks])
    for block, score in zip(blocks, scores, strict=True):
        block.rerank_score = float(score)

    ranked = sorted(blocks, key=_rerank_key, reverse=True)
    kept = ranked if top_n is None else ranked[:top_n]
    log_event(_logger, "reranking_completed",
              candidates=len(blocks), kept=len(kept), top_n=top_n)
    return kept


__all__ = ["rerank_blocks"]
