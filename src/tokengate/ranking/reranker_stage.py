"""Neural reranking stage (CP-018).

A cross-encoder scores (query, chunk) pairs *jointly* — more accurate than comparing
separately-encoded embeddings — to judge whether a chunk actually helps answer the query.
The typical flow: the app retrieves a broad top-N from LanceDB, this stage reranks them,
and only the top ``top_n`` proceed to dedup / compression / MMR / budgeting. Reranking is
query-aware but non-generative (no generation token cost).
"""

from __future__ import annotations

from tokengate.core.block import TokenBlock
from tokengate.models.base import Reranker
from tokengate.ranking.score_normalizer import normalize_min_max
from tokengate.utils.logging import get_logger, log_event

_logger = get_logger("ranking.reranker")


def _rerank_key(block: TokenBlock) -> float:
    return block.rerank_score if block.rerank_score is not None else float("-inf")


def rerank_blocks(
    query: str,
    blocks: list[TokenBlock],
    reranker: Reranker,
    *,
    top_n: int | None = 15,
) -> list[TokenBlock]:
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


def apply_rerank_relevance(
    blocks: list[TokenBlock], *, rerank_weight: float = 0.7,
    normalization_ceiling: float = 0.5,
) -> None:
    """Blend the cross-encoder score into ``final_score`` so the reranker drives
    downstream selection (MMR + budgeting), not just the top-N cutoff (ADR-018).

    When the best raw rerank score exceeds ``normalization_ceiling``, min-max
    normalization is used so the full [0, 1] range is exploited. When all scores are
    below the ceiling (every block is weakly relevant), absolute scaling is used instead
    — keeping near-zero scores near-zero so the relevance floor can drop them, rather
    than inflating the "best of the irrelevant" to look relevant.
    """
    if not blocks:
        return
    raw = [b.rerank_score if b.rerank_score is not None else 0.0 for b in blocks]
    max_raw = max(raw) if raw else 0.0
    if max_raw >= normalization_ceiling:
        normalized = normalize_min_max(raw)
    else:
        normalized = [r / normalization_ceiling for r in raw]
    for block, norm in zip(blocks, normalized, strict=True):
        prior = block.final_score if block.final_score is not None else 0.0
        block.final_score = rerank_weight * norm + (1.0 - rerank_weight) * prior


__all__ = ["rerank_blocks", "apply_rerank_relevance"]
