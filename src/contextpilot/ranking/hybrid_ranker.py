"""Hybrid ranking: combine semantic + keyword into final_score (CP-008).

Pipeline for a candidate set:
1. compute raw keyword scores (``keyword_ranker``),
2. normalize them to ``[0, 1]`` (``score_normalizer``) and store on each block,
3. blend with the caller-provided ``semantic_score`` via configured weights to produce
   ``final_score``,
4. return the blocks ranked by ``final_score`` (descending, stable).

Blocks with no ``semantic_score`` still rank on their keyword signal alone.
"""

from __future__ import annotations

from contextpilot.core.block import ContextBlock
from contextpilot.ranking.keyword_ranker import keyword_raw_scores
from contextpilot.ranking.score_normalizer import normalize_min_max


def combine_scores(
    semantic: float, keyword: float, semantic_weight: float, keyword_weight: float
) -> float:
    """Weighted average of two ``[0, 1]`` scores; result stays in ``[0, 1]``."""
    total = semantic_weight + keyword_weight
    if total <= 0:
        return 0.0
    return (semantic_weight * semantic + keyword_weight * keyword) / total


def rank_blocks(
    query: str,
    blocks: list[ContextBlock],
    *,
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> list[ContextBlock]:
    """Score and rank ``blocks`` for ``query`` (mutates ``keyword_score`` /
    ``final_score`` in place) and return them sorted by ``final_score`` descending.

    Sorting is stable, so blocks tied on score keep their input order.
    """
    if not blocks:
        return []

    normalized_keyword = normalize_min_max(keyword_raw_scores(query, blocks))
    for block, kw in zip(blocks, normalized_keyword, strict=True):
        block.keyword_score = kw
        semantic = block.semantic_score if block.semantic_score is not None else 0.0
        block.final_score = combine_scores(
            semantic, kw, semantic_weight, keyword_weight
        )

    return sorted(blocks, key=lambda b: b.final_score or 0.0, reverse=True)


__all__ = ["combine_scores", "rank_blocks"]
