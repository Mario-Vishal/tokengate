"""Hybrid ranking: combine multiple signals into final_score (CP-008, expanded CP-017).

Signals per block: semantic (caller- or model-provided ``semantic_score``), keyword
(lexical overlap, set-normalized), recency, source priority, and token efficiency. Each
has a configured weight. For a given block we take the **weighted average over only the
signals that are available** (renormalizing the weights), so a missing signal neither
helps nor hurts. Blocks are returned ranked by ``final_score`` (descending, stable).
"""

from __future__ import annotations

from contextpilot.core.block import ContextBlock
from contextpilot.ranking.keyword_ranker import keyword_raw_scores
from contextpilot.ranking.score_normalizer import normalize_min_max
from contextpilot.ranking.signals import (
    recency_scores,
    source_priority_scores,
    token_efficiency_scores,
)


def weighted_average(pairs: list[tuple[float, float]]) -> float:
    """Weighted average of ``(weight, value)`` pairs; 0.0 if total weight is 0."""
    total = sum(w for w, _ in pairs)
    if total <= 0:
        return 0.0
    return sum(w * v for w, v in pairs) / total


def combine_scores(
    semantic: float, keyword: float, semantic_weight: float, keyword_weight: float
) -> float:
    """Two-signal weighted average (kept for compatibility / direct use)."""
    return weighted_average([(semantic_weight, semantic), (keyword_weight, keyword)])


def rank_blocks(
    query: str,
    blocks: list[ContextBlock],
    *,
    semantic_weight: float = 0.45,
    keyword_weight: float = 0.25,
    recency_weight: float = 0.10,
    source_priority_weight: float = 0.10,
    token_efficiency_weight: float = 0.10,
    source_priorities: dict[str, float] | None = None,
) -> list[ContextBlock]:
    """Score and rank ``blocks`` for ``query`` (mutates ``keyword_score``/``final_score``)
    and return them sorted by ``final_score`` descending (stable on ties).

    ``semantic_score`` is read as-is (set upstream from embeddings or by the caller);
    keyword/recency/token-efficiency are computed set-relative here.
    """
    if not blocks:
        return []

    keyword_norm = normalize_min_max(keyword_raw_scores(query, blocks))
    recency = recency_scores(blocks)
    token_eff = token_efficiency_scores(blocks)
    source_prio = source_priority_scores(blocks, source_priorities or {})

    for i, block in enumerate(blocks):
        block.keyword_score = keyword_norm[i]

        pairs: list[tuple[float, float]] = [(keyword_weight, keyword_norm[i])]
        if block.semantic_score is not None:
            pairs.append((semantic_weight, block.semantic_score))
        rec, prio, teff = recency[i], source_prio[i], token_eff[i]
        if rec is not None:
            pairs.append((recency_weight, rec))
        if prio is not None:
            pairs.append((source_priority_weight, prio))
        if teff is not None:
            pairs.append((token_efficiency_weight, teff))

        block.final_score = weighted_average(pairs)

    return sorted(blocks, key=lambda b: b.final_score or 0.0, reverse=True)


__all__ = ["weighted_average", "combine_scores", "rank_blocks"]
