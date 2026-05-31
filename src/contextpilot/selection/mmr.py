"""Maximal Marginal Relevance (MMR) diversity selection (CP-021).

Plain ranking can pick many near-identical chunks. MMR re-selects greedily, each step
maximizing ``λ·relevance − (1−λ)·max_similarity(already_selected)`` — trading query
relevance against redundancy with the chosen set. The result is an information-rich,
non-repetitive ordering for budgeting to consume.

Relevance uses ``rerank_score`` when present (the strongest signal) else ``final_score``,
min-max normalized across the candidates. Similarity is cosine over block vectors;
blocks without vectors contribute 0 similarity (selected on relevance alone).
"""

from __future__ import annotations

import numpy as np

from contextpilot.core.block import ContextBlock
from contextpilot.ranking.score_normalizer import normalize_min_max


def _relevance(blocks: list[ContextBlock]) -> list[float]:
    raw: list[float] = []
    for b in blocks:
        if b.rerank_score is not None:
            raw.append(b.rerank_score)
        elif b.final_score is not None:
            raw.append(b.final_score)
        else:
            raw.append(0.0)
    return normalize_min_max(raw)


def _similarity(a: ContextBlock, b: ContextBlock) -> float:
    if a.vector is None or b.vector is None:
        return 0.0
    return float(np.dot(a.vector, b.vector))


def mmr_select(
    blocks: list[ContextBlock], *, lambda_: float = 0.5, top_k: int | None = None
) -> list[ContextBlock]:
    """Return blocks reordered by MMR. ``top_k=None`` returns all (reordered).

    ``lambda_=1.0`` is pure relevance order; ``lambda_=0.0`` is pure diversity.
    """
    if not blocks:
        return []

    relevance = _relevance(blocks)
    remaining = list(range(len(blocks)))
    selected: list[int] = []
    limit = len(blocks) if top_k is None else min(top_k, len(blocks))

    while remaining and len(selected) < limit:
        best_i = remaining[0]
        best_score = float("-inf")
        for i in remaining:
            redundancy = (
                max(_similarity(blocks[i], blocks[j]) for j in selected)
                if selected else 0.0
            )
            score = lambda_ * relevance[i] - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
        remaining.remove(best_i)

    return [blocks[i] for i in selected]


__all__ = ["mmr_select"]
