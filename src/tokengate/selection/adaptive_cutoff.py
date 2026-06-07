"""Per-query adaptive relevance cutoff (CP-030).

Replaces *fixed* relevance thresholds (``relevance_floor``, ``min_rerank_score``) with a
cutoff read from the **shape** of this query's score distribution — the gap/knee-detection
idea from Cluster-based Adaptive Retrieval (CAR, arXiv:2511.14769). A focused query whose
top block dominates keeps few blocks; a broad/synthesis query whose blocks score similarly
keeps many. There is **no tuned threshold** — the cutoff is derived per query.

Method (on blocks sorted by score, descending):

    composite(k) = gap(k) / max_gap  +  k / n        for k = 1 .. n-1
    composite(n) = 1.0                               (keep everything; no trailing gap)

where ``gap(k) = score[k-1] - score[k]`` is the drop right after the k-th kept block. The
first term rewards cutting at a sharp **relevance cliff**; the second rewards retrieval
**depth** (so we don't cut after block 1 unless the cliff there is genuinely the sharpest).
We keep the ``k`` that maximizes ``composite``. If all scores are equal (no positive gap)
every block is equally relevant, so we keep them all.
"""

from __future__ import annotations

from tokengate.core.block import TokenBlock


def _block_score(block: TokenBlock) -> float:
    """Relevance signal for cutoff: blended ``final_score``, else raw rerank, else 0."""
    if block.final_score is not None:
        return block.final_score
    if block.rerank_score is not None:
        return block.rerank_score
    return 0.0


def adaptive_cutoff_count(scores: list[float], *, min_keep: int = 1) -> int:
    """How many of ``scores`` (sorted descending) to keep, by the relevance-cliff rule.

    Returns ``len(scores)`` when there is no informative cliff (≤2 items, or all equal).
    Never returns less than ``min_keep`` (clamped to the available count).
    """
    n = len(scores)
    if n <= max(min_keep, 2):
        return n

    gaps = [scores[k - 1] - scores[k] for k in range(1, n)]  # gap after keeping k blocks
    max_gap = max(gaps)
    if max_gap <= 0:  # non-increasing with no real drop → all equally relevant
        return n

    best_k, best_score = n, 1.0  # composite(n) = n/n = 1.0 (the "keep all" baseline)
    for k in range(1, n):
        composite = gaps[k - 1] / max_gap + k / n
        if composite > best_score:
            best_score, best_k = composite, k
    return min(n, max(min_keep, best_k))


def select_by_adaptive_cutoff(
    blocks: list[TokenBlock], *, min_keep: int = 1
) -> tuple[list[TokenBlock], list[TokenBlock]]:
    """Split ``blocks`` into ``(kept, dropped)`` at the per-query relevance cliff.

    Required blocks are always kept (the caller asked for them) and never counted toward the
    cliff — the cutoff is computed over the optional blocks only. The cutoff is found on
    score order, but both returned lists preserve the **input order** so downstream ordering
    (dedup / MMR / prompt assembly) is unchanged.
    """
    optional = [b for b in blocks if not b.required]
    if len(optional) <= max(min_keep, 2):
        return list(blocks), []

    ordered = sorted(optional, key=_block_score, reverse=True)
    k = adaptive_cutoff_count([_block_score(b) for b in ordered], min_keep=min_keep)
    keep_ids = {id(b) for b in ordered[:k]}
    kept = [b for b in blocks if b.required or id(b) in keep_ids]
    dropped = [b for b in optional if id(b) not in keep_ids]
    return kept, dropped


__all__ = ["adaptive_cutoff_count", "select_by_adaptive_cutoff"]
