"""Exact, hash-based deduplication (CP-007).

Retrieval often surfaces the same text twice (overlapping chunks, the same file from
two sources). Exact dedup collapses byte-for-byte identical ``content`` to a single
representative, keeping the most useful copy and reporting the rest as dropped so the
audit can explain it. Fuzzy/semantic dedup is a separate V2 concern.
"""

from __future__ import annotations

from tokengate.core.block import TokenBlock
from tokengate.utils.hashing import hash_content


def _effective_score(block: TokenBlock) -> float:
    """Best available score for choosing a representative among duplicates."""
    for score in (block.final_score, block.semantic_score, block.keyword_score):
        if score is not None:
            return score
    return 0.0


def _preferred(existing: TokenBlock, candidate: TokenBlock) -> TokenBlock:
    """Pick which of two identical-content blocks to keep.

    Priority: ``required`` wins; then higher effective score; ties keep ``existing``
    (the earlier occurrence) for deterministic, stable output.
    """
    if candidate.required and not existing.required:
        return candidate
    if existing.required and not candidate.required:
        return existing
    if _effective_score(candidate) > _effective_score(existing):
        return candidate
    return existing


def deduplicate_exact(
    blocks: list[TokenBlock],
) -> tuple[list[TokenBlock], list[TokenBlock]]:
    """Remove exact-content duplicates.

    Returns ``(kept, dropped)``. ``kept`` preserves the order of first occurrence; for
    each duplicate group a single representative is retained (see :func:`_preferred`)
    and every other copy is added to ``dropped``.
    """
    index_by_hash: dict[str, int] = {}
    kept: list[TokenBlock] = []
    dropped: list[TokenBlock] = []

    for block in blocks:
        key = hash_content(block.content)
        if key not in index_by_hash:
            index_by_hash[key] = len(kept)
            kept.append(block)
            continue
        pos = index_by_hash[key]
        winner = _preferred(kept[pos], block)
        loser = block if winner is kept[pos] else kept[pos]
        kept[pos] = winner
        dropped.append(loser)

    return kept, dropped


__all__ = ["deduplicate_exact"]
