"""Semantic (near-duplicate) deduplication (CP-019).

Exact dedup only catches byte-identical text. Two chunks can say the *same thing* in
different words ("requires Python, FastAPI, Docker" vs "stack includes Docker, Python
APIs, FastAPI"). This stage compares block embeddings: when a block's cosine similarity
to an already-kept block reaches ``threshold``, it is dropped as a semantic duplicate.

Representative selection is by **input order** — call this on a list already sorted
best-first (e.g. after reranking), so the higher-ranked block of a near-duplicate pair is
the one kept. Blocks without a vector cannot be compared and are always kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tokengate.core.block import TokenBlock
from tokengate.utils.logging import get_logger, log_event

_logger = get_logger("deduplication.semantic")


@dataclass
class SemanticDedupResult:
    """Outcome of semantic dedup, with provenance for the audit."""

    kept: list[TokenBlock] = field(default_factory=list)
    dropped: list[TokenBlock] = field(default_factory=list)
    # dropped block_id -> (representative kept block_id, similarity)
    duplicate_of: dict[str, str] = field(default_factory=dict)
    similarity: dict[str, float] = field(default_factory=dict)


def deduplicate_semantic(
    blocks: list[TokenBlock], *, threshold: float = 0.9
) -> SemanticDedupResult:
    """Drop blocks whose embedding is ≥ ``threshold`` similar to an earlier kept block.

    Assumes block vectors are L2-normalized (cosine == dot). Iterates in input order,
    keeping the first (best-ranked) of each near-duplicate group.
    """
    result = SemanticDedupResult()
    kept_vecs: list[np.ndarray | None] = []

    for block in blocks:
        vec = block.vector
        if vec is None:
            result.kept.append(block)
            kept_vecs.append(None)
            continue

        best_sim = -1.0
        best_rep: TokenBlock | None = None
        for kept_block, kept_vec in zip(result.kept, kept_vecs, strict=True):
            if kept_vec is None:
                continue
            sim = float(np.dot(vec, kept_vec))
            if sim > best_sim:
                best_sim = sim
                best_rep = kept_block

        if best_rep is not None and best_sim >= threshold:
            result.dropped.append(block)
            result.duplicate_of[block.block_id] = best_rep.block_id
            result.similarity[block.block_id] = best_sim
        else:
            result.kept.append(block)
            kept_vecs.append(vec)

    if result.dropped:
        log_event(_logger, "semantic_dedup_completed",
                  kept=len(result.kept), dropped=len(result.dropped), threshold=threshold)
    return result


__all__ = ["SemanticDedupResult", "deduplicate_semantic"]
