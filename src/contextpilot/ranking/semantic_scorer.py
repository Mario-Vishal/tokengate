"""Semantic scoring from embeddings (CP-017).

Cosine similarity between the query vector and each block vector. Because the embedder
returns L2-normalized vectors (ADR-013/CP-015), cosine is a plain dot product. Scores are
clamped to ``[0, 1]`` (negative similarities mean "not relevant") to match the
``semantic_score`` field's range and to combine cleanly with other signals.
"""

from __future__ import annotations

import numpy as np

from contextpilot.core.block import ContextBlock


def cosine_scores(query_vec: np.ndarray, block_matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of ``query_vec`` against each row of ``block_matrix``.

    Inputs are assumed L2-normalized; the result is clamped to ``[0, 1]``. Returns a 1-D
    array of length ``block_matrix.shape[0]`` (empty if there are no blocks).
    """
    if block_matrix.size == 0:
        return np.empty((0,), dtype=np.float32)
    sims = block_matrix @ query_vec
    return np.asarray(np.clip(sims, 0.0, 1.0), dtype=np.float32)


def apply_semantic_scores(
    blocks: list[ContextBlock], query_vec: np.ndarray, block_matrix: np.ndarray
) -> None:
    """Set ``semantic_score`` on each block from cosine(query, block_vector)."""
    scores = cosine_scores(query_vec, block_matrix)
    for block, score in zip(blocks, scores, strict=True):
        block.semantic_score = float(score)


__all__ = ["cosine_scores", "apply_semantic_scores"]
