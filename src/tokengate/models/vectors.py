"""Block vector resolution: reuse or compute (CP-016).

The app (Beacon) usually supplies each block's BGE-M3 vector from LanceDB. We reuse those
to avoid re-embedding (ADR-011); only blocks missing a vector — or carrying one whose
dimension doesn't match the active model — are (re)computed, in a single batched
``embed`` call. Computed vectors are cached back onto the blocks.
"""

from __future__ import annotations

import numpy as np

from tokengate.core.block import TokenBlock
from tokengate.models.base import EmbeddingModel
from tokengate.utils.logging import get_logger, log_event

_logger = get_logger("models.vectors")


def _needs_embedding(block: TokenBlock, dim: int) -> bool:
    return block.vector is None or block.vector.shape[0] != dim


def ensure_block_vectors(
    blocks: list[TokenBlock], model: EmbeddingModel
) -> np.ndarray:
    """Ensure every block has a ``dim``-matching vector; return them as one matrix.

    Reuses present, correctly-sized vectors; (re)computes the rest in a single batched
    ``model.embed`` call and caches them back. A stored vector whose dim mismatches the
    model is recomputed (logged), since mixing dimensions would corrupt similarity.

    Returns an array of shape ``(len(blocks), model.dim)`` in block order. Empty input
    yields shape ``(0, model.dim)``.
    """
    if not blocks:
        return np.empty((0, model.dim), dtype=np.float32)

    missing = [i for i, b in enumerate(blocks) if _needs_embedding(b, model.dim)]
    mismatched = [
        i for i in missing
        if blocks[i].vector is not None and blocks[i].vector.shape[0] != model.dim  # type: ignore[union-attr]
    ]
    if mismatched:
        log_event(_logger, "block_vector_dim_mismatch_recompute",
                  count=len(mismatched), model_dim=model.dim)

    if missing:
        computed = model.embed([blocks[i].content for i in missing])
        for j, i in enumerate(missing):
            blocks[i].vector = np.asarray(computed[j], dtype=np.float32)
        log_event(_logger, "block_vectors_computed",
                  computed=len(missing), reused=len(blocks) - len(missing))

    resolved: list[np.ndarray] = []
    for b in blocks:
        assert b.vector is not None  # every block was just ensured to have a vector
        resolved.append(b.vector)
    return np.vstack(resolved).astype(np.float32)


def embed_query(query: str, model: EmbeddingModel) -> np.ndarray:
    """Embed the query into a 1-D float32 vector."""
    return np.asarray(model.embed([query])[0], dtype=np.float32)


__all__ = ["ensure_block_vectors", "embed_query"]
