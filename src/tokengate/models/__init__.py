"""TokenGate neural model layer (embeddings + reranker).

Importing this subpackage is cheap; the heavy ML libraries load only when a concrete
real model (e.g. :class:`~tokengate.models.bge.BGEM3Embedder`) is constructed. The
fakes are import-light and offline.
"""

from __future__ import annotations

from tokengate.models.base import EmbeddingModel, Reranker, resolve_device
from tokengate.models.fakes import FakeEmbeddingModel, FakeReranker
from tokengate.models.vectors import embed_query, ensure_block_vectors

__all__ = [
    "EmbeddingModel",
    "Reranker",
    "resolve_device",
    "FakeEmbeddingModel",
    "FakeReranker",
    "ensure_block_vectors",
    "embed_query",
    # Real models imported lazily to avoid pulling torch on `import tokengate.models`:
    #   from tokengate.models.bge import BGEM3Embedder, BGEReranker
]
