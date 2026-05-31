"""Default BGE model implementations (CP-015).

- :class:`BGEM3Embedder` — ``BAAI/bge-m3`` dense embeddings (dim 1024) via
  sentence-transformers (ADR-013). Dense only for now; sparse/ColBERT modes are a later
  enhancement behind the ``flag`` extra.
- :class:`BGEReranker` — ``BAAI/bge-reranker-v2-m3`` cross-encoder.

Both are GPU-aware (ADR-017): they use CUDA when available, else CPU. Heavy libraries are
imported lazily inside ``__init__`` so importing this module stays cheap, and model
weights download to the HuggingFace cache on first construction.
"""

from __future__ import annotations

import numpy as np

from contextpilot.models.base import resolve_device
from contextpilot.utils.logging import get_logger, log_event

_logger = get_logger("models.bge")

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class BGEM3Embedder:
    """Dense BGE-M3 embedder producing L2-normalized float32 vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device or resolve_device()
        self.batch_size = batch_size
        log_event(_logger, "embedding_model_loading", model=model_name, device=self.device)
        self._model = SentenceTransformer(model_name, device=self.device)
        # Method was renamed across sentence-transformers versions; support both.
        if hasattr(self._model, "get_embedding_dimension"):
            dim = self._model.get_embedding_dimension()
        else:
            dim = self._model.get_sentence_embedding_dimension()
        if dim is None:  # pragma: no cover - sentence-transformers always reports this
            raise RuntimeError(f"could not determine embedding dim for {model_name!r}")
        self.dim: int = int(dim)
        log_event(_logger, "embedding_model_loaded", model=model_name, dim=self.dim)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # L2-normalized → cosine == dot product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class BGEReranker:
    """Cross-encoder reranker scoring (query, text) pairs."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = device or resolve_device()
        self.batch_size = batch_size
        log_event(_logger, "reranker_loading", model=model_name, device=self.device)
        self._model = CrossEncoder(model_name, device=self.device)
        log_event(_logger, "reranker_loaded", model=model_name)

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(s) for s in np.asarray(scores).ravel()]


__all__ = [
    "BGEM3Embedder",
    "BGEReranker",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_RERANKER_MODEL",
]
