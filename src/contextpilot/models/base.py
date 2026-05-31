"""Model interfaces for the neural engine (CP-015).

These protocols exist for **configurability and testing**, not optionality (ADR-010):
the engine always has a real model behind them. ``EmbeddingModel`` produces L2-normalized
vectors (so cosine similarity is a plain dot product); ``Reranker`` is a query-aware
cross-encoder that scores (query, text) pairs.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


def resolve_device(mode: str = "auto") -> str:
    """Resolve a torch device string from a mode.

    ``"auto"`` → CUDA if available else CPU. ``"cpu_only"`` → CPU. ``"force_gpu"`` → CUDA
    (raises if unavailable). GPU is preferred, never required (ADR-017).
    """
    import torch

    if mode == "cpu_only":
        return "cpu"
    if mode == "force_gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("force_gpu requested but CUDA is not available")
        return "cuda"
    if mode != "auto":
        raise ValueError(f"unknown device mode {mode!r}")
    return "cuda" if torch.cuda.is_available() else "cpu"


@runtime_checkable
class EmbeddingModel(Protocol):
    """Encodes texts into L2-normalized dense vectors."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape ``(len(texts), dim)``, dtype float32, L2-normalized."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Scores how well each text answers the query (higher = more relevant)."""

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text, in input order."""
        ...


__all__ = ["EmbeddingModel", "Reranker", "resolve_device"]
