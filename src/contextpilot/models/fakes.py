"""Deterministic fake models for fast, offline unit tests (CP-015).

These are a *test* substitution only (ADR-010) — never a product mode. They implement the
same protocols as the real BGE models, with meaningful behavior so pipeline stages can be
tested without downloading weights or using a GPU:

- :class:`FakeEmbeddingModel` hashes tokens into buckets, so texts sharing words get
  higher cosine similarity (useful for semantic dedup / MMR tests).
- :class:`FakeReranker` scores by query-term overlap (query-aware, deterministic).
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _stable_bucket(token: str, dim: int) -> int:
    """Process-stable hash bucket (Python's built-in hash() is salted per run)."""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dim


class FakeEmbeddingModel:
    """Hashing bag-of-words embedder producing L2-normalized vectors."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokens(text):
            vec[_stable_bucket(tok, self.dim)] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)


class FakeReranker:
    """Lexical-overlap reranker: score = |query∩text terms| / |query terms|."""

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        q = set(_tokens(query))
        if not q:
            return [0.0 for _ in texts]
        return [len(q & set(_tokens(t))) / len(q) for t in texts]


__all__ = ["FakeEmbeddingModel", "FakeReranker"]
