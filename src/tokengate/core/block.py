"""TokenBlock model + validation (CP-003).

A :class:`TokenBlock` is the unit TokenGate operates on. Anything that might be
sent to an LLM — a retrieved chunk, a system instruction, the user's own notes —
becomes a block. The library reads ``content`` / ``semantic_score`` and writes
``keyword_score`` / ``final_score`` / ``token_count`` as it runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from tokengate.utils.errors import InvalidBlockError
from tokengate.utils.hashing import block_id_from_content

if TYPE_CHECKING:  # avoid import cycle; TokenCounter lands in CP-006
    from tokengate.budgeting.token_counter import TokenCounter

# Scores supplied/used by the library are expected in this inclusive range.
_SCORE_MIN = 0.0
_SCORE_MAX = 1.0


def _validate_score(name: str, value: float | None) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)):
        raise InvalidBlockError(f"{name} must be a number, got {type(value).__name__}")
    if not (_SCORE_MIN <= float(value) <= _SCORE_MAX):
        raise InvalidBlockError(
            f"{name} must be within [{_SCORE_MIN}, {_SCORE_MAX}], got {value}"
        )


def _coerce_vector(value: Any) -> np.ndarray | None:
    """Normalize a provided vector to a 1-D float32 ndarray (or None)."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise InvalidBlockError("vector must be a non-empty 1-D sequence of numbers")
    return arr


@dataclass
class TokenBlock:
    """A single candidate piece of context.

    Fields mirror docs/ARCHITECTURE.md. ``block_id`` is derived from a stable hash of
    ``content`` when not provided, so identical content yields identical ids.
    """

    content: str
    block_id: str = ""
    block_type: str = "document"
    source_id: str | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    semantic_score: float | None = None
    keyword_score: float | None = None
    final_score: float | None = None
    # Cross-encoder reranker score (raw, unbounded — not constrained to [0,1]).
    rerank_score: float | None = None
    required: bool = False
    cacheable: bool = False
    compressible: bool = True
    # Dense embedding (e.g. BGE-M3 from the app's LanceDB). Reused when present so the
    # engine doesn't re-embed (ADR-011). Excluded from eq/repr to avoid ndarray-in-eq
    # ambiguity and huge reprs; serialized as a plain list by to_dict().
    vector: np.ndarray | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise InvalidBlockError("TokenBlock.content must be a non-empty string")
        if not self.block_id:
            self.block_id = block_id_from_content(self.content)
        _validate_score("semantic_score", self.semantic_score)
        _validate_score("keyword_score", self.keyword_score)
        _validate_score("final_score", self.final_score)
        if self.token_count is not None and self.token_count < 0:
            raise InvalidBlockError("token_count must be non-negative")
        self.vector = _coerce_vector(self.vector)

    # --- token counting ---------------------------------------------------

    def ensure_token_count(self, counter: TokenCounter) -> int:
        """Compute and cache ``token_count`` via ``counter`` if not already set.

        Returns the (now-populated) token count. Idempotent.
        """
        if self.token_count is None:
            self.token_count = counter.count(self.content)
        return self.token_count

    # --- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "content": self.content,
            "block_type": self.block_type,
            "source_id": self.source_id,
            "token_count": self.token_count,
            "metadata": dict(self.metadata),
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "final_score": self.final_score,
            "rerank_score": self.rerank_score,
            "required": self.required,
            "cacheable": self.cacheable,
            "compressible": self.compressible,
            "vector": self.vector.tolist() if self.vector is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenBlock:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def copy(self, **overrides: Any) -> TokenBlock:
        """Return a shallow copy with optional field overrides."""
        data = self.to_dict()
        data.update(overrides)
        return TokenBlock.from_dict(data)


__all__ = ["TokenBlock"]
