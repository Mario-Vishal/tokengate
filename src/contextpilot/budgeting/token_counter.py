"""Token counting (CP-006).

Token counts depend on the target model's tokenizer, so counting is pluggable: any
object satisfying the :class:`TokenCounter` protocol works. The default
:class:`HeuristicTokenCounter` is dependency-free and slightly conservative (it never
*under*-counts vs the common ~4-chars/token rule, so the budgeter errs toward staying
under budget). :class:`TiktokenCounter` gives exact counts when the optional
``tiktoken`` extra is installed (``pip install "contextpilot[tiktoken]"``).
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Anything that can estimate the number of tokens in a string."""

    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Dependency-free token estimator.

    Estimate = ``max(ceil(chars / 4), word_count)``. The ``chars/4`` term tracks the
    well-known rule of thumb; the ``word_count`` floor guards short, word-dense text.
    The result is non-negative and monotonic non-decreasing with text length.
    """

    _CHARS_PER_TOKEN = 4

    def count(self, text: str) -> int:
        if not text:
            return 0
        char_estimate = math.ceil(len(text) / self._CHARS_PER_TOKEN)
        word_floor = len(text.split())
        return max(char_estimate, word_floor)


class TiktokenCounter:
    """Exact, model-aware token counts via the optional ``tiktoken`` dependency.

    Raises :class:`ImportError` with installation guidance if ``tiktoken`` is missing,
    so callers can fall back to :class:`HeuristicTokenCounter` gracefully.
    """

    def __init__(self, encoding: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "TiktokenCounter requires the optional 'tiktoken' extra. "
                'Install it with: pip install "contextpilot[tiktoken]"'
            ) from exc
        self._encoding_name = encoding
        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))


def resolve_counter(counter: TokenCounter | None) -> TokenCounter:
    """Return ``counter`` if given, else a fresh :class:`HeuristicTokenCounter`."""
    return counter if counter is not None else HeuristicTokenCounter()


__all__ = [
    "TokenCounter",
    "HeuristicTokenCounter",
    "TiktokenCounter",
    "resolve_counter",
]
