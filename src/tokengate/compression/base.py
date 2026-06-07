"""Compressor abstraction (CP-031).

The budgeter shrinks a kept block by delegating to a ``Compressor``. This lets the engine
swap compression strategies — embedding sentence-selection (:class:`ExtractiveCompressor`)
or a learned token-classifier (:class:`LLMLinguaCompressor`) — behind one interface, so the
budgeting logic never changes. A compressor must be **lossless for relevance**: it only
removes low-value text and returns the block unchanged when it can't shrink it.
"""

from __future__ import annotations

from typing import Protocol

from tokengate.core.block import TokenBlock


class Compressor(Protocol):
    """Shrinks a block's text while preserving its query-relevant content."""

    def compress_block(
        self, block: TokenBlock, query: str, *, keep_ratio: float
    ) -> TokenBlock:
        """Return a compressed copy of ``block``, or the block unchanged.

        ``keep_ratio`` is the aggressiveness knob in ``(0, 1]`` — roughly the fraction of
        content to retain (lower = more aggressive). Implementations must return the input
        block object unchanged when no compression is applied, so callers can detect a
        no-op by identity.
        """
        ...


__all__ = ["Compressor"]
