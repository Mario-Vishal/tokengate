"""Per-stage tracing for the optimizer pipeline (CP-028).

A tiny, dependency-free accumulator the optimizer feeds one :class:`StageRecord`
per stage. Timing uses :func:`time.perf_counter`; token sums reuse each block's
cached token count (``ensure_token_count``) so tracing adds negligible overhead.
When ``enabled`` is False every method is a cheap no-op, so callers don't branch.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from tokengate.budgeting.token_counter import TokenCounter
from tokengate.core.block import TokenBlock
from tokengate.core.result import StageRecord


class StageTracer:
    """Collects :class:`StageRecord` rows describing the optimizer funnel."""

    def __init__(self, counter: TokenCounter, *, enabled: bool = True) -> None:
        self._counter = counter
        self.enabled = enabled
        self.records: list[StageRecord] = []

    def start(self) -> float:
        """Timestamp to pass back to :meth:`add` (0.0 when disabled)."""
        return time.perf_counter() if self.enabled else 0.0

    def _tokens(self, blocks: Sequence[TokenBlock]) -> int:
        return sum(b.ensure_token_count(self._counter) for b in blocks)

    def add(
        self,
        stage: str,
        *,
        before: Sequence[TokenBlock],
        after: Sequence[TokenBlock],
        t0: float,
        tokens_out: int | None = None,
        dropped: int | None = None,
    ) -> None:
        """Record one stage. ``tokens_out`` overrides the summed count (e.g. after
        compression); ``dropped`` overrides the default ``len(before) - len(after)``."""
        if not self.enabled:
            return
        duration_ms = round((time.perf_counter() - t0) * 1000, 3)
        tokens_in = self._tokens(before)
        tout = tokens_out if tokens_out is not None else self._tokens(after)
        self.records.append(
            StageRecord(
                stage=stage,
                blocks_in=len(before),
                blocks_out=len(after),
                tokens_in=tokens_in,
                tokens_out=tout,
                dropped=dropped if dropped is not None else max(0, len(before) - len(after)),
                duration_ms=duration_ms,
            )
        )


__all__ = ["StageTracer"]
