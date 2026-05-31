"""ContextPilot exception hierarchy.

All errors raised by the library derive from :class:`ContextPilotError`, so callers
can catch the whole family with a single ``except``. Implemented in CP-002.
"""

from __future__ import annotations


class ContextPilotError(Exception):
    """Base class for every error raised by ContextPilot."""


class InvalidBlockError(ContextPilotError):
    """A :class:`~contextpilot.core.block.ContextBlock` is malformed.

    Raised for empty content, out-of-range scores, or otherwise invalid fields.
    """


class BudgetError(ContextPilotError):
    """The token budget is impossible to satisfy or is misconfigured.

    Example: ``max_prompt_tokens`` is non-positive, or required blocks alone exceed
    the budget in a configuration that forbids that.
    """


class OptimizationError(ContextPilotError):
    """An unexpected failure occurred while running the optimization pipeline."""


__all__ = [
    "ContextPilotError",
    "InvalidBlockError",
    "BudgetError",
    "OptimizationError",
]
