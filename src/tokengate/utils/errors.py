"""TokenGate exception hierarchy.

All errors raised by the library derive from :class:`TokenGateError`, so callers
can catch the whole family with a single ``except``. Implemented in CP-002.
"""

from __future__ import annotations


class TokenGateError(Exception):
    """Base class for every error raised by TokenGate."""


class InvalidBlockError(TokenGateError):
    """A :class:`~tokengate.core.block.TokenBlock` is malformed.

    Raised for empty content, out-of-range scores, or otherwise invalid fields.
    """


class BudgetError(TokenGateError):
    """The token budget is impossible to satisfy or is misconfigured.

    Example: ``max_prompt_tokens`` is non-positive, or required blocks alone exceed
    the budget in a configuration that forbids that.
    """


class OptimizationError(TokenGateError):
    """An unexpected failure occurred while running the optimization pipeline."""


class ConfigurationError(TokenGateError):
    """The optimizer was given invalid configuration (bad weights, strategy, etc.)."""


__all__ = [
    "TokenGateError",
    "InvalidBlockError",
    "BudgetError",
    "OptimizationError",
    "ConfigurationError",
]
