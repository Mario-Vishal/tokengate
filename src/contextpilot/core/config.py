"""Optimizer configuration & strategy presets (CP-005).

``OptimizerConfig`` holds every knob the pipeline reads. V1 ships a single functional
strategy, ``"balanced"``; the config shape already carries the fields future presets
(speed/quality/max_compression — V2) will set, so adding them is data, not an API
change (ADR-008).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from contextpilot.utils.errors import ConfigurationError

if TYPE_CHECKING:  # avoid import cycle; TokenCounter lands in CP-006
    from contextpilot.budgeting.token_counter import TokenCounter

STRATEGY_BALANCED = "balanced"
# Reserved for V2 (accepted by name once implemented).
_V1_STRATEGIES = frozenset({STRATEGY_BALANCED})


@dataclass
class OptimizerConfig:
    """Tunable settings for one :class:`~contextpilot.ContextPilot` instance."""

    max_prompt_tokens: int = 4096
    strategy: str = STRATEGY_BALANCED
    # --- hybrid ranking signal weights (renormalized per block over available signals) ---
    semantic_weight: float = 0.45
    keyword_weight: float = 0.25
    recency_weight: float = 0.10
    source_priority_weight: float = 0.10
    token_efficiency_weight: float = 0.10
    # Optional per-source priority in [0,1]; sources absent from the map skip that signal.
    source_priorities: dict[str, float] = field(default_factory=dict)
    enable_dedup: bool = True
    enable_compression: bool = True
    # Fraction of the budget held back as headroom against tokenizer estimation drift.
    safety_margin: float = 0.05
    # Optional injected counter; the optimizer falls back to the heuristic default.
    token_counter: TokenCounter | None = None

    def __post_init__(self) -> None:
        if self.max_prompt_tokens <= 0:
            raise ConfigurationError("max_prompt_tokens must be a positive integer")
        if self.strategy not in _V1_STRATEGIES:
            raise ConfigurationError(
                f"unknown strategy {self.strategy!r}; "
                f"supported in V1: {sorted(_V1_STRATEGIES)}"
            )
        if any(w < 0 for w in self._weights.values()):
            raise ConfigurationError("scoring weights must be non-negative")
        if sum(self._weights.values()) <= 0:
            raise ConfigurationError("at least one scoring weight must be > 0")
        if not all(0.0 <= p <= 1.0 for p in self.source_priorities.values()):
            raise ConfigurationError("source_priorities values must be in [0.0, 1.0]")
        if not (0.0 <= self.safety_margin < 1.0):
            raise ConfigurationError("safety_margin must be in [0.0, 1.0)")

    @property
    def _weights(self) -> dict[str, float]:
        return {
            "semantic": self.semantic_weight,
            "keyword": self.keyword_weight,
            "recency": self.recency_weight,
            "source_priority": self.source_priority_weight,
            "token_efficiency": self.token_efficiency_weight,
        }

    @property
    def effective_budget(self) -> int:
        """Token budget after applying the safety margin (floored, >= 1)."""
        return max(1, int(self.max_prompt_tokens * (1.0 - self.safety_margin)))

    @classmethod
    def for_strategy(
        cls, strategy: str = STRATEGY_BALANCED, **overrides: object
    ) -> OptimizerConfig:
        """Build a config for a named strategy with optional field overrides.

        In V1 only ``"balanced"`` is available; the defaults already encode it.
        """
        preset = _PRESETS.get(strategy)
        if preset is None:
            raise ConfigurationError(
                f"unknown strategy {strategy!r}; supported in V1: {sorted(_PRESETS)}"
            )
        return cls(**{**preset, **overrides})  # type: ignore[arg-type]


# Per-strategy default fields. V2 presets get added here without API changes.
_PRESETS: dict[str, dict[str, object]] = {
    STRATEGY_BALANCED: {
        "strategy": STRATEGY_BALANCED,
        "semantic_weight": 0.45,
        "keyword_weight": 0.25,
        "recency_weight": 0.10,
        "source_priority_weight": 0.10,
        "token_efficiency_weight": 0.10,
        "enable_dedup": True,
        "enable_compression": True,
        "safety_margin": 0.05,
    },
}


__all__ = ["OptimizerConfig", "STRATEGY_BALANCED"]
