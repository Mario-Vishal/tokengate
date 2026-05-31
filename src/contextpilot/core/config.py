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

STRATEGY_SPEED = "speed"
STRATEGY_BALANCED = "balanced"
STRATEGY_QUALITY = "quality"
STRATEGY_MAX_COMPRESSION = "max_compression"
_STRATEGIES = frozenset(
    {STRATEGY_SPEED, STRATEGY_BALANCED, STRATEGY_QUALITY, STRATEGY_MAX_COMPRESSION}
)


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
    # Relevance-driven compression: keep sentences scoring >= ratio * best sentence
    # (drop boilerplate). No token target (ADR-019). Higher = more aggressive.
    compression_keep_ratio: float = 0.5
    # Embedding-cosine dedup: blocks at/above this similarity are treated as duplicates.
    enable_semantic_dedup: bool = True
    semantic_dedup_threshold: float = 0.9
    # Keep the top-N blocks after neural reranking before later stages (None = keep all).
    rerank_top_n: int | None = 15
    # MMR diversity selection: balance relevance vs redundancy (lambda in [0,1]).
    enable_mmr: bool = True
    mmr_lambda: float = 0.5
    # How much the cross-encoder rerank score dominates final_score post-rerank (ADR-018).
    rerank_weight: float = 0.7
    # Drop blocks whose (rerank-blended) final_score is below this before budgeting, so
    # cheap low-relevance noise can't crowd out relevant content. 0.0 disables.
    relevance_floor: float = 0.15
    # Fraction of the budget held back as headroom against tokenizer estimation drift.
    safety_margin: float = 0.05
    # Optional injected counter; the optimizer falls back to the heuristic default.
    token_counter: TokenCounter | None = None

    def __post_init__(self) -> None:
        if self.max_prompt_tokens <= 0:
            raise ConfigurationError("max_prompt_tokens must be a positive integer")
        if self.strategy not in _STRATEGIES:
            raise ConfigurationError(
                f"unknown strategy {self.strategy!r}; supported: {sorted(_STRATEGIES)}"
            )
        if any(w < 0 for w in self._weights.values()):
            raise ConfigurationError("scoring weights must be non-negative")
        if sum(self._weights.values()) <= 0:
            raise ConfigurationError("at least one scoring weight must be > 0")
        if not all(0.0 <= p <= 1.0 for p in self.source_priorities.values()):
            raise ConfigurationError("source_priorities values must be in [0.0, 1.0]")
        if self.rerank_top_n is not None and self.rerank_top_n <= 0:
            raise ConfigurationError("rerank_top_n must be a positive integer or None")
        if not (0.0 <= self.semantic_dedup_threshold <= 1.0):
            raise ConfigurationError("semantic_dedup_threshold must be in [0.0, 1.0]")
        if not (0.0 <= self.mmr_lambda <= 1.0):
            raise ConfigurationError("mmr_lambda must be in [0.0, 1.0]")
        if not (0.0 <= self.rerank_weight <= 1.0):
            raise ConfigurationError("rerank_weight must be in [0.0, 1.0]")
        if not (0.0 <= self.relevance_floor <= 1.0):
            raise ConfigurationError("relevance_floor must be in [0.0, 1.0]")
        if not (0.0 < self.compression_keep_ratio <= 1.0):
            raise ConfigurationError("compression_keep_ratio must be in (0.0, 1.0]")
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
        """Build a config for a named strategy with optional field overrides."""
        preset = _PRESETS.get(strategy)
        if preset is None:
            raise ConfigurationError(
                f"unknown strategy {strategy!r}; supported: {sorted(_PRESETS)}"
            )
        return cls(**{**preset, **overrides})  # type: ignore[arg-type]


# Per-strategy presets. Each meaningfully changes the pipeline (CP-023).
_PRESETS: dict[str, dict[str, object]] = {
    # Fewest stages for lowest latency: shallow rerank, no semantic dedup / MMR /
    # compression. Leans on lexical + semantic scoring.
    STRATEGY_SPEED: {
        "strategy": STRATEGY_SPEED,
        "rerank_top_n": 5,
        "enable_semantic_dedup": False,
        "enable_compression": False,
        "enable_mmr": False,
        "relevance_floor": 0.2,  # prune aggressively for speed
    },
    # Default all-round profile.
    STRATEGY_BALANCED: {
        "strategy": STRATEGY_BALANCED,
        "rerank_top_n": 15,
        "enable_semantic_dedup": True,
        "enable_compression": True,
        "enable_mmr": True,
    },
    # Deepest rerank, semantic-weighted, full diversity — best answers, slower.
    STRATEGY_QUALITY: {
        "strategy": STRATEGY_QUALITY,
        "semantic_weight": 0.55,
        "keyword_weight": 0.20,
        "rerank_top_n": 30,
        "enable_semantic_dedup": True,
        "enable_compression": True,
        "enable_mmr": True,
        "mmr_lambda": 0.7,
        "relevance_floor": 0.10,  # keep more borderline context
    },
    # Squeeze the most into the budget: aggressive compression + larger safety margin +
    # more diversity so varied evidence is compressed in.
    STRATEGY_MAX_COMPRESSION: {
        "strategy": STRATEGY_MAX_COMPRESSION,
        "rerank_top_n": 20,
        "enable_semantic_dedup": True,
        "enable_compression": True,
        "enable_mmr": True,
        "mmr_lambda": 0.4,
        "safety_margin": 0.10,
    },
}


__all__ = [
    "OptimizerConfig",
    "STRATEGY_SPEED",
    "STRATEGY_BALANCED",
    "STRATEGY_QUALITY",
    "STRATEGY_MAX_COMPRESSION",
]
