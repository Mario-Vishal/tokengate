"""Optimizer configuration & strategy presets (CP-005).

``OptimizerConfig`` holds every knob the pipeline reads. V1 ships a single functional
strategy, ``"balanced"``; the config shape already carries the fields future presets
(speed/quality/max_compression — V2) will set, so adding them is data, not an API
change (ADR-008).
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tokengate.utils.errors import ConfigurationError

if TYPE_CHECKING:  # avoid import cycle; TokenCounter lands in CP-006
    from tokengate.budgeting.token_counter import TokenCounter

STRATEGY_SPEED = "speed"
STRATEGY_BALANCED = "balanced"
STRATEGY_QUALITY = "quality"
STRATEGY_MAX_COMPRESSION = "max_compression"
_STRATEGIES = frozenset(
    {STRATEGY_SPEED, STRATEGY_BALANCED, STRATEGY_QUALITY, STRATEGY_MAX_COMPRESSION}
)


@dataclass
class OptimizerConfig:
    """Tunable settings for one :class:`~tokengate.TokenGate` instance."""

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
    # OFF by default. Compression is an opt-in lever: the conservative "extractive" backend
    # is a near-no-op on structured content, and the aggressive "llmlingua2" backend can drop
    # the linking entities that make answers correct (measured end-to-end). Turn it on per
    # strategy (max_compression) or explicitly when you've validated it on your corpus.
    enable_compression: bool = False
    # Which compressor backend to use: "extractive" (embedding sentence-selection, no extra
    # model) or "llmlingua2" (a small pretrained token-classification transformer that keeps
    # high-information tokens — works on prose/blob content where sentence-extraction can't).
    compression_backend: str = "extractive"
    # Model id for the llmlingua2 backend (small multilingual BERT by default).
    llmlingua_model: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    # Compress every kept block up front (the main token-saving lever), not only blocks
    # that overflow the budget. When False, compression is compress-to-fit: blocks that
    # already fit are kept verbatim. Requires enable_compression.
    compress_always: bool = True
    # Relevance-driven compression: keep sentences scoring >= ratio * best sentence
    # (drop boilerplate). No token target (ADR-019). Higher = more aggressive.
    compression_keep_ratio: float = 0.5
    # Within-block sentence dedup during compression: collapse sentences whose cosine is
    # at/above this to a single copy (1.0 = exact only).
    compression_sentence_dedup_threshold: float = 0.95
    # Embedding-cosine dedup: blocks at/above this similarity are treated as duplicates.
    enable_semantic_dedup: bool = True
    semantic_dedup_threshold: float = 0.9
    # Run the cross-encoder reranker. When False, final_score stays the hybrid-rank score
    # (lets recipes test "no reranker" / naive baselines). The expensive stage to skip.
    enable_reranker: bool = True
    # Keep the top-N blocks after neural reranking before later stages (None = keep all).
    rerank_top_n: int | None = 15
    # MMR diversity selection: balance relevance vs redundancy (lambda in [0,1]).
    enable_mmr: bool = True
    mmr_lambda: float = 0.5
    # How much the cross-encoder rerank score dominates final_score post-rerank (ADR-018).
    rerank_weight: float = 0.7
    # Minimum ceiling used when normalizing raw rerank scores. When all raw scores are
    # below this value, absolute scaling is used instead of min-max — preventing
    # irrelevant blocks from being inflated to look relevant relative to each other.
    # E.g. scores [0.001, 0.002, 0.003] stay near zero instead of mapping to [0, 0.5, 1].
    rerank_normalization_ceiling: float = 0.5
    # Hard absolute floor on the raw cross-encoder rerank score. Blocks below this are
    # dropped unconditionally before budgeting — no embedding similarity can rescue them.
    # 0.0 disables. Keep low (~0.01) so synthesis queries (where relevant docs score
    # 0.013–0.07) are not cut; only true zero-signal blocks (0.000–0.008) are dropped.
    min_rerank_score: float = 0.01
    # Drop blocks whose (rerank-blended) final_score is below this before budgeting, so
    # cheap low-relevance noise can't crowd out relevant content. 0.0 disables. Ignored
    # when adaptive_cutoff is on (the cutoff replaces it).
    relevance_floor: float = 0.15
    # Per-query adaptive relevance cutoff (CP-030): derive how many blocks to keep from the
    # shape of this query's score distribution (relevance-cliff detection) instead of the
    # fixed relevance_floor / min_rerank_score. Focused queries keep few, broad/synthesis
    # queries keep many. When True, relevance_floor and min_rerank_score are bypassed.
    adaptive_cutoff: bool = True
    # Never let the adaptive cutoff keep fewer than this many blocks (>= 1).
    adaptive_cutoff_min_keep: int = 1
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
        if not (0.0 < self.rerank_normalization_ceiling <= 1.0):
            raise ConfigurationError("rerank_normalization_ceiling must be in (0.0, 1.0]")
        if not (0.0 <= self.min_rerank_score <= 1.0):
            raise ConfigurationError("min_rerank_score must be in [0.0, 1.0]")
        if not (0.0 <= self.relevance_floor <= 1.0):
            raise ConfigurationError("relevance_floor must be in [0.0, 1.0]")
        if self.adaptive_cutoff_min_keep < 1:
            raise ConfigurationError("adaptive_cutoff_min_keep must be >= 1")
        if self.compression_backend not in ("extractive", "llmlingua2"):
            raise ConfigurationError(
                "compression_backend must be 'extractive' or 'llmlingua2'"
            )
        if not (0.0 < self.compression_keep_ratio <= 1.0):
            raise ConfigurationError("compression_keep_ratio must be in (0.0, 1.0]")
        if not (0.0 <= self.compression_sentence_dedup_threshold <= 1.0):
            raise ConfigurationError(
                "compression_sentence_dedup_threshold must be in [0.0, 1.0]"
            )
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

    # --- config-file loading (CP-050) -------------------------------------
    # Layer a user's settings over a strategy's defaults: omit a key, keep the default.
    # One source of truth so "set it in tokengate.toml" and "set it in the dashboard"
    # resolve to the same file.

    @classmethod
    def from_dict(
        cls, settings: Mapping[str, Any], *, strategy: str | None = None
    ) -> OptimizerConfig:
        """Build a config from a mapping, overriding only the provided keys.

        The base strategy is ``strategy`` (if given), else the mapping's ``strategy``
        key, else ``"balanced"``. Unknown keys raise — typos shouldn't silently no-op.
        """
        loadable = {f.name for f in fields(cls) if f.name != "token_counter"}
        unknown = set(settings) - loadable
        if unknown:
            raise ConfigurationError(
                f"unknown config key(s): {sorted(unknown)}; supported: {sorted(loadable)}"
            )
        base = strategy or str(settings.get("strategy", STRATEGY_BALANCED))
        overrides = {k: v for k, v in settings.items() if k != "strategy"}
        return cls.for_strategy(base, **overrides)

    @classmethod
    def from_file(cls, path: str | Path, *, strategy: str | None = None) -> OptimizerConfig:
        """Load a TOML config file. Reads the ``[tokengate]`` table if present, else
        the whole document as the settings mapping."""
        text = Path(path).read_text(encoding="utf-8")
        data = tomllib.loads(text)
        section = data.get("tokengate", data)
        if not isinstance(section, dict):
            raise ConfigurationError("[tokengate] section must be a table")
        return cls.from_dict(section, strategy=strategy)

    @classmethod
    def load(
        cls, path: str | Path | None = None, *, strategy: str | None = None
    ) -> OptimizerConfig:
        """Resolve a config file and load it, or return strategy defaults if none exists.

        Resolution order: explicit ``path`` → ``$TOKENGATE_CONFIG`` → ``./tokengate.toml``
        → built-in defaults. An explicit ``path`` or env var that points nowhere is an error
        (a silent fallback would hide a misconfiguration); a missing ``./tokengate.toml`` is not.
        """
        resolved = resolve_config_path(path)
        if resolved is None:
            return cls.for_strategy(strategy or STRATEGY_BALANCED)
        return cls.from_file(resolved, strategy=strategy)


# Per-strategy presets. Each meaningfully changes the pipeline (CP-023).
_PRESETS: dict[str, dict[str, object]] = {
    # Fewest stages for lowest latency: shallow rerank, no semantic dedup / MMR /
    # compression. Leans on lexical + semantic scoring.
    STRATEGY_SPEED: {
        "strategy": STRATEGY_SPEED,
        "enable_reranker": False,  # skip the cross-encoder — the big latency cost
        "rerank_top_n": 5,
        "enable_semantic_dedup": False,
        "enable_compression": False,
        "enable_mmr": False,
        "adaptive_cutoff": False,  # speed uses a cheap fixed floor, not per-query detection
        "relevance_floor": 0.2,  # prune aggressively for speed
    },
    # Default all-round profile.
    STRATEGY_BALANCED: {
        "strategy": STRATEGY_BALANCED,
        "rerank_top_n": 15,
        "enable_semantic_dedup": True,
        "enable_compression": False,  # savings come from dedup + selection, not lossy compression
        "enable_mmr": True,
    },
    # Deepest rerank, semantic-weighted, full diversity — best answers, slower.
    STRATEGY_QUALITY: {
        "strategy": STRATEGY_QUALITY,
        "semantic_weight": 0.55,
        "keyword_weight": 0.20,
        "rerank_top_n": 30,
        "enable_semantic_dedup": True,
        "enable_compression": False,  # quality favors answerability; compression risks it
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


def resolve_config_path(path: str | Path | None = None) -> Path | None:
    """Find the config file to load: explicit arg → ``$TOKENGATE_CONFIG`` → ``./tokengate.toml``.

    Returns ``None`` when no file is found via the implicit path (caller falls back to
    defaults). An explicit ``path`` or ``$TOKENGATE_CONFIG`` that doesn't exist is an error.
    """
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ConfigurationError(f"config file not found: {p}")
        return p
    env = os.environ.get("TOKENGATE_CONFIG")
    if env:
        p = Path(env)
        if not p.is_file():
            raise ConfigurationError(f"$TOKENGATE_CONFIG points to a missing file: {p}")
        return p
    local = Path.cwd() / "tokengate.toml"
    return local if local.is_file() else None


__all__ = [
    "OptimizerConfig",
    "STRATEGY_SPEED",
    "STRATEGY_BALANCED",
    "STRATEGY_QUALITY",
    "STRATEGY_MAX_COMPRESSION",
    "resolve_config_path",
]
