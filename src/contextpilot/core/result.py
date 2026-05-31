"""Result & audit models (CP-004).

These are the value objects ``ContextPilot.optimize()`` returns. They are plain,
serializable dataclasses so applications (e.g. Beacon's AuditPage) can render or
persist them directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from contextpilot.core.block import ContextBlock

# The three possible fates of a candidate block.
Decision = Literal["included", "compressed", "dropped"]
DECISION_INCLUDED: Decision = "included"
DECISION_COMPRESSED: Decision = "compressed"
DECISION_DROPPED: Decision = "dropped"


@dataclass
class BlockDecision:
    """Why a single block was included, compressed, or dropped."""

    block_id: str
    decision: Decision
    reason: str
    original_tokens: int
    final_tokens: int
    score: float | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "decision": self.decision,
            "reason": self.reason,
            "original_tokens": self.original_tokens,
            "final_tokens": self.final_tokens,
            "score": self.score,
            "rerank_score": self.rerank_score,
        }


@dataclass
class AuditReport:
    """A complete, serializable explanation of one optimization run."""

    total_candidate_blocks: int
    total_candidate_tokens: int
    final_prompt_tokens: int
    tokens_saved: int
    tokens_saved_percent: float
    included_count: int
    compressed_count: int
    dropped_count: int
    decisions: list[BlockDecision] = field(default_factory=list)
    # Which models drove the run, e.g. {"embedding_model": "...", "reranker": "...",
    # "final_llm": None}. final_llm is filled in by the app, not the library.
    models_used: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_candidate_blocks": self.total_candidate_blocks,
            "total_candidate_tokens": self.total_candidate_tokens,
            "final_prompt_tokens": self.final_prompt_tokens,
            "tokens_saved": self.tokens_saved,
            "tokens_saved_percent": self.tokens_saved_percent,
            "included_count": self.included_count,
            "compressed_count": self.compressed_count,
            "dropped_count": self.dropped_count,
            "decisions": [d.to_dict() for d in self.decisions],
            "models_used": dict(self.models_used),
        }


@dataclass
class OptimizationResult:
    """What ``optimize()`` returns: the prompt plus the full block breakdown."""

    query: str
    final_prompt: str
    included_blocks: list[ContextBlock] = field(default_factory=list)
    compressed_blocks: list[ContextBlock] = field(default_factory=list)
    dropped_blocks: list[ContextBlock] = field(default_factory=list)
    audit: AuditReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "final_prompt": self.final_prompt,
            "included_blocks": [b.to_dict() for b in self.included_blocks],
            "compressed_blocks": [b.to_dict() for b in self.compressed_blocks],
            "dropped_blocks": [b.to_dict() for b in self.dropped_blocks],
            "audit": self.audit.to_dict() if self.audit else None,
        }


__all__ = [
    "Decision",
    "DECISION_INCLUDED",
    "DECISION_COMPRESSED",
    "DECISION_DROPPED",
    "BlockDecision",
    "AuditReport",
    "OptimizationResult",
]
