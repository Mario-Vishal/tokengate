"""Result & audit models (CP-004).

These are the value objects ``TokenGate.optimize()`` returns. They are plain,
serializable dataclasses so applications (e.g. Beacon's AuditPage) can render or
persist them directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from tokengate.core.block import TokenBlock

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
    # Human-readable source name (e.g. filename) — shown in the dashboard instead of block_id.
    source_id: str | None = None
    # First 300 chars of the content that actually enters the prompt (compressed version for
    # compressed blocks, original for included/dropped). Lets the dashboard show what was sent
    # vs what was excluded without storing the full text in the audit.
    content_preview: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "decision": self.decision,
            "reason": self.reason,
            "original_tokens": self.original_tokens,
            "final_tokens": self.final_tokens,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "source_id": self.source_id,
            "content_preview": self.content_preview,
        }


@dataclass
class StageRecord:
    """The before/after footprint of one pipeline stage (CP-028).

    The funnel the optimizer walks — exact dedup, rerank, semantic dedup, MMR,
    budgeting — turned into one observable row per stage so applications can show
    *what each stage took in, what it emitted, what it dropped, and how long it took*.
    Token counts are summed over each stage's block list (``tokens_out`` reflects
    compression at the budgeting stage).
    """

    stage: str
    blocks_in: int
    blocks_out: int
    tokens_in: int
    tokens_out: int
    dropped: int
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "blocks_in": self.blocks_in,
            "blocks_out": self.blocks_out,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "dropped": self.dropped,
            "duration_ms": self.duration_ms,
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
    # Per-stage funnel (CP-028); empty when tracing is disabled.
    stages: list[StageRecord] = field(default_factory=list)

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
            "stages": [s.to_dict() for s in self.stages],
        }


@dataclass
class OptimizationResult:
    """What ``optimize()`` returns: the prompt plus the full block breakdown."""

    query: str
    final_prompt: str
    included_blocks: list[TokenBlock] = field(default_factory=list)
    compressed_blocks: list[TokenBlock] = field(default_factory=list)
    dropped_blocks: list[TokenBlock] = field(default_factory=list)
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
    "StageRecord",
    "AuditReport",
    "OptimizationResult",
]
