"""Build an :class:`AuditReport` from pipeline state (CP-004).

The optimizer accumulates a :class:`BlockDecision` per candidate block, then calls
:func:`build_audit_report` to derive the counts and token-savings math. Kept separate
from the model so the arithmetic has one well-tested home.
"""

from __future__ import annotations

from contextpilot.core.result import (
    DECISION_COMPRESSED,
    DECISION_DROPPED,
    DECISION_INCLUDED,
    AuditReport,
    BlockDecision,
)


def _percent_saved(total_candidate_tokens: int, tokens_saved: int) -> float:
    """tokens_saved as a percentage of candidate tokens; 0.0 when nothing to save."""
    if total_candidate_tokens <= 0:
        return 0.0
    return round(tokens_saved / total_candidate_tokens * 100, 2)


def build_audit_report(
    *,
    total_candidate_blocks: int,
    total_candidate_tokens: int,
    final_prompt_tokens: int,
    decisions: list[BlockDecision],
    models_used: dict[str, str | None] | None = None,
) -> AuditReport:
    """Assemble a complete :class:`AuditReport`.

    Counts are derived from ``decisions``; ``tokens_saved`` is the difference between
    all candidate tokens and the final prompt tokens (may be negative if prompt
    scaffolding outweighs trimming — reported honestly).
    """
    included = sum(1 for d in decisions if d.decision == DECISION_INCLUDED)
    compressed = sum(1 for d in decisions if d.decision == DECISION_COMPRESSED)
    dropped = sum(1 for d in decisions if d.decision == DECISION_DROPPED)

    tokens_saved = total_candidate_tokens - final_prompt_tokens

    return AuditReport(
        total_candidate_blocks=total_candidate_blocks,
        total_candidate_tokens=total_candidate_tokens,
        final_prompt_tokens=final_prompt_tokens,
        tokens_saved=tokens_saved,
        tokens_saved_percent=_percent_saved(total_candidate_tokens, tokens_saved),
        included_count=included,
        compressed_count=compressed,
        dropped_count=dropped,
        decisions=list(decisions),
        models_used=dict(models_used) if models_used else {},
    )


__all__ = ["build_audit_report"]
