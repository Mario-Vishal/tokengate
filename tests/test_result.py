"""CP-004 tests: result/audit models + audit builder (counts, token math)."""

from __future__ import annotations

from contextpilot import AuditReport, BlockDecision, ContextBlock, OptimizationResult
from contextpilot.audit.audit_report import build_audit_report
from contextpilot.core.result import (
    DECISION_COMPRESSED,
    DECISION_DROPPED,
    DECISION_INCLUDED,
)


def _decisions() -> list[BlockDecision]:
    return [
        BlockDecision("a", DECISION_INCLUDED, "top score", 100, 100, 0.9),
        BlockDecision("b", DECISION_COMPRESSED, "large, compressible", 200, 80, 0.7),
        BlockDecision("c", DECISION_DROPPED, "over budget", 150, 0, 0.2),
    ]


def test_block_decision_serialization() -> None:
    d = BlockDecision("a", DECISION_INCLUDED, "r", 10, 10, 0.5)
    assert d.to_dict()["decision"] == "included"


def test_build_audit_counts_and_savings() -> None:
    audit = build_audit_report(
        total_candidate_blocks=3,
        total_candidate_tokens=450,
        final_prompt_tokens=180,
        decisions=_decisions(),
    )
    assert (audit.included_count, audit.compressed_count, audit.dropped_count) == (1, 1, 1)
    assert audit.tokens_saved == 270
    assert audit.tokens_saved_percent == 60.0


def test_percent_safe_when_no_candidate_tokens() -> None:
    audit = build_audit_report(
        total_candidate_blocks=0,
        total_candidate_tokens=0,
        final_prompt_tokens=0,
        decisions=[],
    )
    assert audit.tokens_saved_percent == 0.0
    assert audit.tokens_saved == 0


def test_negative_savings_reported_honestly() -> None:
    audit = build_audit_report(
        total_candidate_blocks=1,
        total_candidate_tokens=50,
        final_prompt_tokens=70,  # scaffolding outweighs trimming
        decisions=[BlockDecision("a", DECISION_INCLUDED, "r", 50, 50, 1.0)],
    )
    assert audit.tokens_saved == -20
    assert audit.tokens_saved_percent == -40.0


def test_audit_report_to_dict_roundtrip_shape() -> None:
    audit = build_audit_report(
        total_candidate_blocks=3,
        total_candidate_tokens=450,
        final_prompt_tokens=180,
        decisions=_decisions(),
    )
    d = audit.to_dict()
    assert isinstance(d["decisions"], list) and len(d["decisions"]) == 3
    assert set(d) == {
        "total_candidate_blocks",
        "total_candidate_tokens",
        "final_prompt_tokens",
        "tokens_saved",
        "tokens_saved_percent",
        "included_count",
        "compressed_count",
        "dropped_count",
        "decisions",
        "models_used",
    }


def test_optimization_result_to_dict() -> None:
    blk = ContextBlock(content="hello")
    audit = build_audit_report(
        total_candidate_blocks=1,
        total_candidate_tokens=10,
        final_prompt_tokens=10,
        decisions=[BlockDecision(blk.block_id, DECISION_INCLUDED, "r", 10, 10, 1.0)],
    )
    result = OptimizationResult(
        query="q",
        final_prompt="PROMPT",
        included_blocks=[blk],
        audit=audit,
    )
    d = result.to_dict()
    assert d["query"] == "q"
    assert d["final_prompt"] == "PROMPT"
    assert d["included_blocks"][0]["content"] == "hello"
    assert d["audit"]["included_count"] == 1
    assert isinstance(audit, AuditReport)
