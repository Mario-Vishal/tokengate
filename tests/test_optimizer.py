"""CP-012 tests: ContextPilot.optimize() end-to-end + edge cases."""

from __future__ import annotations

import pytest

from contextpilot import (
    ContextBlock,
    ContextPilot,
    InvalidBlockError,
    OptimizationResult,
)


def test_empty_blocks_returns_query_only_prompt() -> None:
    pilot = ContextPilot(max_prompt_tokens=1000)
    result = pilot.optimize("what is X?", [])
    assert isinstance(result, OptimizationResult)
    assert result.final_prompt == "Question: what is X?"
    assert result.audit.total_candidate_blocks == 0
    assert result.audit.tokens_saved_percent == 0.0


def test_end_to_end_includes_relevant_drops_irrelevant() -> None:
    blocks = [
        ContextBlock(content="System: answer concisely.", block_type="system",
                     required=True, cacheable=True, compressible=False, token_count=8),
        ContextBlock(content="Resume mentions Python FastAPI and a 2026 job search.",
                     source_id="file:resume.pdf", semantic_score=0.82, token_count=12),
        ContextBlock(content="Unrelated grocery list bananas milk eggs bread",
                     source_id="file:notes.txt", semantic_score=0.10, token_count=12,
                     compressible=False),
    ]
    pilot = ContextPilot(max_prompt_tokens=30)
    result = pilot.optimize("Find documents related to job search", blocks)

    ids_included = {b.block_id for b in result.included_blocks}
    ids_dropped = {b.block_id for b in result.dropped_blocks}
    assert blocks[0].block_id in ids_included  # required system
    assert blocks[1].block_id in ids_included  # relevant resume
    assert blocks[2].block_id in ids_dropped   # irrelevant, didn't fit
    assert "System: answer concisely." in result.final_prompt
    assert "job search" in result.final_prompt


def test_audit_counts_reconcile_with_candidates() -> None:
    blocks = [
        ContextBlock(content=f"doc number {i} about topic {i}", semantic_score=0.5,
                     token_count=20)
        for i in range(6)
    ]
    pilot = ContextPilot(max_prompt_tokens=50)
    result = pilot.optimize("topic", blocks)
    a = result.audit
    assert a.total_candidate_blocks == 6
    assert a.included_count + a.compressed_count + a.dropped_count == 6
    assert len(a.decisions) == 6


def test_exact_duplicates_dropped_and_audited() -> None:
    blocks = [
        ContextBlock(content="identical text", semantic_score=0.9, token_count=10),
        ContextBlock(content="identical text", semantic_score=0.9, token_count=10),
    ]
    pilot = ContextPilot(max_prompt_tokens=1000)
    result = pilot.optimize("text", blocks)
    assert len(result.included_blocks) == 1
    assert len(result.dropped_blocks) == 1
    dup = next(d for d in result.audit.decisions if d.decision == "dropped")
    assert dup.reason == "exact duplicate"


def test_optional_blocks_respect_budget() -> None:
    # No required blocks: total in-prompt block tokens must stay within effective budget.
    blocks = [
        ContextBlock(content=f"chunk {i}", semantic_score=0.9 - i * 0.1, token_count=40)
        for i in range(5)
    ]
    pilot = ContextPilot(max_prompt_tokens=100)  # effective = 95
    result = pilot.optimize("chunk", blocks)
    in_prompt = result.included_blocks + result.compressed_blocks
    total = sum(b.token_count or 0 for b in in_prompt)
    assert total <= pilot.config.effective_budget


def test_required_kept_even_when_over_budget() -> None:
    blocks = [
        ContextBlock(content="huge required system block", required=True, token_count=500),
    ]
    pilot = ContextPilot(max_prompt_tokens=50)
    result = pilot.optimize("q", blocks)
    assert len(result.included_blocks) == 1
    assert result.dropped_blocks == []


def test_tokens_saved_positive_when_dropping() -> None:
    blocks = [
        ContextBlock(content="relevant job search resume content here",
                     semantic_score=0.9, token_count=15),
        ContextBlock(content="irrelevant filler text " * 20, semantic_score=0.05,
                     token_count=120, compressible=False),
    ]
    pilot = ContextPilot(max_prompt_tokens=40)
    result = pilot.optimize("job search", blocks)
    assert result.audit.tokens_saved > 0


def test_invalid_query_type_raises() -> None:
    pilot = ContextPilot()
    with pytest.raises(InvalidBlockError):
        pilot.optimize(123, [])  # type: ignore[arg-type]


def test_non_block_in_list_raises() -> None:
    pilot = ContextPilot()
    with pytest.raises(InvalidBlockError):
        pilot.optimize("q", ["not a block"])  # type: ignore[list-item]


def test_result_is_serializable() -> None:
    blocks = [ContextBlock(content="hello world", semantic_score=0.5)]
    result = ContextPilot(max_prompt_tokens=100).optimize("hello", blocks)
    d = result.to_dict()
    assert d["query"] == "hello"
    assert "final_prompt" in d
    assert d["audit"]["total_candidate_blocks"] == 1


def test_deterministic_output() -> None:
    blocks = [
        ContextBlock(content="job search resume tips", semantic_score=0.8),
        ContextBlock(content="weather report today", semantic_score=0.2),
    ]
    pilot = ContextPilot(max_prompt_tokens=200)
    r1 = pilot.optimize("job search", blocks)
    r2 = ContextPilot(max_prompt_tokens=200).optimize(
        "job search",
        [ContextBlock(content="job search resume tips", semantic_score=0.8),
         ContextBlock(content="weather report today", semantic_score=0.2)],
    )
    assert r1.final_prompt == r2.final_prompt
