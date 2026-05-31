"""CP-012/CP-023 tests: ContextPilot.optimize() full neural pipeline (with fakes)."""

from __future__ import annotations

import pytest

from contextpilot import (
    ContextBlock,
    ContextPilot,
    InvalidBlockError,
    OptimizationResult,
)
from contextpilot.models import FakeEmbeddingModel, FakeReranker


def _pilot(**kwargs: object) -> ContextPilot:
    """ContextPilot wired with deterministic fakes (no model downloads)."""
    return ContextPilot(
        embedding_model=FakeEmbeddingModel(dim=128),
        reranker=FakeReranker(),
        **kwargs,  # type: ignore[arg-type]
    )


def test_empty_blocks_returns_query_only_prompt() -> None:
    result = _pilot(max_prompt_tokens=1000).optimize("what is X?", [])
    assert isinstance(result, OptimizationResult)
    assert result.final_prompt == "Question: what is X?"
    assert result.audit.total_candidate_blocks == 0


def test_end_to_end_includes_relevant_drops_irrelevant() -> None:
    blocks = [
        ContextBlock(content="System: answer concisely.", block_type="system",
                     required=True, cacheable=True, compressible=False, token_count=8),
        ContextBlock(content="Resume notes about the 2026 job search and resume tips.",
                     source_id="file:resume.pdf", token_count=12),
        ContextBlock(content="Unrelated grocery list bananas milk eggs bread",
                     source_id="file:notes.txt", token_count=12, compressible=False),
    ]
    result = _pilot(max_prompt_tokens=30).optimize("job search resume", blocks)
    included = {b.block_id for b in result.included_blocks}
    dropped = {b.block_id for b in result.dropped_blocks}
    assert blocks[0].block_id in included          # required system kept
    assert blocks[1].block_id in included           # relevant resume kept
    assert blocks[2].block_id in dropped            # irrelevant dropped
    assert "System: answer concisely." in result.final_prompt


def test_audit_counts_reconcile_with_candidates() -> None:
    blocks = [
        ContextBlock(content=f"doc number {i} about topic alpha beta", token_count=20)
        for i in range(6)
    ]
    result = _pilot(max_prompt_tokens=60).optimize("topic alpha", blocks)
    a = result.audit
    assert a.total_candidate_blocks == 6
    assert a.included_count + a.compressed_count + a.dropped_count == 6
    assert len(a.decisions) == 6


def test_exact_duplicates_dropped_and_audited() -> None:
    blocks = [
        ContextBlock(content="identical text here", token_count=10),
        ContextBlock(content="identical text here", token_count=10),
    ]
    result = _pilot(max_prompt_tokens=1000).optimize("text", blocks)
    assert len(result.included_blocks) == 1
    assert any(d.reason == "exact duplicate" for d in result.audit.decisions)


def test_required_kept_even_when_over_budget() -> None:
    blocks = [ContextBlock(content="huge required system block", required=True, token_count=500)]
    result = _pilot(max_prompt_tokens=50).optimize("q", blocks)
    assert len(result.included_blocks) == 1
    assert result.dropped_blocks == []


def test_optional_blocks_respect_budget() -> None:
    blocks = [ContextBlock(content=f"chunk topic {i}", token_count=40) for i in range(5)]
    pilot = _pilot(max_prompt_tokens=100)
    result = pilot.optimize("chunk topic", blocks)
    in_prompt = result.included_blocks + result.compressed_blocks
    assert sum(b.token_count or 0 for b in in_prompt) <= pilot.config.effective_budget


def test_audit_records_models_used() -> None:
    result = _pilot(max_prompt_tokens=200).optimize(
        "hello", [ContextBlock(content="hello world there")]
    )
    mu = result.audit.models_used
    assert mu["embedding_model"] == "FakeEmbeddingModel"
    assert mu["reranker"] == "FakeReranker"
    assert mu["final_llm"] is None


def test_decisions_carry_rerank_score() -> None:
    result = _pilot(max_prompt_tokens=500).optimize(
        "job search", [ContextBlock(content="job search resume content here")]
    )
    inc = next(d for d in result.audit.decisions if d.decision == "included")
    assert inc.rerank_score is not None


def test_rerank_cutoff_drops_recorded() -> None:
    blocks = [ContextBlock(content=f"doc {i} job search", token_count=10) for i in range(8)]
    result = _pilot(max_prompt_tokens=1000, strategy="speed").optimize("job search", blocks)
    # speed preset rerank_top_n=5 -> at least 3 dropped below cutoff
    cutoff = [d for d in result.audit.decisions if d.reason == "below rerank cutoff"]
    assert len(cutoff) >= 3


def test_invalid_query_type_raises() -> None:
    with pytest.raises(InvalidBlockError):
        _pilot().optimize(123, [])  # type: ignore[arg-type]


def test_non_block_in_list_raises() -> None:
    with pytest.raises(InvalidBlockError):
        _pilot().optimize("q", ["not a block"])  # type: ignore[list-item]


def test_result_is_serializable() -> None:
    result = _pilot(max_prompt_tokens=200).optimize(
        "hello", [ContextBlock(content="hello world")]
    )
    d = result.to_dict()
    assert d["query"] == "hello"
    assert d["audit"]["total_candidate_blocks"] == 1
    assert "models_used" in d["audit"]


def test_deterministic_output() -> None:
    make = lambda: [  # noqa: E731
        ContextBlock(content="job search resume tips", token_count=10),
        ContextBlock(content="weather report today sunny", token_count=10),
    ]
    r1 = _pilot(max_prompt_tokens=200).optimize("job search", make())
    r2 = _pilot(max_prompt_tokens=200).optimize("job search", make())
    assert r1.final_prompt == r2.final_prompt


def test_strategy_presets_differ() -> None:
    from contextpilot import OptimizerConfig

    speed = OptimizerConfig.for_strategy("speed")
    quality = OptimizerConfig.for_strategy("quality")
    assert speed.enable_mmr is False
    assert quality.enable_mmr is True
    assert quality.rerank_top_n > speed.rerank_top_n
