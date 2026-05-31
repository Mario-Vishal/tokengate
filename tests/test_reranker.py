"""CP-018 tests: neural reranking stage (with FakeReranker)."""

from __future__ import annotations

import pytest

from contextpilot import ContextBlock
from contextpilot.models import FakeReranker
from contextpilot.ranking.reranker_stage import apply_rerank_relevance, rerank_blocks


def test_empty_returns_empty() -> None:
    assert rerank_blocks("q", [], FakeReranker()) == []


def test_sets_rerank_score_on_all_blocks() -> None:
    blocks = [ContextBlock(content="job search resume"), ContextBlock(content="weather")]
    rerank_blocks("job search", blocks, FakeReranker(), top_n=None)
    assert all(b.rerank_score is not None for b in blocks)


def test_orders_by_rerank_score() -> None:
    relevant = ContextBlock(content="recruiter action item send resume", block_id="rel")
    noise = ContextBlock(content="generic ai recruiting trends", block_id="noise")
    ranked = rerank_blocks("recruiter action items", [noise, relevant], FakeReranker())
    assert ranked[0].block_id == "rel"


def test_top_n_cutoff() -> None:
    blocks = [
        ContextBlock(content="job search resume cisco", block_id="a"),
        ContextBlock(content="job search", block_id="b"),
        ContextBlock(content="totally unrelated", block_id="c"),
    ]
    ranked = rerank_blocks("job search resume cisco", blocks, FakeReranker(), top_n=2)
    assert len(ranked) == 2
    assert "c" not in {b.block_id for b in ranked}  # weakest dropped


def test_top_n_none_keeps_all() -> None:
    blocks = [ContextBlock(content=f"doc {i} job", block_id=str(i)) for i in range(4)]
    assert len(rerank_blocks("job", blocks, FakeReranker(), top_n=None)) == 4


def test_top_n_larger_than_input() -> None:
    blocks = [ContextBlock(content="one job"), ContextBlock(content="two job")]
    assert len(rerank_blocks("job", blocks, FakeReranker(), top_n=10)) == 2


def test_apply_rerank_relevance_blends_into_final_score() -> None:
    a = ContextBlock(content="x", rerank_score=1.0, final_score=0.2, block_id="a")
    b = ContextBlock(content="y", rerank_score=0.0, final_score=0.2, block_id="b")
    apply_rerank_relevance([a, b], rerank_weight=0.7)
    # a: norm rerank 1.0 -> 0.7*1 + 0.3*0.2 = 0.76 ; b: norm 0 -> 0.3*0.2 = 0.06
    assert a.final_score == pytest.approx(0.76)
    assert b.final_score == pytest.approx(0.06)
    assert a.final_score > b.final_score


def test_apply_rerank_relevance_empty_noop() -> None:
    apply_rerank_relevance([])  # must not raise
