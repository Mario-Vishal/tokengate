"""CP-010 tests: greedy budgeter (fit, required-first, compress-to-fit, drops)."""

from __future__ import annotations

import pytest

from tokengate import TokenBlock, HeuristicTokenCounter
from tokengate.budgeting.budgeter import budget_blocks
from tokengate.core.result import (
    DECISION_COMPRESSED,
    DECISION_DROPPED,
    DECISION_INCLUDED,
)
from tokengate.models import FakeEmbeddingModel
from tokengate.utils.errors import BudgetError

_COUNTER = HeuristicTokenCounter()


def _block(bid: str, tokens: int, *, score: float, required=False, compressible=False):
    # Fixed token_count makes budgeting deterministic regardless of content.
    return TokenBlock(
        content=f"content for {bid}",
        block_id=bid,
        token_count=tokens,
        final_score=score,
        required=required,
        compressible=compressible,
    )


def test_non_positive_budget_raises() -> None:
    with pytest.raises(BudgetError):
        budget_blocks([], query="q", budget_tokens=0)


def test_all_fit_included() -> None:
    blocks = [_block("a", 30, score=0.9), _block("b", 30, score=0.8)]
    out = budget_blocks(blocks, query="q", budget_tokens=100, counter=_COUNTER)
    assert [b.block_id for b in out.included] == ["a", "b"]
    assert out.dropped == []
    assert out.used_tokens == 60


def test_over_budget_block_dropped() -> None:
    blocks = [_block("a", 60, score=0.9), _block("b", 60, score=0.8)]
    out = budget_blocks(blocks, query="q", budget_tokens=100, counter=_COUNTER)
    assert [b.block_id for b in out.included] == ["a"]
    assert [b.block_id for b in out.dropped] == ["b"]
    assert out.used_tokens == 60
    drop_decision = next(d for d in out.decisions if d.block_id == "b")
    assert drop_decision.decision == DECISION_DROPPED
    assert drop_decision.final_tokens == 0


def test_greedy_continues_after_drop() -> None:
    blocks = [
        _block("a", 60, score=0.9),
        _block("b", 60, score=0.8),   # won't fit -> dropped
        _block("c", 30, score=0.7),   # smaller, still fits
    ]
    out = budget_blocks(blocks, query="q", budget_tokens=100, counter=_COUNTER)
    assert [b.block_id for b in out.included] == ["a", "c"]
    assert [b.block_id for b in out.dropped] == ["b"]
    assert out.used_tokens == 90


def test_required_always_included_even_over_budget() -> None:
    blocks = [_block("req", 200, score=0.1, required=True)]
    out = budget_blocks(blocks, query="q", budget_tokens=100, counter=_COUNTER)
    assert [b.block_id for b in out.included] == ["req"]
    assert out.used_tokens == 200
    d = out.decisions[0]
    assert d.decision == DECISION_INCLUDED
    assert d.reason == "required block"


def test_budget_not_exceeded_by_optional_blocks() -> None:
    blocks = [_block(f"b{i}", 40, score=1.0 - i * 0.1) for i in range(5)]
    out = budget_blocks(blocks, query="q", budget_tokens=100, counter=_COUNTER)
    assert out.used_tokens <= 100


def test_compression_to_fit_included_as_compressed() -> None:
    doc = (
        "The budget report covers expenses for the quarter in detail. "
        "Our job search strategy focuses on resume keywords and networking. "
        "The cafeteria menu changes every Friday without fail. "
        "Please water the office plants at least once per week."
    )
    big = TokenBlock(content=doc, block_id="big", final_score=0.9, compressible=True)
    full_tokens = _COUNTER.count(doc)
    budget = full_tokens - 10  # forces it not to fit as-is

    out = budget_blocks([big], query="job search resume", budget_tokens=budget,
                        counter=_COUNTER, embedding_model=FakeEmbeddingModel(dim=128))
    assert [b.block_id for b in out.compressed] == ["big"]
    assert out.included == []
    assert out.used_tokens <= budget
    d = out.decisions[0]
    assert d.decision == DECISION_COMPRESSED
    assert d.final_tokens < d.original_tokens
    # the query-relevant sentence survived
    assert "job search" in out.compressed[0].content


def test_value_per_token_prefers_dense_blocks() -> None:
    # Budget fits two small high-value blocks OR one large block. Value-per-token should
    # pick the two small ones (total value 1.35 > 0.80), unlike a rank-by-score budgeter
    # which would take the single highest-score 'big' block first.
    big = _block("big", 400, score=0.80)     # density 0.0020
    s1 = _block("s1", 200, score=0.70)        # density 0.0035
    s2 = _block("s2", 200, score=0.65)        # density 0.00325
    out = budget_blocks([big, s1, s2], query="q", budget_tokens=400, counter=_COUNTER)
    kept = {b.block_id for b in out.included}
    assert kept == {"s1", "s2"}
    assert "big" in {b.block_id for b in out.dropped}
    assert out.used_tokens == 400


def test_relevance_floor_drops_low_score_blocks() -> None:
    hi = _block("hi", 20, score=0.9)
    lo = _block("lo", 10, score=0.05)   # below floor despite being cheap
    out = budget_blocks([hi, lo], query="q", budget_tokens=100, counter=_COUNTER,
                        relevance_floor=0.15)
    assert [b.block_id for b in out.included] == ["hi"]
    assert [b.block_id for b in out.dropped] == ["lo"]
    d = next(d for d in out.decisions if d.block_id == "lo")
    assert "relevance floor" in d.reason


def test_relevance_floor_zero_disabled() -> None:
    lo = _block("lo", 10, score=0.05)
    out = budget_blocks([lo], query="q", budget_tokens=100, counter=_COUNTER,
                        relevance_floor=0.0)
    assert [b.block_id for b in out.included] == ["lo"]


def test_non_compressible_over_budget_is_dropped_not_compressed() -> None:
    doc = "one. two. three. four. five. six. seven. eight."
    block = TokenBlock(content=doc, block_id="nc", final_score=0.9, compressible=False)
    budget = _COUNTER.count(doc) - 2
    out = budget_blocks([block], query="three", budget_tokens=budget, counter=_COUNTER)
    assert [b.block_id for b in out.dropped] == ["nc"]
    assert out.compressed == []
