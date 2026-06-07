"""CP-010 tests: greedy budgeter (fit, required-first, compress-to-fit, drops)."""

from __future__ import annotations

import pytest

from tokengate import HeuristicTokenCounter, TokenBlock
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


def test_line_aware_split_breaks_table_rows() -> None:
    from tokengate.compression.extractive import split_units
    blob = "Amount Due $154.61\nDue by 06/27/2026\nProvider   Pine River Electric"
    # sentence split keeps it as ~1 unit (no periods); line-aware splits rows + columns.
    assert len(split_units(blob, line_aware=False)) == 1
    assert len(split_units(blob, line_aware=True)) >= 3


def test_keep_data_lines_preserves_amounts_and_dates() -> None:
    from tokengate.compression.extractive import compress_text
    from tokengate.models import FakeEmbeddingModel
    doc = ("Customer service address and mailing preferences.\n"
           "Amount Due $154.61\nDue by 06/27/2026\n"
           "Please retain this notice for your records.")
    # query is about the irrelevant boilerplate, so relevance alone would drop the data lines.
    out = compress_text(doc, "mailing preferences address", FakeEmbeddingModel(dim=128),
                        keep_ratio=0.9, line_aware=True, keep_data_lines=True)
    assert "$154.61" in out
    assert "06/27/2026" in out


def test_compress_always_squeezes_blocks_that_already_fit() -> None:
    # A block that fits the budget as-is: legacy compress-to-fit leaves it verbatim, but
    # compress_always squeezes out off-query boilerplate up front (the main token lever).
    doc = (
        "The budget report covers expenses for the quarter in detail. "
        "Our job search strategy focuses on resume keywords and networking. "
        "The cafeteria menu changes every Friday without fail. "
        "Please water the office plants at least once per week."
    )
    full_tokens = _COUNTER.count(doc)
    budget = full_tokens + 100  # block fits comfortably as-is

    # compress_always=False (legacy): kept verbatim.
    verbatim = TokenBlock(content=doc, block_id="b", final_score=0.9, compressible=True)
    out_off = budget_blocks([verbatim], query="job search resume", budget_tokens=budget,
                            counter=_COUNTER, embedding_model=FakeEmbeddingModel(dim=128),
                            compress_always=False)
    assert [b.block_id for b in out_off.included] == ["b"]
    assert out_off.compressed == []

    # compress_always=True: same fitting block is compressed up front to save tokens.
    squeezable = TokenBlock(content=doc, block_id="b", final_score=0.9, compressible=True)
    out_on = budget_blocks([squeezable], query="job search resume", budget_tokens=budget,
                           counter=_COUNTER, embedding_model=FakeEmbeddingModel(dim=128),
                           compress_always=True)
    assert [b.block_id for b in out_on.compressed] == ["b"]
    assert out_on.included == []
    assert out_on.used_tokens < full_tokens
    assert "job search" in out_on.compressed[0].content


def test_injected_compressor_is_used() -> None:
    # The budgeter delegates compression to whatever Compressor is injected (CP-031).
    class HalvingCompressor:
        def compress_block(self, block, query, *, keep_ratio):
            half = block.content[: len(block.content) // 2]
            return block.copy(content=half, token_count=max(1, block.token_count // 2))

    doc = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    big = TokenBlock(content=doc, block_id="b", token_count=40, final_score=0.9, compressible=True)
    out = budget_blocks([big], query="q", budget_tokens=1000, counter=_COUNTER,
                        compress_always=True, compressor=HalvingCompressor())
    assert [b.block_id for b in out.compressed] == ["b"]
    assert out.decisions[0].decision == DECISION_COMPRESSED
    assert out.decisions[0].final_tokens < out.decisions[0].original_tokens


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
    # A small fitting block keeps *something* included so the fallback-top-1 rule (which
    # only fires when nothing is included) doesn't mask the drop we're asserting.
    doc = "one. two. three. four. five. six. seven. eight."
    keep = _block("keep", 3, score=0.95)
    nc = TokenBlock(content=doc, block_id="nc", final_score=0.9, compressible=False)
    budget = _COUNTER.count(doc) - 2  # fits "keep" but not "nc"
    out = budget_blocks([keep, nc], query="three", budget_tokens=budget, counter=_COUNTER)
    assert "nc" in [b.block_id for b in out.dropped]
    assert "nc" not in [b.block_id for b in out.compressed]
