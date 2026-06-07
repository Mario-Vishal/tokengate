"""CP-030 tests: per-query adaptive relevance cutoff (relevance-cliff detection)."""

from __future__ import annotations

from tokengate.core.block import TokenBlock
from tokengate.selection.adaptive_cutoff import (
    adaptive_cutoff_count,
    select_by_adaptive_cutoff,
)


def _block(bid: str, score: float) -> TokenBlock:
    return TokenBlock(content=f"c {bid}", block_id=bid, token_count=10, final_score=score)


def test_focused_query_keeps_one() -> None:
    # One dominant block, then a sharp cliff → keep only the top.
    assert adaptive_cutoff_count([0.9, 0.05, 0.04, 0.03]) == 1


def test_synthesis_query_keeps_the_plateau() -> None:
    # Five similar blocks then a cliff → keep all five.
    scores = [0.07, 0.065, 0.06, 0.05, 0.04, 0.013, 0.012, 0.005]
    assert adaptive_cutoff_count(scores) == 5


def test_all_equal_keeps_all() -> None:
    # No cliff anywhere → every block is equally relevant.
    assert adaptive_cutoff_count([0.4, 0.4, 0.4, 0.4]) == 4


def test_gentle_decline_keeps_most() -> None:
    # No sharp cliff (uniform gaps); the depth term keeps nearly all — only the single
    # weakest tail block is trimmed (tied max-gaps resolve to the deepest cut).
    assert adaptive_cutoff_count([0.5, 0.48, 0.46, 0.44, 0.42]) == 4


def test_too_few_blocks_keeps_all() -> None:
    assert adaptive_cutoff_count([0.9, 0.1]) == 2
    assert adaptive_cutoff_count([0.9]) == 1
    assert adaptive_cutoff_count([]) == 0


def test_min_keep_is_respected() -> None:
    # Sharp cliff would keep 1, but min_keep forces at least 3.
    assert adaptive_cutoff_count([0.9, 0.05, 0.04, 0.03, 0.02], min_keep=3) == 3


def test_select_preserves_input_order() -> None:
    # Cutoff is computed on score order, but kept/dropped keep the original list order.
    blocks = [_block("a", 0.05), _block("b", 0.9), _block("c", 0.04), _block("d", 0.03)]
    kept, dropped = select_by_adaptive_cutoff(blocks)
    assert [b.block_id for b in kept] == ["b"]          # only the dominant block
    assert [b.block_id for b in dropped] == ["a", "c", "d"]  # original order preserved


def test_select_falls_back_to_rerank_score() -> None:
    # final_score absent → cutoff uses rerank_score.
    b1 = TokenBlock(content="x", block_id="hi", token_count=5, rerank_score=0.9)
    b2 = TokenBlock(content="y", block_id="lo1", token_count=5, rerank_score=0.05)
    b3 = TokenBlock(content="z", block_id="lo2", token_count=5, rerank_score=0.04)
    kept, dropped = select_by_adaptive_cutoff([b1, b2, b3])
    assert [b.block_id for b in kept] == ["hi"]
