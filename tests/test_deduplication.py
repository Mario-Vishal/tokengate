"""CP-007 tests: exact deduplication."""

from __future__ import annotations

from tokengate import TokenBlock
from tokengate.deduplication.exact import deduplicate_exact


def test_no_duplicates_passes_through() -> None:
    blocks = [TokenBlock(content="a"), TokenBlock(content="b")]
    kept, dropped = deduplicate_exact(blocks)
    assert len(kept) == 2
    assert dropped == []


def test_exact_duplicate_collapsed() -> None:
    blocks = [TokenBlock(content="same"), TokenBlock(content="same")]
    kept, dropped = deduplicate_exact(blocks)
    assert len(kept) == 1
    assert len(dropped) == 1


def test_different_whitespace_is_not_exact_duplicate() -> None:
    kept, dropped = deduplicate_exact(
        [TokenBlock(content="hello world"), TokenBlock(content="hello  world")]
    )
    assert len(kept) == 2
    assert dropped == []


def test_higher_score_representative_kept() -> None:
    low = TokenBlock(content="dup", semantic_score=0.2, block_id="low")
    high = TokenBlock(content="dup", semantic_score=0.9, block_id="high")
    kept, dropped = deduplicate_exact([low, high])
    assert len(kept) == 1
    assert kept[0].block_id == "high"
    assert dropped[0].block_id == "low"


def test_required_representative_kept_over_higher_score() -> None:
    req = TokenBlock(content="dup", required=True, semantic_score=0.1, block_id="req")
    hi = TokenBlock(content="dup", semantic_score=0.99, block_id="hi")
    kept, _ = deduplicate_exact([req, hi])
    assert kept[0].block_id == "req"


def test_first_occurrence_kept_on_tie_and_order_preserved() -> None:
    blocks = [
        TokenBlock(content="x", block_id="x1"),
        TokenBlock(content="y", block_id="y1"),
        TokenBlock(content="x", block_id="x2"),  # tie with x1
    ]
    kept, dropped = deduplicate_exact(blocks)
    assert [b.block_id for b in kept] == ["x1", "y1"]
    assert dropped[0].block_id == "x2"


def test_empty_input() -> None:
    assert deduplicate_exact([]) == ([], [])
