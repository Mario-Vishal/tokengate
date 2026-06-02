"""CP-003 tests: TokenBlock validation, id derivation, serialization."""

from __future__ import annotations

import pytest

from tokengate import TokenBlock, InvalidBlockError


class _FakeCounter:
    """Minimal TokenCounter-shaped stub (word count)."""

    def count(self, text: str) -> int:
        return len(text.split())


# --- validation -----------------------------------------------------------

def test_empty_content_rejected() -> None:
    with pytest.raises(InvalidBlockError):
        TokenBlock(content="")
    with pytest.raises(InvalidBlockError):
        TokenBlock(content="   ")


@pytest.mark.parametrize("field", ["semantic_score", "keyword_score", "final_score"])
@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
def test_out_of_range_scores_rejected(field: str, bad: float) -> None:
    with pytest.raises(InvalidBlockError):
        TokenBlock(content="hi", **{field: bad})


def test_in_range_scores_accepted() -> None:
    b = TokenBlock(content="hi", semantic_score=0.0, keyword_score=1.0, final_score=0.5)
    assert b.semantic_score == 0.0


def test_negative_token_count_rejected() -> None:
    with pytest.raises(InvalidBlockError):
        TokenBlock(content="hi", token_count=-1)


# --- id derivation --------------------------------------------------------

def test_block_id_auto_derived_and_stable() -> None:
    a = TokenBlock(content="same content")
    b = TokenBlock(content="same content")
    assert a.block_id == b.block_id
    assert a.block_id.startswith("blk_")


def test_block_id_differs_by_content() -> None:
    assert TokenBlock(content="a").block_id != TokenBlock(content="b").block_id


def test_explicit_block_id_preserved() -> None:
    assert TokenBlock(content="x", block_id="my-id").block_id == "my-id"


# --- token counting -------------------------------------------------------

def test_ensure_token_count_uses_counter_and_caches() -> None:
    b = TokenBlock(content="one two three")
    assert b.token_count is None
    assert b.ensure_token_count(_FakeCounter()) == 3
    assert b.token_count == 3
    # idempotent: a different counter is ignored once cached
    assert b.ensure_token_count(_FakeCounter()) == 3


# --- serialization --------------------------------------------------------

def test_round_trip_dict() -> None:
    b = TokenBlock(
        content="hello world",
        source_id="file:1",
        metadata={"page": 2},
        semantic_score=0.7,
        required=True,
        cacheable=True,
        compressible=False,
    )
    restored = TokenBlock.from_dict(b.to_dict())
    assert restored.to_dict() == b.to_dict()


def test_from_dict_ignores_unknown_keys() -> None:
    b = TokenBlock.from_dict({"content": "hi", "bogus": 123})
    assert b.content == "hi"


def test_copy_with_overrides() -> None:
    b = TokenBlock(content="hi", block_id="id1")
    c = b.copy(final_score=0.9)
    assert c.block_id == "id1"
    assert c.final_score == 0.9
    assert b.final_score is None  # original untouched
