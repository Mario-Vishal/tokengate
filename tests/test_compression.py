"""CP-009 tests: extractive compression."""

from __future__ import annotations

from contextpilot import ContextBlock, HeuristicTokenCounter
from contextpilot.compression.extractive import (
    compress_block,
    compress_text,
    split_sentences,
)

_COUNTER = HeuristicTokenCounter()

_DOC = (
    "The quarterly budget report covers expenses. "
    "Our job search strategy focuses on resume keywords and networking. "
    "The cafeteria menu changes on Fridays. "
    "Remember to water the office plants weekly."
)


def test_split_sentences() -> None:
    assert len(split_sentences(_DOC)) == 4
    assert split_sentences("") == []


def test_compress_text_stays_under_target() -> None:
    target = 12
    out = compress_text(_DOC, "job search resume", target_tokens=target, counter=_COUNTER)
    assert _COUNTER.count(out) <= max(
        target, _COUNTER.count("Our job search strategy focuses on resume keywords and networking.")
    )
    assert _COUNTER.count(out) <= _COUNTER.count(_DOC)


def test_compress_keeps_query_relevant_sentence() -> None:
    out = compress_text(_DOC, "job search resume", target_tokens=12, counter=_COUNTER)
    assert "job search strategy" in out


def test_compress_never_empties() -> None:
    out = compress_text(_DOC, "no matching terms here xyz", target_tokens=1, counter=_COUNTER)
    assert out.strip() != ""


def test_compress_returns_original_when_already_fits() -> None:
    out = compress_text(_DOC, "job", target_tokens=10_000, counter=_COUNTER)
    assert out == _DOC


def test_compress_single_sentence_unchanged() -> None:
    one = "This is a single long sentence that cannot be split any further at all"
    assert compress_text(one, "single", target_tokens=2, counter=_COUNTER) == one


def test_compress_block_respects_non_compressible() -> None:
    block = ContextBlock(content=_DOC, compressible=False)
    result = compress_block(block, "job search", target_tokens=5, counter=_COUNTER)
    assert result is block  # untouched


def test_compress_block_produces_smaller_copy_with_metadata() -> None:
    block = ContextBlock(content=_DOC, block_id="orig")
    result = compress_block(block, "job search resume", target_tokens=12, counter=_COUNTER)
    assert result is not block
    assert result.token_count is not None
    assert result.token_count <= _COUNTER.count(_DOC)
    assert result.metadata["compressed"] is True
    assert result.metadata["original_token_count"] == _COUNTER.count(_DOC)
    assert block.content == _DOC  # original untouched


def test_compress_block_unchanged_when_fits() -> None:
    block = ContextBlock(content=_DOC)
    result = compress_block(block, "job", target_tokens=10_000, counter=_COUNTER)
    assert result is block
