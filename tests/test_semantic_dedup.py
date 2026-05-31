"""CP-019 tests: semantic (near-duplicate) deduplication."""

from __future__ import annotations

import numpy as np

from contextpilot import ContextBlock
from contextpilot.deduplication.semantic import deduplicate_semantic
from contextpilot.models import FakeEmbeddingModel, ensure_block_vectors


def _with_vectors(blocks: list[ContextBlock]) -> list[ContextBlock]:
    ensure_block_vectors(blocks, FakeEmbeddingModel(dim=128))
    return blocks


def test_empty() -> None:
    r = deduplicate_semantic([])
    assert r.kept == [] and r.dropped == []


def test_paraphrase_collapsed() -> None:
    # The fake embedder is bag-of-words, so a "paraphrase" is the same tokens reordered
    # (real BGE handles true synonyms). Same multiset -> identical vector -> cosine 1.0.
    blocks = _with_vectors([
        ContextBlock(content="requires python fastapi postgres docker", block_id="a"),
        ContextBlock(content="docker postgres fastapi python requires", block_id="b"),
    ])
    r = deduplicate_semantic(blocks, threshold=0.9)
    assert [b.block_id for b in r.kept] == ["a"]
    assert [b.block_id for b in r.dropped] == ["b"]
    assert r.duplicate_of["b"] == "a"
    assert r.similarity["b"] >= 0.9


def test_distinct_blocks_survive() -> None:
    blocks = _with_vectors([
        ContextBlock(content="cisco software engineer job description", block_id="a"),
        ContextBlock(content="grocery shopping list bananas milk", block_id="b"),
    ])
    r = deduplicate_semantic(blocks, threshold=0.9)
    assert len(r.kept) == 2
    assert r.dropped == []


def test_keeps_first_in_order_as_representative() -> None:
    # identical content -> identical vectors; first one (best-ranked) kept
    blocks = _with_vectors([
        ContextBlock(content="same meaning text", block_id="first"),
        ContextBlock(content="same meaning text", block_id="second"),
        ContextBlock(content="same meaning text", block_id="third"),
    ])
    r = deduplicate_semantic(blocks, threshold=0.95)
    assert [b.block_id for b in r.kept] == ["first"]
    assert {b.block_id for b in r.dropped} == {"second", "third"}
    assert r.duplicate_of["second"] == "first"
    assert r.duplicate_of["third"] == "first"


def test_blocks_without_vectors_are_kept() -> None:
    a = ContextBlock(content="x", block_id="a")  # no vector
    b = ContextBlock(content="y", block_id="b", vector=np.ones(8, dtype=np.float32))
    r = deduplicate_semantic([a, b], threshold=0.9)
    assert {blk.block_id for blk in r.kept} == {"a", "b"}


def test_threshold_one_only_drops_near_identical() -> None:
    blocks = _with_vectors([
        ContextBlock(content="alpha beta gamma delta", block_id="a"),
        ContextBlock(content="alpha beta gamma epsilon", block_id="b"),  # similar, not equal
    ])
    r = deduplicate_semantic(blocks, threshold=1.0)
    assert len(r.kept) == 2  # not identical enough at threshold 1.0
