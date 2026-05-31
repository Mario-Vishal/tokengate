"""CP-009/CP-020 tests: embedding-based extractive compression (with FakeEmbeddingModel)."""

from __future__ import annotations

from contextpilot import ContextBlock, HeuristicTokenCounter
from contextpilot.compression.extractive import (
    compress_block,
    compress_text,
    score_sentences,
    split_sentences,
)
from contextpilot.models import FakeEmbeddingModel

_COUNTER = HeuristicTokenCounter()
_MODEL = FakeEmbeddingModel(dim=128)

_DOC = (
    "The quarterly budget report covers expenses. "
    "Our job search strategy focuses on resume keywords and networking. "
    "The cafeteria menu changes on Fridays. "
    "Remember to water the office plants weekly."
)


def test_split_sentences() -> None:
    assert len(split_sentences(_DOC)) == 4
    assert split_sentences("") == []


def test_score_sentences_returns_scores_and_vectors() -> None:
    sents = split_sentences(_DOC)
    scores, vecs = score_sentences("job search resume", sents, _MODEL)
    assert len(scores) == len(sents)
    assert vecs.shape == (len(sents), _MODEL.dim)
    # the job-search sentence should score highest
    assert scores[1] == max(scores)


def test_compress_text_stays_under_target_and_keeps_relevant() -> None:
    out = compress_text(_DOC, "job search resume", _MODEL, target_tokens=12, counter=_COUNTER)
    assert _COUNTER.count(out) <= _COUNTER.count(_DOC)
    assert "job search strategy" in out


def test_compress_never_empties() -> None:
    out = compress_text(_DOC, "xyzzy no match", _MODEL, target_tokens=1, counter=_COUNTER)
    assert out.strip() != ""


def test_compress_returns_original_when_already_fits() -> None:
    assert compress_text(_DOC, "job", _MODEL, target_tokens=10_000, counter=_COUNTER) == _DOC


def test_compress_single_sentence_unchanged() -> None:
    one = "This is a single long sentence that cannot be split any further at all"
    assert compress_text(one, "single", _MODEL, target_tokens=2, counter=_COUNTER) == one


def test_compress_block_respects_non_compressible() -> None:
    block = ContextBlock(content=_DOC, compressible=False)
    assert compress_block(block, "job search", _MODEL, target_tokens=5, counter=_COUNTER) is block


def test_compress_block_smaller_copy_with_metadata() -> None:
    block = ContextBlock(content=_DOC, block_id="orig")
    result = compress_block(block, "job search resume", _MODEL, target_tokens=12, counter=_COUNTER)
    assert result is not block
    assert result.token_count is not None and result.token_count <= _COUNTER.count(_DOC)
    assert result.metadata["compressed"] is True
    assert result.metadata["compression_method"] == "embedding_extractive"
    assert result.metadata["original_token_count"] == _COUNTER.count(_DOC)
    assert block.content == _DOC  # original untouched


def test_compress_block_unchanged_when_fits() -> None:
    block = ContextBlock(content=_DOC)
    assert compress_block(block, "job", _MODEL, target_tokens=10_000, counter=_COUNTER) is block
