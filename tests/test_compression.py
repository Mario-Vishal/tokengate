"""CP-026 tests: relevance-driven extractive compression (no token target)."""

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
_MODEL = FakeEmbeddingModel(dim=256)

# One clearly-relevant sentence among off-topic boilerplate.
_DOC = (
    "Our job search strategy focuses on resume keywords and recruiter networking. "
    "The cafeteria menu changes on Fridays. "
    "Remember to water the office plants weekly. "
    "Equal opportunity employer statement and standard benefits boilerplate."
)
_ALL_RELEVANT = (
    "Send the updated resume to the recruiter. "
    "Schedule the recruiter screening call. "
    "Prepare resume talking points for the recruiter."
)


def test_split_sentences() -> None:
    assert len(split_sentences(_DOC)) == 4
    assert split_sentences("") == []


def test_score_sentences_shapes() -> None:
    sents = split_sentences(_DOC)
    scores, vecs = score_sentences("job search resume recruiter", sents, _MODEL)
    assert len(scores) == len(sents)
    assert vecs.shape == (len(sents), _MODEL.dim)


def test_drops_boilerplate_keeps_relevant() -> None:
    out = compress_text(_DOC, "job search resume recruiter networking", _MODEL)
    assert "job search strategy" in out
    assert "cafeteria" not in out and "plants" not in out
    assert _COUNTER.count(out) < _COUNTER.count(_DOC)


def test_all_relevant_block_kept_whole() -> None:
    # every sentence is on-topic -> nothing should be dropped (no forced target)
    out = compress_text(_ALL_RELEVANT, "resume recruiter call", _MODEL, keep_ratio=0.5)
    assert out == _ALL_RELEVANT


def test_single_sentence_unchanged() -> None:
    one = "This is a single sentence with no internal boundaries to split on"
    assert compress_text(one, "single", _MODEL) == one


def test_no_token_target_param() -> None:
    # compress_text must not require a target; this is the whole point of CP-026
    import inspect

    params = inspect.signature(compress_text).parameters
    assert "target_tokens" not in params
    assert "keep_ratio" in params


def test_compress_block_non_compressible_unchanged() -> None:
    block = ContextBlock(content=_DOC, compressible=False)
    assert compress_block(block, "job search", _MODEL) is block


def test_compress_block_metadata_and_copy() -> None:
    block = ContextBlock(content=_DOC, block_id="orig")
    result = compress_block(block, "job search resume recruiter", _MODEL, counter=_COUNTER)
    assert result is not block
    assert result.metadata["compressed"] is True
    assert result.metadata["compression_method"] == "embedding_extractive"
    assert result.metadata["original_token_count"] == _COUNTER.count(_DOC)
    assert result.token_count == _COUNTER.count(result.content)
    assert block.content == _DOC  # original untouched


def test_compress_block_unchanged_when_all_relevant() -> None:
    block = ContextBlock(content=_ALL_RELEVANT)
    assert compress_block(block, "resume recruiter call", _MODEL) is block


def test_within_block_sentence_dedup_collapses_repeats() -> None:
    # the same relevant sentence repeated 5x should collapse to one copy
    sentence = "The required skills include python fastapi and distributed systems."
    doc = (sentence + " ") * 5
    out = compress_text(doc, "required skills python", _MODEL, sentence_dedup_threshold=0.95)
    assert out.count("The required skills include") == 1


def test_distinct_relevant_sentences_not_deduped() -> None:
    out = compress_text(_ALL_RELEVANT, "resume recruiter call", _MODEL,
                        keep_ratio=0.5, sentence_dedup_threshold=0.95)
    # three distinct (non-identical) sentences -> none collapsed
    assert out == _ALL_RELEVANT
