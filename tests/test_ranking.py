"""CP-008 tests: keyword scoring, normalization, hybrid ranking."""

from __future__ import annotations

import pytest

from contextpilot import ContextBlock
from contextpilot.ranking.hybrid_ranker import combine_scores, rank_blocks
from contextpilot.ranking.keyword_ranker import (
    keyword_raw_score,
    keyword_raw_scores,
    query_terms,
    tokenize,
)
from contextpilot.ranking.score_normalizer import normalize_min_max

# --- keyword scoring ------------------------------------------------------

def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_keyword_raw_score_distinct_coverage() -> None:
    terms = query_terms("job search resume")
    block = ContextBlock(content="my job search and job hunt notes")  # 'job','search'
    assert keyword_raw_score(terms, block) == 2.0  # distinct: job, search


def test_keyword_raw_score_empty_query() -> None:
    assert keyword_raw_score(set(), ContextBlock(content="anything")) == 0.0


def test_keyword_raw_scores_order() -> None:
    blocks = [ContextBlock(content="job search"), ContextBlock(content="unrelated")]
    assert keyword_raw_scores("job search", blocks) == [2.0, 0.0]


# --- normalization --------------------------------------------------------

def test_normalize_empty() -> None:
    assert normalize_min_max([]) == []


def test_normalize_basic_range() -> None:
    assert normalize_min_max([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_normalize_all_equal_positive() -> None:
    assert normalize_min_max([3.0, 3.0]) == [1.0, 1.0]


def test_normalize_all_zero() -> None:
    assert normalize_min_max([0.0, 0.0]) == [0.0, 0.0]


# --- combine --------------------------------------------------------------

def test_combine_weighted_average() -> None:
    assert combine_scores(1.0, 0.0, 0.6, 0.4) == pytest.approx(0.6)
    assert combine_scores(0.0, 1.0, 0.6, 0.4) == pytest.approx(0.4)


def test_combine_zero_weights_safe() -> None:
    assert combine_scores(1.0, 1.0, 0.0, 0.0) == 0.0


# --- hybrid ranking -------------------------------------------------------

def test_rank_sets_scores_in_range() -> None:
    blocks = [
        ContextBlock(content="job search resume tips", semantic_score=0.8),
        ContextBlock(content="grocery list", semantic_score=0.1),
    ]
    ranked = rank_blocks("job search", blocks)
    for b in ranked:
        assert 0.0 <= (b.keyword_score or 0) <= 1.0
        assert 0.0 <= (b.final_score or 0) <= 1.0


def test_rank_orders_by_final_score() -> None:
    relevant = ContextBlock(content="job search resume", semantic_score=0.9, block_id="rel")
    irrelevant = ContextBlock(content="weather forecast", semantic_score=0.1, block_id="irr")
    ranked = rank_blocks("job search", [irrelevant, relevant])
    assert ranked[0].block_id == "rel"


def test_rank_keyword_only_when_no_semantic() -> None:
    a = ContextBlock(content="job search help", block_id="a")  # matches both terms
    b = ContextBlock(content="random text", block_id="b")
    ranked = rank_blocks("job search", [a, b], semantic_weight=0.0, keyword_weight=1.0)
    assert ranked[0].block_id == "a"
    assert ranked[0].final_score == pytest.approx(1.0)


def test_rank_empty() -> None:
    assert rank_blocks("q", []) == []


def test_rank_is_stable_on_ties() -> None:
    blocks = [
        ContextBlock(content="same words here", block_id="1", semantic_score=0.5),
        ContextBlock(content="same words here", block_id="2", semantic_score=0.5),
    ]
    ranked = rank_blocks("nomatch", blocks)
    assert [b.block_id for b in ranked] == ["1", "2"]
