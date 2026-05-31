"""CP-017 tests: semantic scoring + auxiliary signals + expanded hybrid ranking."""

from __future__ import annotations

import numpy as np

from contextpilot import ContextBlock
from contextpilot.models import FakeEmbeddingModel, embed_query, ensure_block_vectors
from contextpilot.ranking.hybrid_ranker import rank_blocks, weighted_average
from contextpilot.ranking.semantic_scorer import apply_semantic_scores, cosine_scores
from contextpilot.ranking.signals import (
    recency_scores,
    source_priority_scores,
    token_efficiency_scores,
)

# --- semantic scorer ------------------------------------------------------

def test_cosine_scores_empty() -> None:
    assert cosine_scores(np.zeros(4, dtype=np.float32), np.empty((0, 4))).shape == (0,)


def test_cosine_scores_in_unit_range_and_relevant_higher() -> None:
    model = FakeEmbeddingModel(dim=128)
    blocks = [
        ContextBlock(content="python fastapi job search resume"),
        ContextBlock(content="weather forecast sunny tomorrow"),
    ]
    mat = ensure_block_vectors(blocks, model)
    qvec = embed_query("job search resume", model)
    scores = cosine_scores(qvec, mat)
    assert scores.shape == (2,)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > scores[1]


def test_apply_semantic_scores_sets_field() -> None:
    model = FakeEmbeddingModel(dim=64)
    blocks = [ContextBlock(content="job search"), ContextBlock(content="grocery list")]
    mat = ensure_block_vectors(blocks, model)
    apply_semantic_scores(blocks, embed_query("job search", model), mat)
    assert all(b.semantic_score is not None for b in blocks)
    assert blocks[0].semantic_score >= blocks[1].semantic_score


# --- signals --------------------------------------------------------------

def test_recency_newer_scores_higher() -> None:
    blocks = [
        ContextBlock(content="old", metadata={"modified_time": "2020-01-01T00:00:00"}),
        ContextBlock(content="new", metadata={"modified_time": "2026-01-01T00:00:00"}),
    ]
    r = recency_scores(blocks)
    assert r[1] == 1.0 and r[0] == 0.0


def test_recency_none_when_missing_or_bad() -> None:
    blocks = [
        ContextBlock(content="a"),
        ContextBlock(content="b", metadata={"modified_time": "nope"}),
    ]
    assert recency_scores(blocks) == [None, None]


def test_recency_accepts_epoch_numbers() -> None:
    blocks = [
        ContextBlock(content="a", metadata={"modified_time": 1000}),
        ContextBlock(content="b", metadata={"modified_time": 2000}),
    ]
    assert recency_scores(blocks) == [0.0, 1.0]


def test_token_efficiency_smaller_is_higher() -> None:
    blocks = [
        ContextBlock(content="big", token_count=1000),
        ContextBlock(content="small", token_count=100),
    ]
    eff = token_efficiency_scores(blocks)
    assert eff[1] == 1.0 and eff[0] == 0.0


def test_token_efficiency_none_when_unset() -> None:
    assert token_efficiency_scores([ContextBlock(content="x")]) == [None]


def test_source_priority_lookup() -> None:
    blocks = [
        ContextBlock(content="a", source_id="downloads"),
        ContextBlock(content="b", source_id="desktop"),
        ContextBlock(content="c"),
    ]
    out = source_priority_scores(blocks, {"downloads": 0.9})
    assert out == [0.9, None, None]


def test_source_priority_empty_map() -> None:
    assert source_priority_scores([ContextBlock(content="a", source_id="x")], {}) == [None]


# --- weighted average + hybrid ranking ------------------------------------

def test_weighted_average_renormalizes() -> None:
    # missing signals simply absent -> average over the rest
    assert weighted_average([(0.45, 1.0), (0.25, 0.0)]) == (0.45 * 1.0) / 0.70


def test_weighted_average_zero_total() -> None:
    assert weighted_average([(0.0, 1.0)]) == 0.0


def test_hybrid_uses_all_signals_and_ranks() -> None:
    relevant = ContextBlock(
        content="cisco software engineer job search resume",
        semantic_score=0.9, source_id="downloads", token_count=120,
        metadata={"modified_time": "2026-05-01T00:00:00"}, block_id="rel",
    )
    noise = ContextBlock(
        content="grocery receipt total amount", semantic_score=0.1,
        source_id="trash", token_count=900,
        metadata={"modified_time": "2019-01-01T00:00:00"}, block_id="noise",
    )
    ranked = rank_blocks(
        "cisco job search resume", [noise, relevant],
        source_priorities={"downloads": 1.0, "trash": 0.0},
    )
    assert ranked[0].block_id == "rel"
    for b in ranked:
        assert 0.0 <= (b.final_score or 0) <= 1.0


def test_hybrid_keyword_only_when_no_other_signals() -> None:
    a = ContextBlock(content="job search help", block_id="a")
    b = ContextBlock(content="unrelated stuff", block_id="b")
    # no semantic/recency/source; token_count unset -> only keyword contributes
    ranked = rank_blocks("job search", [b, a])
    assert ranked[0].block_id == "a"
