"""CP-021 tests: MMR diversity selection."""

from __future__ import annotations

from tokengate import TokenBlock
from tokengate.models import FakeEmbeddingModel, ensure_block_vectors
from tokengate.selection.mmr import mmr_select


def _vec(blocks: list[TokenBlock]) -> list[TokenBlock]:
    ensure_block_vectors(blocks, FakeEmbeddingModel(dim=128))
    return blocks


def test_empty() -> None:
    assert mmr_select([]) == []


def test_lambda_one_is_relevance_order() -> None:
    blocks = _vec([
        TokenBlock(content="a", rerank_score=0.2, block_id="a"),
        TokenBlock(content="b", rerank_score=0.9, block_id="b"),
        TokenBlock(content="c", rerank_score=0.5, block_id="c"),
    ])
    ranked = mmr_select(blocks, lambda_=1.0)
    assert [b.block_id for b in ranked] == ["b", "c", "a"]


def test_top_k_limits_output() -> None:
    blocks = _vec([TokenBlock(content=f"doc {i}", rerank_score=float(i)) for i in range(5)])
    assert len(mmr_select(blocks, lambda_=0.7, top_k=3)) == 3


def test_diversity_prefers_dissimilar_over_near_duplicate() -> None:
    # two near-identical high-relevance blocks + one distinct lower-relevance block
    # dup2 has HIGHER relevance than 'other', so pure-relevance order is dup1, dup2,
    # other. With diversity weighted (low lambda), MMR demotes the redundant dup2 below
    # the distinct 'other' -> dup1, other, dup2. This proves MMR != relevance ordering.
    dup1 = TokenBlock(
        content="python fastapi docker postgres", rerank_score=0.95, block_id="dup1"
    )
    dup2 = TokenBlock(
        content="docker postgres python fastapi", rerank_score=0.90, block_id="dup2"
    )
    other = TokenBlock(
        content="cisco recruiter interview schedule", rerank_score=0.70, block_id="other"
    )
    _vec([dup1, dup2, other])

    ranked = mmr_select([dup1, dup2, other], lambda_=0.3)
    assert [b.block_id for b in ranked] == ["dup1", "other", "dup2"]


def test_blocks_without_vectors_still_selectable() -> None:
    blocks = [
        TokenBlock(content="a", final_score=0.3, block_id="a"),
        TokenBlock(content="b", final_score=0.8, block_id="b"),
    ]
    ranked = mmr_select(blocks, lambda_=0.5)  # no vectors -> redundancy 0 -> relevance order
    assert ranked[0].block_id == "b"


def test_falls_back_to_final_score_when_no_rerank() -> None:
    blocks = _vec([
        TokenBlock(content="x", final_score=0.1, block_id="x"),
        TokenBlock(content="y", final_score=0.9, block_id="y"),
    ])
    ranked = mmr_select(blocks, lambda_=1.0)
    assert ranked[0].block_id == "y"
