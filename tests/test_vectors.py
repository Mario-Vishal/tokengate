"""CP-016 tests: TokenBlock.vector + reuse-or-compute resolution."""

from __future__ import annotations

import numpy as np
import pytest

from tokengate import InvalidBlockError, TokenBlock
from tokengate.models import FakeEmbeddingModel, embed_query, ensure_block_vectors

# --- TokenBlock.vector field -------------------------------------------

def test_vector_defaults_none() -> None:
    assert TokenBlock(content="hi").vector is None


def test_vector_coerced_to_float32_1d() -> None:
    b = TokenBlock(content="hi", vector=[0.1, 0.2, 0.3])
    assert isinstance(b.vector, np.ndarray)
    assert b.vector.dtype == np.float32
    assert b.vector.shape == (3,)


def test_vector_rejects_2d_and_empty() -> None:
    with pytest.raises(InvalidBlockError):
        TokenBlock(content="hi", vector=[[1.0, 2.0]])
    with pytest.raises(InvalidBlockError):
        TokenBlock(content="hi", vector=[])


def test_vector_serializes_as_list_and_roundtrips() -> None:
    b = TokenBlock(content="hi", vector=np.array([1.0, 2.0], dtype=np.float32))
    d = b.to_dict()
    assert d["vector"] == [1.0, 2.0]
    restored = TokenBlock.from_dict(d)
    assert np.array_equal(restored.vector, b.vector)


def test_copy_preserves_vector() -> None:
    b = TokenBlock(content="hi", vector=[1.0, 2.0, 3.0])
    c = b.copy(final_score=0.5)
    assert np.array_equal(c.vector, b.vector)


# --- ensure_block_vectors -------------------------------------------------

def test_empty_blocks_returns_empty_matrix() -> None:
    model = FakeEmbeddingModel(dim=16)
    out = ensure_block_vectors([], model)
    assert out.shape == (0, 16)


def test_computes_missing_vectors_and_caches() -> None:
    model = FakeEmbeddingModel(dim=16)
    blocks = [TokenBlock(content="job search"), TokenBlock(content="resume tips")]
    out = ensure_block_vectors(blocks, model)
    assert out.shape == (2, 16)
    assert all(b.vector is not None for b in blocks)
    # cached back, normalized
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_reuses_present_vectors() -> None:
    model = FakeEmbeddingModel(dim=16)
    preset = np.ones(16, dtype=np.float32)
    block = TokenBlock(content="anything", vector=preset.copy())
    out = ensure_block_vectors([block], model)
    assert np.array_equal(out[0], preset)  # untouched


def test_recomputes_on_dim_mismatch() -> None:
    model = FakeEmbeddingModel(dim=16)
    block = TokenBlock(content="hello", vector=[1.0, 2.0, 3.0])  # dim 3 != 16
    out = ensure_block_vectors([block], model)
    assert out.shape == (1, 16)
    assert block.vector.shape == (16,)  # replaced


def test_mixed_reuse_and_compute_order_preserved() -> None:
    model = FakeEmbeddingModel(dim=16)
    has_vec = TokenBlock(content="x", vector=np.full(16, 0.25, dtype=np.float32))
    no_vec = TokenBlock(content="job search resume")
    out = ensure_block_vectors([has_vec, no_vec], model)
    assert np.array_equal(out[0], np.full(16, 0.25, dtype=np.float32))
    assert out[1].shape == (16,)


def test_embed_query_returns_1d() -> None:
    model = FakeEmbeddingModel(dim=16)
    q = embed_query("find my job search files", model)
    assert q.shape == (16,)
