"""CP-015 tests: model protocols, fakes (unit), and opt-in real BGE models."""

from __future__ import annotations

import os

import numpy as np
import pytest

from tokengate.models import (
    EmbeddingModel,
    FakeEmbeddingModel,
    FakeReranker,
    Reranker,
    resolve_device,
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # vectors are L2-normalized


# --- protocols ------------------------------------------------------------

def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeEmbeddingModel(), EmbeddingModel)
    assert isinstance(FakeReranker(), Reranker)


def test_resolve_device_modes() -> None:
    assert resolve_device("cpu_only") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}
    with pytest.raises(ValueError):
        resolve_device("bogus")


# --- fake embedder --------------------------------------------------------

def test_fake_embed_shape_and_norm() -> None:
    model = FakeEmbeddingModel(dim=32)
    vecs = model.embed(["hello world", "another text here"])
    assert vecs.shape == (2, 32)
    assert vecs.dtype == np.float32
    for v in vecs:
        assert np.isclose(np.linalg.norm(v), 1.0)


def test_fake_embed_empty() -> None:
    vecs = FakeEmbeddingModel(dim=16).embed([])
    assert vecs.shape == (0, 16)


def test_fake_embed_deterministic_across_instances() -> None:
    a = FakeEmbeddingModel(dim=32).embed(["job search resume"])
    b = FakeEmbeddingModel(dim=32).embed(["job search resume"])
    assert np.array_equal(a, b)


def test_fake_embed_similarity_reflects_overlap() -> None:
    m = FakeEmbeddingModel(dim=128)
    v = m.embed([
        "python fastapi postgres docker",       # 0
        "docker postgres python fastapi",        # 1: same words, reordered
        "weather forecast sunny tomorrow",       # 2: unrelated
    ])
    assert _cosine(v[0], v[1]) > _cosine(v[0], v[2])


# --- fake reranker --------------------------------------------------------

def test_fake_rerank_orders_by_overlap() -> None:
    scores = FakeReranker().rerank(
        "recruiter action items",
        ["send resume by friday to recruiter action", "ai recruiting trends article"],
    )
    assert scores[0] > scores[1]


def test_fake_rerank_empty_inputs() -> None:
    assert FakeReranker().rerank("q", []) == []
    assert FakeReranker().rerank("", ["text"]) == [0.0]


# --- opt-in real BGE models (download ~4.5GB; GPU if available) -----------

@pytest.mark.skipif(
    os.environ.get("CONTEXTPILOT_TEST_REAL_MODELS") != "1",
    reason="set CONTEXTPILOT_TEST_REAL_MODELS=1 to run real BGE model tests (downloads weights)",
)
def test_real_bge_embedder_and_reranker() -> None:
    from tokengate.models.bge import BGEM3Embedder, BGEReranker

    emb = BGEM3Embedder()
    assert emb.dim == 1024
    vecs = emb.embed(["a job application email", "a grocery shopping list"])
    assert vecs.shape == (2, 1024)

    rer = BGEReranker()
    scores = rer.rerank(
        "documents about job applications",
        ["Thank you for applying to the Software Engineer position", "Bananas, milk, eggs"],
    )
    assert scores[0] > scores[1]
