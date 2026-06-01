"""Tests for the ContextPilot audit dashboard (CP-029)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextpilot import ContextBlock, ContextPilot
from contextpilot.dashboard import AuditStore
from contextpilot.models.fakes import FakeEmbeddingModel, FakeReranker


@pytest.fixture()
def store(tmp_path: Path) -> AuditStore:
    return AuditStore(tmp_path / "test.db")


@pytest.fixture()
def pilot() -> ContextPilot:
    return ContextPilot(
        max_prompt_tokens=512,
        strategy="balanced",
        embedding_model=FakeEmbeddingModel(dim=64),
        reranker=FakeReranker(),
    )


def _block(text: str, bid: str = "b") -> ContextBlock:
    return ContextBlock(content=text, block_id=bid, source_id="src", semantic_score=0.8)


# --- AuditStore ---

def test_record_and_retrieve(store: AuditStore, pilot: ContextPilot) -> None:
    result = pilot.optimize("what is python?", [_block("Python is a language", "b1")])
    qid = store.record("s1", "what is python?", result, config={"strategy": "balanced"})
    assert isinstance(qid, str) and len(qid) == 16

    detail = store.query_detail(qid)
    assert detail is not None
    assert detail["query_text"] == "what is python?"
    assert detail["session_id"] == "s1"
    assert isinstance(detail["audit"], dict)
    assert "final_prompt_tokens" in detail["audit"]
    assert detail["config"] == {"strategy": "balanced"}


def test_global_stats_empty(store: AuditStore) -> None:
    stats = store.global_stats()
    assert stats["total_sessions"] == 0
    assert stats["total_queries"] == 0


def test_global_stats_after_records(store: AuditStore, pilot: ContextPilot) -> None:
    for i in range(3):
        result = pilot.optimize(f"q{i}", [_block(f"block content {i}", f"b{i}")])
        store.record("sess-a", f"q{i}", result)
    stats = store.global_stats()
    assert stats["total_sessions"] == 1
    assert stats["total_queries"] == 3


def test_sessions_list(store: AuditStore, pilot: ContextPilot) -> None:
    result = pilot.optimize("hello", [_block("hi there", "b1")])
    store.record("sess-1", "hello", result)
    result2 = pilot.optimize("world", [_block("world content", "b2")])
    store.record("sess-2", "world", result2)
    sessions = store.sessions()
    assert len(sessions) == 2
    ids = {s["session_id"] for s in sessions}
    assert ids == {"sess-1", "sess-2"}


def test_session_queries(store: AuditStore, pilot: ContextPilot) -> None:
    for q in ["q1", "q2", "q3"]:
        result = pilot.optimize(q, [_block(f"content for {q}", "b")])
        store.record("sess-x", q, result)
    qs = store.session_queries("sess-x")
    assert len(qs) == 3
    assert [r["query_text"] for r in qs] == ["q1", "q2", "q3"]


def test_query_detail_not_found(store: AuditStore) -> None:
    assert store.query_detail("nonexistent") is None


def test_same_session_upserted(store: AuditStore, pilot: ContextPilot) -> None:
    """Multiple records to the same session update last_seen, don't duplicate the session."""
    for i in range(5):
        result = pilot.optimize(f"q{i}", [_block("x", "b")])
        store.record("sess-only", f"q{i}", result)
    sessions = store.sessions()
    assert len(sessions) == 1
    assert sessions[0]["query_count"] == 5


def test_audit_has_stages(store: AuditStore, pilot: ContextPilot) -> None:
    """Stored audit includes stage trace when trace=True (default)."""
    result = pilot.optimize("find python", [_block("Python docs", "b1"), _block("Java docs", "b2")])
    qid = store.record("s", "find python", result)
    detail = store.query_detail(qid)
    assert detail is not None
    audit = detail["audit"]
    # stages may be empty for trivial inputs but the key should be present
    assert "stages" in audit or "decisions" in audit


# --- FastAPI server (no uvicorn needed; uses TestClient) ---

def test_api_endpoints(store: AuditStore, pilot: ContextPilot) -> None:
    try:
        from fastapi.testclient import TestClient
        from contextpilot.dashboard.server import create_app
    except ImportError:
        pytest.skip("fastapi not installed (dashboard extra not present)")

    result = pilot.optimize("test q", [_block("test content", "b1")])
    qid = store.record("api-sess", "test q", result, config={"strategy": "balanced", "max_prompt_tokens": 512})

    client = TestClient(create_app(store))

    # stats
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_sessions"] == 1
    assert data["total_queries"] == 1

    # sessions
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    sessions = resp.json()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "api-sess"

    # session queries
    resp = client.get("/api/sessions/api-sess")
    assert resp.status_code == 200
    qs = resp.json()
    assert len(qs) == 1
    assert qs[0]["query_text"] == "test q"

    # query detail
    resp = client.get(f"/api/queries/{qid}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["query_id"] == qid
    assert "audit" in detail

    # 404
    resp = client.get("/api/queries/doesnotexist")
    assert resp.status_code == 404

    # index.html
    resp = client.get("/")
    assert resp.status_code == 200
    assert "ContextPilot" in resp.text
