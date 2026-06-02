"""CP-002 tests: errors, hashing, logging."""

from __future__ import annotations

import logging

import tokengate
from tokengate.utils.errors import (
    BudgetError,
    TokenGateError,
    InvalidBlockError,
    OptimizationError,
)
from tokengate.utils.hashing import (
    block_id_from_content,
    hash_content,
    short_hash,
)
from tokengate.utils.logging import get_logger, log_event

# --- errors ---------------------------------------------------------------

def test_error_hierarchy() -> None:
    for exc in (InvalidBlockError, BudgetError, OptimizationError):
        assert issubclass(exc, TokenGateError)
    assert issubclass(TokenGateError, Exception)


def test_errors_exported_at_top_level() -> None:
    assert tokengate.TokenGateError is TokenGateError
    assert tokengate.InvalidBlockError is InvalidBlockError


# --- hashing --------------------------------------------------------------

def test_hash_content_deterministic() -> None:
    assert hash_content("hello") == hash_content("hello")
    assert len(hash_content("hello")) == 64  # full sha256 hex


def test_hash_content_differs_by_input() -> None:
    assert hash_content("hello") != hash_content("world")


def test_short_hash_length_and_clamp() -> None:
    assert len(short_hash("abc", 16)) == 16
    assert len(short_hash("abc", 0)) == 1     # clamped up
    assert len(short_hash("abc", 999)) == 64  # clamped down


def test_block_id_format_and_stability() -> None:
    a = block_id_from_content("same text")
    b = block_id_from_content("same text")
    assert a == b
    assert a.startswith("blk_")


# --- logging --------------------------------------------------------------

def test_get_logger_namespacing_and_nullhandler() -> None:
    logger = get_logger("ranking")
    assert logger.name == "tokengate.ranking"
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
    # idempotent: calling again doesn't stack handlers
    n_before = len(logger.handlers)
    get_logger("ranking")
    assert len(logger.handlers) == n_before


def test_log_event_attaches_structured_fields(caplog) -> None:
    logger = get_logger("test")
    logger.propagate = True  # let caplog capture despite NullHandler
    with caplog.at_level(logging.INFO, logger="tokengate.test"):
        log_event(logger, "thing_happened", duration_ms=12, status="ok")
    rec = next(r for r in caplog.records if r.getMessage() == "thing_happened")
    assert rec.event == "thing_happened"
    assert rec.fields == {"duration_ms": 12, "status": "ok"}
