"""CP-006 tests: token counting (heuristic default + optional tiktoken)."""

from __future__ import annotations

import importlib.util

import pytest

from contextpilot import HeuristicTokenCounter, TokenCounter
from contextpilot.budgeting.token_counter import TiktokenCounter, resolve_counter


def test_heuristic_satisfies_protocol() -> None:
    assert isinstance(HeuristicTokenCounter(), TokenCounter)


def test_empty_is_zero() -> None:
    assert HeuristicTokenCounter().count("") == 0


def test_positive_for_nonempty() -> None:
    assert HeuristicTokenCounter().count("hello world") > 0


def test_monotonic_with_length() -> None:
    c = HeuristicTokenCounter()
    short = c.count("a short bit of text")
    long = c.count("a short bit of text" * 10)
    assert long >= short


def test_word_floor_applies() -> None:
    # 5 single-char words -> char/4 = ceil(9/4)=3, but word floor = 5
    assert HeuristicTokenCounter().count("a b c d e") == 5


def test_resolve_counter_defaults_to_heuristic() -> None:
    assert isinstance(resolve_counter(None), HeuristicTokenCounter)


def test_resolve_counter_passes_through() -> None:
    c = HeuristicTokenCounter()
    assert resolve_counter(c) is c


@pytest.mark.skipif(
    importlib.util.find_spec("tiktoken") is not None,
    reason="tiktoken IS installed; this test covers the missing-dependency path",
)
def test_tiktoken_missing_raises_clear_error() -> None:
    with pytest.raises(ImportError, match="contextpilot\\[tiktoken\\]"):
        TiktokenCounter()


@pytest.mark.skipif(
    importlib.util.find_spec("tiktoken") is None,
    reason="tiktoken not installed; exact-count path unavailable",
)
def test_tiktoken_counts_when_available() -> None:
    counter = TiktokenCounter()
    assert counter.count("") == 0
    assert counter.count("hello world") > 0
