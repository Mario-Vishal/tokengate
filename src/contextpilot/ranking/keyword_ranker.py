"""Keyword-overlap scoring vs the query (CP-008).

Produces a *raw* lexical relevance score per block (how many distinct query terms it
mentions). Raw scores are normalized to ``[0, 1]`` by ``score_normalizer`` before being
combined with semantic scores in ``hybrid_ranker``.
"""

from __future__ import annotations

import re

from contextpilot.core.block import ContextBlock

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split into alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def query_terms(query: str) -> set[str]:
    """Distinct lexical terms in the query."""
    return set(tokenize(query))


def keyword_raw_score(terms: set[str], block: ContextBlock) -> float:
    """Number of distinct query terms that appear in the block's content.

    Distinct-coverage is robust to block length (it does not reward a long block for
    repeating one term). Returns 0.0 when the query has no terms.
    """
    if not terms:
        return 0.0
    block_tokens = set(tokenize(block.content))
    return float(len(terms & block_tokens))


def keyword_raw_scores(query: str, blocks: list[ContextBlock]) -> list[float]:
    """Raw keyword scores for each block, in input order."""
    terms = query_terms(query)
    return [keyword_raw_score(terms, b) for b in blocks]


__all__ = ["tokenize", "query_terms", "keyword_raw_score", "keyword_raw_scores"]
