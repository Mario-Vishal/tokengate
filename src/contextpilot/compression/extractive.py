"""Extractive (sentence-selection) compression (CP-009).

Shrinks a large block by keeping only its most query-relevant sentences, in their
original order, until a token target is met. Extractive (not abstractive) so it never
fabricates text and never needs an LLM — it only *drops* sentences, which guarantees
the result is no larger than the original under a monotonic token counter.
"""

from __future__ import annotations

import re

from contextpilot.budgeting.token_counter import TokenCounter, resolve_counter
from contextpilot.core.block import ContextBlock
from contextpilot.ranking.keyword_ranker import query_terms, tokenize

# Split on sentence-ending punctuation followed by whitespace.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into trimmed, non-empty sentences."""
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _sentence_score(terms: set[str], sentence: str) -> float:
    if not terms:
        return 0.0
    return float(len(terms & set(tokenize(sentence))))


def compress_text(
    content: str,
    query: str,
    *,
    target_tokens: int,
    counter: TokenCounter | None = None,
) -> str:
    """Return a query-relevant excerpt of ``content`` at or under ``target_tokens``.

    Sentences are ranked by query-term overlap; the highest-scoring are kept until the
    budget is reached, then re-emitted in original order. Always keeps at least one
    sentence (never empties). If the text already fits, or has <= 1 sentence, it is
    returned unchanged.
    """
    counter = resolve_counter(counter)
    if target_tokens <= 0 or counter.count(content) <= target_tokens:
        return content

    sentences = split_sentences(content)
    if len(sentences) <= 1:
        return content  # nothing to select between

    terms = query_terms(query)
    # Rank sentence indices by score (desc), stable by original order for ties.
    order = sorted(
        range(len(sentences)),
        key=lambda i: (_sentence_score(terms, sentences[i]), -i),
        reverse=True,
    )

    selected: list[int] = []
    used = 0
    for idx in order:
        cost = counter.count(sentences[idx])
        if selected and used + cost > target_tokens:
            continue
        selected.append(idx)
        used += cost
        if used >= target_tokens:
            break

    if not selected:  # safety: keep the single best sentence
        selected = [order[0]]

    return " ".join(sentences[i] for i in sorted(selected))


def compress_block(
    block: ContextBlock,
    query: str,
    *,
    target_tokens: int,
    counter: TokenCounter | None = None,
) -> ContextBlock:
    """Return a compressed copy of ``block``, or the block unchanged.

    Non-compressible blocks are returned as-is. When compression changes the content,
    a copy is returned with a recomputed ``token_count`` and audit breadcrumbs in
    ``metadata`` (``compressed``, ``original_token_count``).
    """
    counter = resolve_counter(counter)
    if not block.compressible:
        return block

    original_tokens = block.ensure_token_count(counter)
    new_content = compress_text(
        block.content, query, target_tokens=target_tokens, counter=counter
    )
    if new_content == block.content:
        return block

    metadata = dict(block.metadata)
    metadata["compressed"] = True
    metadata["original_token_count"] = original_tokens
    return block.copy(
        content=new_content,
        token_count=counter.count(new_content),
        metadata=metadata,
    )


__all__ = ["split_sentences", "compress_text", "compress_block"]
