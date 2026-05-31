"""Embedding-based extractive compression (CP-009, upgraded CP-020).

Shrinks a large block by keeping only its most query-relevant sentences — **never
rewriting them and never calling a generative LLM** (ADR-014), so it can't hallucinate
and adds no generation cost. Sentence scoring is neural + lexical:

    sentence_score = w_sem·semantic_sim(query, sentence) + w_kw·keyword_overlap
                   + entity_bonus + heading_bonus

Selection is greedy with a redundancy penalty (MMR-style) so the kept sentences are both
relevant and non-repetitive. Because we only ever drop whole original sentences, the
result is no larger than the input under a monotonic token counter.
"""

from __future__ import annotations

import re

import numpy as np

from contextpilot.budgeting.token_counter import TokenCounter, resolve_counter
from contextpilot.core.block import ContextBlock
from contextpilot.models.base import EmbeddingModel
from contextpilot.models.vectors import embed_query
from contextpilot.ranking.keyword_ranker import query_terms, tokenize
from contextpilot.ranking.semantic_scorer import cosine_scores

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_DEFAULT_SEMANTIC_WEIGHT = 0.6
_DEFAULT_KEYWORD_WEIGHT = 0.4
_ENTITY_BONUS = 0.1
_HEADING_BONUS = 0.1
_DEFAULT_REDUNDANCY_WEIGHT = 0.3


def split_sentences(text: str) -> list[str]:
    """Split text into trimmed, non-empty sentences."""
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _query_entity_terms(query: str) -> set[str]:
    """Query tokens that look like entities: capitalized or numeric (lowercased)."""
    terms: set[str] = set()
    for raw in query.split():
        clean = re.sub(r"[^A-Za-z0-9]", "", raw)
        if clean and (clean.isdigit() or any(c.isupper() for c in clean)):
            terms.add(clean.lower())
    return terms


def _is_heading(sentence: str) -> bool:
    stripped = sentence.strip()
    if stripped.startswith("#"):
        return True
    words = stripped.split()
    return len(words) <= 6 and not stripped.endswith((".", "!", "?"))


def score_sentences(
    query: str,
    sentences: list[str],
    model: EmbeddingModel,
    *,
    query_vec: np.ndarray | None = None,
    semantic_weight: float = _DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = _DEFAULT_KEYWORD_WEIGHT,
) -> tuple[list[float], np.ndarray]:
    """Score each sentence for relevance; also return the sentence embedding matrix."""
    if not sentences:
        return [], np.empty((0, model.dim), dtype=np.float32)

    qvec = query_vec if query_vec is not None else embed_query(query, model)
    sent_vecs = model.embed(sentences)
    semantic = cosine_scores(qvec, sent_vecs)

    qterms = query_terms(query)
    qentities = _query_entity_terms(query)

    scores: list[float] = []
    for i, sentence in enumerate(sentences):
        tokens = set(tokenize(sentence))
        keyword = len(qterms & tokens) / len(qterms) if qterms else 0.0
        entity = _ENTITY_BONUS if (qentities & tokens) else 0.0
        heading = _HEADING_BONUS if _is_heading(sentence) else 0.0
        scores.append(
            semantic_weight * float(semantic[i]) + keyword_weight * keyword
            + entity + heading
        )
    return scores, sent_vecs


def compress_text(
    content: str,
    query: str,
    model: EmbeddingModel,
    *,
    target_tokens: int,
    counter: TokenCounter | None = None,
    redundancy_weight: float = _DEFAULT_REDUNDANCY_WEIGHT,
) -> str:
    """Return a query-relevant excerpt of ``content`` at/under ``target_tokens``.

    Greedy MMR over sentences: each pick maximizes ``score − redundancy_weight·maxsim``
    to already-selected sentences. Keeps original sentence order in the output, always
    keeps ≥ 1 sentence, and returns the input unchanged if it already fits or has ≤ 1
    sentence.
    """
    counter = resolve_counter(counter)
    if target_tokens <= 0 or counter.count(content) <= target_tokens:
        return content

    sentences = split_sentences(content)
    if len(sentences) <= 1:
        return content

    scores, vecs = score_sentences(query, sentences, model)

    selected: list[int] = []
    selected_vecs: list[np.ndarray] = []
    used = 0
    remaining = set(range(len(sentences)))

    while remaining:
        best_i = -1
        best_eff = float("-inf")
        for i in remaining:
            redundancy = (
                max(float(np.dot(vecs[i], sv)) for sv in selected_vecs)
                if selected_vecs else 0.0
            )
            eff = scores[i] - redundancy_weight * redundancy
            if eff > best_eff:
                best_eff = eff
                best_i = i

        remaining.discard(best_i)
        cost = counter.count(sentences[best_i])
        if selected and used + cost > target_tokens:
            continue
        selected.append(best_i)
        selected_vecs.append(vecs[best_i])
        used += cost
        if used >= target_tokens:
            break

    if not selected:  # safety: keep the single best-scoring sentence
        selected = [max(range(len(sentences)), key=lambda i: scores[i])]

    return " ".join(sentences[i] for i in sorted(selected))


def compress_block(
    block: ContextBlock,
    query: str,
    model: EmbeddingModel,
    *,
    target_tokens: int,
    counter: TokenCounter | None = None,
) -> ContextBlock:
    """Return a compressed copy of ``block`` (or the block unchanged).

    Non-compressible blocks are returned as-is. When content shrinks, a copy is returned
    with a recomputed ``token_count`` and audit breadcrumbs in ``metadata``.
    """
    counter = resolve_counter(counter)
    if not block.compressible:
        return block

    original_tokens = block.ensure_token_count(counter)
    new_content = compress_text(
        block.content, query, model, target_tokens=target_tokens, counter=counter
    )
    if new_content == block.content:
        return block

    metadata = dict(block.metadata)
    metadata["compressed"] = True
    metadata["original_token_count"] = original_tokens
    metadata["compression_method"] = "embedding_extractive"
    return block.copy(
        content=new_content,
        token_count=counter.count(new_content),
        metadata=metadata,
    )


__all__ = ["split_sentences", "score_sentences", "compress_text", "compress_block"]
