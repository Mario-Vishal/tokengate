"""Embedding-based extractive compression (CP-009, CP-020, relevance-driven CP-026).

Shrinks a block by keeping only its query-relevant sentences and dropping boilerplate —
**never rewriting them and never calling a generative LLM** (ADR-014). Sentence scoring is
neural + lexical:

    sentence_score = w_sem·semantic_sim(query, sentence) + w_kw·keyword_overlap
                   + entity_bonus + heading_bonus

Compression is **relevance-driven, not target-driven** (ADR-019): a sentence is kept when
its score is at least ``keep_ratio × best_sentence_score`` (a gate relative to the block's
own best sentence). There is **no token target** — an all-relevant block keeps every
sentence; a boilerplate-heavy block keeps only the on-topic ones. The caller (budgeter)
treats the token budget as a hard bound: if the pruned block still doesn't fit, it is
dropped, not truncated.
"""

from __future__ import annotations

import re

import numpy as np

from tokengate.budgeting.token_counter import TokenCounter, resolve_counter
from tokengate.core.block import TokenBlock
from tokengate.models.base import EmbeddingModel
from tokengate.models.vectors import embed_query
from tokengate.ranking.keyword_ranker import query_terms, tokenize
from tokengate.ranking.semantic_scorer import cosine_scores

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# Line-aware split: newlines, runs of 2+ spaces (PDF table columns), or sentence ends — so
# table/bill rows become their own units where sentence-splitting can't break them apart.
_LINE_SPLIT_RE = re.compile(r"\n+|\s{2,}|(?<=[.!?])\s+")
# A "data" unit carries a number, currency, percent, date-ish, or email — the literal facts
# (amounts, dates) that relevance scoring tends to drop because they don't resemble the query.
_DATA_RE = re.compile(r"\d|[$£€%@]")
_DEFAULT_SEMANTIC_WEIGHT = 0.6
_DEFAULT_KEYWORD_WEIGHT = 0.4
_ENTITY_BONUS = 0.1
_HEADING_BONUS = 0.1
_DEFAULT_KEEP_RATIO = 0.5
_DEFAULT_SENTENCE_DEDUP_THRESHOLD = 0.95


def split_sentences(text: str) -> list[str]:
    """Split text into trimmed, non-empty sentences."""
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def split_units(text: str, *, line_aware: bool = False) -> list[str]:
    """Split into scoring units — sentences, or (line_aware) lines/rows + sentences."""
    rx = _LINE_SPLIT_RE if line_aware else _SENTENCE_RE
    return [s.strip() for s in rx.split(text.strip()) if s.strip()]


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


def _dedup_sentences(
    kept: list[int], vecs: np.ndarray, threshold: float
) -> list[int]:
    """Drop later sentences that are ≥ ``threshold`` cosine to an already-kept one."""
    final: list[int] = []
    for i in kept:
        if any(float(np.dot(vecs[i], vecs[j])) >= threshold for j in final):
            continue
        final.append(i)
    return final


def compress_text(
    content: str,
    query: str,
    model: EmbeddingModel,
    *,
    keep_ratio: float = _DEFAULT_KEEP_RATIO,
    sentence_dedup_threshold: float = _DEFAULT_SENTENCE_DEDUP_THRESHOLD,
    line_aware: bool = False,
    keep_data_lines: bool = False,
) -> str:
    """Return ``content`` with low-relevance and duplicate units dropped.

    Keeps every unit whose score is ≥ ``keep_ratio × best_unit_score``, then collapses
    near-identical kept units (cosine ≥ ``sentence_dedup_threshold``) to a single copy —
    preserving original order. **No token target** (ADR-019). Returns the input unchanged
    when it has ≤ 1 unit or when nothing would be dropped.

    ``line_aware`` splits on lines/table-rows (not just sentences) so structured content
    (bills, tables) is divisible. ``keep_data_lines`` force-keeps units carrying a number,
    currency, or date — the literal facts relevance scoring otherwise drops.
    """
    units = split_units(content, line_aware=line_aware)
    if len(units) <= 1:
        return content

    scores, vecs = score_sentences(query, units, model)
    best = max(scores)
    if best <= 0:  # no relevance signal — keep all by relevance, but still dedup repeats
        kept = list(range(len(units)))
    else:
        threshold = keep_ratio * best
        kept = [i for i, s in enumerate(scores) if s >= threshold]
        if keep_data_lines:  # never drop a unit that holds an amount / date / figure
            kept = sorted(set(kept) | {i for i, u in enumerate(units) if _DATA_RE.search(u)})
        if not kept:  # safety: keep the single best unit
            kept = [max(range(len(units)), key=lambda i: scores[i])]

    kept = _dedup_sentences(kept, vecs, sentence_dedup_threshold)
    if len(kept) == len(units):  # nothing to drop
        return content

    return " ".join(units[i] for i in kept)  # kept is already in order


def compress_block(
    block: TokenBlock,
    query: str,
    model: EmbeddingModel,
    *,
    counter: TokenCounter | None = None,
    keep_ratio: float = _DEFAULT_KEEP_RATIO,
    sentence_dedup_threshold: float = _DEFAULT_SENTENCE_DEDUP_THRESHOLD,
    line_aware: bool = False,
    keep_data_lines: bool = False,
) -> TokenBlock:
    """Return a relevance-compressed copy of ``block`` (or the block unchanged).

    Non-compressible blocks are returned as-is. When boilerplate or duplicate units
    are dropped, a copy is returned with a recomputed ``token_count`` and audit
    breadcrumbs in ``metadata``. Size is content-determined — there is no token target.
    """
    counter = resolve_counter(counter)
    if not block.compressible:
        return block

    original_tokens = block.ensure_token_count(counter)
    new_content = compress_text(
        block.content, query, model,
        keep_ratio=keep_ratio, sentence_dedup_threshold=sentence_dedup_threshold,
        line_aware=line_aware, keep_data_lines=keep_data_lines,
    )
    if new_content == block.content:
        return block

    metadata = dict(block.metadata)
    metadata["compressed"] = True
    metadata["original_token_count"] = original_tokens
    metadata["compression_method"] = (
        "embedding_extractive_line" if line_aware else "embedding_extractive"
    )
    return block.copy(
        content=new_content,
        token_count=counter.count(new_content),
        metadata=metadata,
    )


class ExtractiveCompressor:
    """:class:`~tokengate.compression.base.Compressor` over embedding sentence-selection.

    Wraps :func:`compress_block` so the budgeter can treat compression as a pluggable
    backend. Holds the embedding model + counter so the budgeter only passes ``keep_ratio``.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        *,
        counter: TokenCounter | None = None,
        sentence_dedup_threshold: float = _DEFAULT_SENTENCE_DEDUP_THRESHOLD,
        line_aware: bool = False,
        keep_data_lines: bool = False,
    ) -> None:
        self._model = embedding_model
        self._counter = counter
        self._sentence_dedup_threshold = sentence_dedup_threshold
        self._line_aware = line_aware
        self._keep_data_lines = keep_data_lines

    def compress_block(
        self, block: TokenBlock, query: str, *, keep_ratio: float
    ) -> TokenBlock:
        return compress_block(
            block, query, self._model,
            counter=self._counter,
            keep_ratio=keep_ratio,
            sentence_dedup_threshold=self._sentence_dedup_threshold,
            line_aware=self._line_aware,
            keep_data_lines=self._keep_data_lines,
        )


__all__ = [
    "split_sentences", "score_sentences", "compress_text", "compress_block",
    "ExtractiveCompressor",
]
