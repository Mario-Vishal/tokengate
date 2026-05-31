"""ContextPilot optimizer — the public entrypoint (CP-012, neural rewire CP-023).

Runs the full neural pipeline and returns an :class:`OptimizationResult`::

    exact dedup -> embed (reuse vectors) -> semantic + hybrid scoring -> neural rerank
    -> semantic dedup -> MMR diversity -> value-per-token budget (compress-to-fit)
    -> prompt assembly -> audit

Embeddings and the reranker are required, first-class models (ADR-010). They default to
the BGE implementations (loaded lazily on first ``optimize``) and can be injected — e.g.
fakes in tests, or the app's own models. No generative LLM runs here (ADR-014).
"""

from __future__ import annotations

from contextpilot.audit.audit_report import build_audit_report
from contextpilot.budgeting.budgeter import budget_blocks
from contextpilot.budgeting.token_counter import TokenCounter, resolve_counter
from contextpilot.core.block import ContextBlock
from contextpilot.core.config import OptimizerConfig
from contextpilot.core.result import (
    DECISION_DROPPED,
    BlockDecision,
    OptimizationResult,
)
from contextpilot.deduplication.exact import deduplicate_exact
from contextpilot.deduplication.semantic import deduplicate_semantic
from contextpilot.models.base import EmbeddingModel, Reranker
from contextpilot.models.vectors import embed_query, ensure_block_vectors
from contextpilot.prompts.prompt_builder import build_prompt
from contextpilot.ranking.hybrid_ranker import rank_blocks
from contextpilot.ranking.reranker_stage import apply_rerank_relevance, rerank_blocks
from contextpilot.ranking.semantic_scorer import apply_semantic_scores
from contextpilot.selection.mmr import mmr_select
from contextpilot.utils.errors import ContextPilotError, InvalidBlockError, OptimizationError
from contextpilot.utils.logging import get_logger, log_event

_logger = get_logger("optimizer")


def _model_name(model: object) -> str:
    return str(getattr(model, "model_name", type(model).__name__))


class ContextPilot:
    """Optimize candidate context blocks into a budgeted, audited prompt.

    Example::

        pilot = ContextPilot(max_prompt_tokens=4096, strategy="balanced")
        result = pilot.optimize(query="...", blocks=[...])
        result.final_prompt   # send to any LLM
        result.audit          # included / compressed / dropped, scores, models, reasons
    """

    def __init__(
        self,
        max_prompt_tokens: int = 4096,
        strategy: str = "balanced",
        *,
        config: OptimizerConfig | None = None,
        embedding_model: EmbeddingModel | None = None,
        reranker: Reranker | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.config = config or OptimizerConfig.for_strategy(
            strategy, max_prompt_tokens=max_prompt_tokens
        )
        self._counter = resolve_counter(self.config.token_counter or token_counter)
        self._embedding_model = embedding_model
        self._reranker = reranker

    # Models default to the BGE implementations, loaded lazily so constructing a
    # ContextPilot is cheap and weights download only on first real use.
    @property
    def embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            from contextpilot.models.bge import BGEM3Embedder

            self._embedding_model = BGEM3Embedder()
        return self._embedding_model

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            from contextpilot.models.bge import BGEReranker

            self._reranker = BGEReranker()
        return self._reranker

    def optimize(self, query: str, blocks: list[ContextBlock]) -> OptimizationResult:
        """Run the full neural pipeline and return the result + audit."""
        if not isinstance(query, str):
            raise InvalidBlockError("query must be a string")
        for b in blocks:
            if not isinstance(b, ContextBlock):
                raise InvalidBlockError("all blocks must be ContextBlock instances")

        try:
            return self._run(query, blocks)
        except ContextPilotError:
            raise
        except Exception as exc:  # unexpected -> wrap, never leak raw internals
            raise OptimizationError(f"optimization failed: {exc}") from exc

    # --- internal pipeline ------------------------------------------------

    def _run(self, query: str, blocks: list[ContextBlock]) -> OptimizationResult:
        cfg = self.config
        counter = self._counter

        total_candidate_blocks = len(blocks)
        total_candidate_tokens = sum(b.ensure_token_count(counter) for b in blocks)
        log_event(_logger, "context_optimization_started",
                  candidate_blocks=total_candidate_blocks,
                  candidate_tokens=total_candidate_tokens)

        decisions: list[BlockDecision] = []
        dropped: list[ContextBlock] = []

        def drop(block: ContextBlock, reason: str) -> None:
            decisions.append(BlockDecision(
                block.block_id, DECISION_DROPPED, reason,
                block.ensure_token_count(counter), 0, block.final_score,
                rerank_score=block.rerank_score,
            ))
            dropped.append(block)

        # (1) exact dedup
        if cfg.enable_dedup:
            kept, exact_dropped = deduplicate_exact(blocks)
        else:
            kept, exact_dropped = list(blocks), []
        for b in exact_dropped:
            drop(b, "exact duplicate")

        if kept:
            embedder = self.embedding_model
            # (2) reuse/compute vectors + (3) semantic scoring
            matrix = ensure_block_vectors(kept, embedder)
            apply_semantic_scores(kept, embed_query(query, embedder), matrix)

            # (4) multi-signal hybrid ranking
            ranked = rank_blocks(
                query, kept,
                semantic_weight=cfg.semantic_weight,
                keyword_weight=cfg.keyword_weight,
                recency_weight=cfg.recency_weight,
                source_priority_weight=cfg.source_priority_weight,
                token_efficiency_weight=cfg.token_efficiency_weight,
                source_priorities=cfg.source_priorities,
            )

            # (5) neural reranking + top-N cutoff; then let the reranker drive final_score
            reranked = rerank_blocks(query, ranked, self.reranker, top_n=cfg.rerank_top_n)
            kept_ids = {id(b) for b in reranked}
            for b in ranked:
                if id(b) not in kept_ids:
                    drop(b, "below rerank cutoff")
            apply_rerank_relevance(reranked, rerank_weight=cfg.rerank_weight)

            # (6) semantic deduplication
            if cfg.enable_semantic_dedup:
                sd = deduplicate_semantic(
                    reranked, threshold=cfg.semantic_dedup_threshold
                )
                for b in sd.dropped:
                    rep = sd.duplicate_of[b.block_id]
                    sim = sd.similarity[b.block_id]
                    drop(b, f"semantic duplicate of {rep} (cosine {sim:.2f})")
                deduped = sd.kept
            else:
                deduped = reranked

            # (7) MMR diversity ordering
            selected = mmr_select(deduped, lambda_=cfg.mmr_lambda) if cfg.enable_mmr else deduped

            # (8) value-per-token budgeting (compress-to-fit)
            outcome = budget_blocks(
                selected,
                query=query,
                budget_tokens=cfg.effective_budget,
                counter=counter,
                enable_compression=cfg.enable_compression,
                embedding_model=embedder,
                relevance_floor=cfg.relevance_floor,
                compression_keep_ratio=cfg.compression_keep_ratio,
            )
            decisions.extend(outcome.decisions)
            dropped.extend(outcome.dropped)
            included, compressed = outcome.included, outcome.compressed
            prompt_blocks = outcome.prompt_blocks
        else:
            included, compressed, prompt_blocks = [], [], []

        # (9) prompt assembly
        final_prompt = build_prompt(query, prompt_blocks)
        final_prompt_tokens = counter.count(final_prompt)

        # (10) audit
        emb_name = _model_name(self._embedding_model) if self._embedding_model else None
        rer_name = _model_name(self._reranker) if self._reranker else None
        audit = build_audit_report(
            total_candidate_blocks=total_candidate_blocks,
            total_candidate_tokens=total_candidate_tokens,
            final_prompt_tokens=final_prompt_tokens,
            decisions=decisions,
            models_used={
                "embedding_model": emb_name,
                "reranker": rer_name,
                "final_llm": None,  # filled in by the app
            },
        )
        log_event(_logger, "context_optimization_completed",
                  final_prompt_tokens=final_prompt_tokens,
                  tokens_saved=audit.tokens_saved,
                  included=audit.included_count,
                  compressed=audit.compressed_count,
                  dropped=audit.dropped_count)

        return OptimizationResult(
            query=query,
            final_prompt=final_prompt,
            included_blocks=included,
            compressed_blocks=compressed,
            dropped_blocks=dropped,
            audit=audit,
        )


__all__ = ["ContextPilot"]
