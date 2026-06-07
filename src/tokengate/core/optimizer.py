"""TokenGate optimizer — the public entrypoint (CP-012, neural rewire CP-023).

Runs the full neural pipeline and returns an :class:`OptimizationResult`::

    exact dedup -> embed (reuse vectors) -> semantic + hybrid scoring -> neural rerank
    -> adaptive relevance cutoff -> semantic dedup -> MMR diversity
    -> value-per-token budget (compress-always) -> prompt assembly -> audit

Embeddings and the reranker are required, first-class models (ADR-010). They default to
the BGE implementations (loaded lazily on first ``optimize``) and can be injected — e.g.
fakes in tests, or the app's own models. No generative LLM runs here (ADR-014).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokengate.audit.audit_report import build_audit_report
from tokengate.audit.stage_trace import StageTracer
from tokengate.budgeting.budgeter import budget_blocks
from tokengate.budgeting.token_counter import TokenCounter, resolve_counter
from tokengate.compression.base import Compressor
from tokengate.core.block import TokenBlock
from tokengate.core.config import OptimizerConfig
from tokengate.core.result import (
    DECISION_DROPPED,
    BlockDecision,
    OptimizationResult,
)
from tokengate.deduplication.exact import deduplicate_exact
from tokengate.deduplication.semantic import deduplicate_semantic
from tokengate.models.base import EmbeddingModel, Reranker
from tokengate.models.vectors import embed_query, ensure_block_vectors
from tokengate.prompts.prompt_builder import build_prompt
from tokengate.ranking.hybrid_ranker import rank_blocks
from tokengate.ranking.reranker_stage import apply_rerank_relevance, rerank_blocks
from tokengate.ranking.semantic_scorer import apply_semantic_scores
from tokengate.selection.adaptive_cutoff import select_by_adaptive_cutoff
from tokengate.selection.mmr import mmr_select
from tokengate.utils.errors import InvalidBlockError, OptimizationError, TokenGateError
from tokengate.utils.logging import get_logger, log_event

if TYPE_CHECKING:
    from tokengate.eval.compare import (
        AblationResult,
        RecipeComparisonResult,
        RecipeRunResult,
    )
    from tokengate.eval.recipe import RecipeConfig

_logger = get_logger("optimizer")


def _model_name(model: object) -> str:
    return str(getattr(model, "model_name", type(model).__name__))


class TokenGate:
    """Optimize candidate context blocks into a budgeted, audited prompt.

    Example::

        pilot = TokenGate(max_prompt_tokens=4096, strategy="balanced")
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
        recipe: str | RecipeConfig | None = None,
        embedding_model: EmbeddingModel | None = None,
        reranker: Reranker | None = None,
        token_counter: TokenCounter | None = None,
        trace: bool = True,
    ) -> None:
        if config is None and recipe is not None:
            from tokengate.eval.recipe import get_recipe

            config = get_recipe(recipe).to_optimizer_config(max_prompt_tokens=max_prompt_tokens)
        self.config = config or OptimizerConfig.for_strategy(
            strategy, max_prompt_tokens=max_prompt_tokens
        )
        self._counter = resolve_counter(self.config.token_counter or token_counter)
        self._embedding_model = embedding_model
        self._reranker = reranker
        self._compressor: Compressor | None = None
        # Per-stage tracing (CP-028); cheap, on by default. Set trace=False to skip.
        self._trace = trace

    # Models default to the BGE implementations, loaded lazily so constructing a
    # TokenGate is cheap and weights download only on first real use.
    @property
    def embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            from tokengate.models.bge import BGEM3Embedder

            self._embedding_model = BGEM3Embedder()
        return self._embedding_model

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            from tokengate.models.bge import BGEReranker

            self._reranker = BGEReranker()
        return self._reranker

    @property
    def compressor(self) -> Compressor:
        """The configured compression backend, built lazily (CP-031)."""
        if self._compressor is None:
            if self.config.compression_backend == "llmlingua2":
                from tokengate.compression.llmlingua_compressor import LLMLinguaCompressor

                self._compressor = LLMLinguaCompressor(
                    self.config.llmlingua_model, counter=self._counter
                )
            else:
                from tokengate.compression.extractive import ExtractiveCompressor

                self._compressor = ExtractiveCompressor(
                    self.embedding_model, counter=self._counter,
                    sentence_dedup_threshold=self.config.compression_sentence_dedup_threshold,
                )
        return self._compressor

    def optimize(self, query: str, blocks: list[TokenBlock]) -> OptimizationResult:
        """Run the full neural pipeline and return the result + audit."""
        if not isinstance(query, str):
            raise InvalidBlockError("query must be a string")
        for b in blocks:
            if not isinstance(b, TokenBlock):
                raise InvalidBlockError("all blocks must be TokenBlock instances")

        try:
            return self._run(query, blocks)
        except TokenGateError:
            raise
        except Exception as exc:  # unexpected -> wrap, never leak raw internals
            raise OptimizationError(f"optimization failed: {exc}") from exc

    # --- recipe lab (CP-040/041) ------------------------------------------
    # Run/compare/ablate recipes on the *same* candidate blocks, reusing this instance's
    # models (loaded once). Retrieval is the caller's job — these never re-retrieve.

    def run_recipe(
        self, query: str, blocks: list[TokenBlock], recipe: str | RecipeConfig
    ) -> RecipeRunResult:
        """Run a single named/custom recipe and return its token economics + stage trace."""
        from tokengate.eval.compare import run_recipe as _run_recipe

        return _run_recipe(
            query, blocks, recipe,
            embedding_model=self.embedding_model, reranker=self.reranker,
            counter=self._counter, max_prompt_tokens=self.config.max_prompt_tokens,
        )

    def compare_recipes(
        self,
        query: str,
        blocks: list[TokenBlock],
        recipes: list[str | RecipeConfig],
        *,
        objective: str = "balanced",
    ) -> RecipeComparisonResult:
        """Run several recipes on the same candidates and rank them by ``objective``."""
        from tokengate.eval.compare import compare_recipes as _compare

        return _compare(
            query, blocks, recipes,
            embedding_model=self.embedding_model, reranker=self.reranker,
            counter=self._counter, max_prompt_tokens=self.config.max_prompt_tokens,
            objective=objective,
        )

    def ablation(
        self, query: str, blocks: list[TokenBlock], *, base: str | RecipeConfig = "full_tg"
    ) -> AblationResult:
        """Disable each stage in turn and measure the token delta (where savings come from)."""
        from tokengate.eval.compare import ablation as _ablation

        return _ablation(
            query, blocks, base=base,
            embedding_model=self.embedding_model, reranker=self.reranker,
            counter=self._counter, max_prompt_tokens=self.config.max_prompt_tokens,
        )

    # --- internal pipeline ------------------------------------------------

    def _run(self, query: str, blocks: list[TokenBlock]) -> OptimizationResult:
        cfg = self.config
        counter = self._counter

        total_candidate_blocks = len(blocks)
        total_candidate_tokens = sum(b.ensure_token_count(counter) for b in blocks)
        log_event(_logger, "context_optimization_started",
                  candidate_blocks=total_candidate_blocks,
                  candidate_tokens=total_candidate_tokens)

        decisions: list[BlockDecision] = []
        dropped: list[TokenBlock] = []
        tracer = StageTracer(counter, enabled=self._trace)

        def drop(block: TokenBlock, reason: str) -> None:
            decisions.append(BlockDecision(
                block.block_id, DECISION_DROPPED, reason,
                block.ensure_token_count(counter), 0, block.final_score,
                rerank_score=block.rerank_score,
                source_id=block.source_id,
                content_preview=block.content[:300],
            ))
            dropped.append(block)

        # (1) exact dedup
        t = tracer.start()
        if cfg.enable_dedup:
            kept, exact_dropped = deduplicate_exact(blocks)
        else:
            kept, exact_dropped = list(blocks), []
        for b in exact_dropped:
            drop(b, "exact duplicate")
        tracer.add("exact_dedup", before=blocks, after=kept, t0=t)

        if kept:
            embedder = self.embedding_model
            # (2) reuse/compute vectors + (3) semantic scoring + (4) multi-signal ranking
            t = tracer.start()
            matrix = ensure_block_vectors(kept, embedder)
            apply_semantic_scores(kept, embed_query(query, embedder), matrix)
            ranked = rank_blocks(
                query, kept,
                semantic_weight=cfg.semantic_weight,
                keyword_weight=cfg.keyword_weight,
                recency_weight=cfg.recency_weight,
                source_priority_weight=cfg.source_priority_weight,
                token_efficiency_weight=cfg.token_efficiency_weight,
                source_priorities=cfg.source_priorities,
            )
            tracer.add("embed_rank", before=kept, after=ranked, t0=t)

            # (5) neural reranking + top-N cutoff; then let the reranker drive final_score.
            # When disabled (e.g. speed / top_k_only recipes) the hybrid-rank final_score
            # stands and no cross-encoder runs.
            if cfg.enable_reranker:
                t = tracer.start()
                reranked = rerank_blocks(query, ranked, self.reranker, top_n=cfg.rerank_top_n)
                kept_ids = {id(b) for b in reranked}
                for b in ranked:
                    if id(b) not in kept_ids:
                        drop(b, "below rerank cutoff")
                apply_rerank_relevance(reranked, rerank_weight=cfg.rerank_weight,
                                       normalization_ceiling=cfg.rerank_normalization_ceiling)
                tracer.add("rerank", before=ranked, after=reranked, t0=t)
            else:
                reranked = ranked

            # (5b) per-query adaptive relevance cutoff (CP-030): keep blocks above this
            # query's relevance cliff instead of a fixed floor — few for focused queries,
            # many for broad/synthesis ones. Replaces relevance_floor + min_rerank_score.
            if cfg.adaptive_cutoff:
                t = tracer.start()
                before_ac = reranked
                kept_ac, dropped_ac = select_by_adaptive_cutoff(
                    reranked, min_keep=cfg.adaptive_cutoff_min_keep
                )
                for b in dropped_ac:
                    drop(b, "below adaptive relevance cutoff")
                reranked = kept_ac
                tracer.add("adaptive_cutoff", before=before_ac, after=reranked, t0=t)

            # (6) semantic deduplication
            t = tracer.start()
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
            tracer.add("semantic_dedup", before=reranked, after=deduped, t0=t)

            # (7) MMR diversity ordering
            t = tracer.start()
            selected = mmr_select(deduped, lambda_=cfg.mmr_lambda) if cfg.enable_mmr else deduped
            tracer.add("mmr", before=deduped, after=selected, t0=t)

            # (8) value-per-token budgeting (compress-to-fit). When the adaptive cutoff
            # already pruned by relevance, bypass the fixed floors so we don't double-cut.
            t = tracer.start()
            outcome = budget_blocks(
                selected,
                query=query,
                budget_tokens=cfg.effective_budget,
                counter=counter,
                enable_compression=cfg.enable_compression,
                embedding_model=embedder,
                relevance_floor=0.0 if cfg.adaptive_cutoff else cfg.relevance_floor,
                # Keep the absolute rerank floor even with the adaptive cutoff: the cutoff
                # runs on final_score (rerank blended with semantic), so a block the
                # cross-encoder scored ~0 gets inflated by semantic similarity and survives
                # as noise. The floor drops "reranker says irrelevant" blocks regardless;
                # the budgeter's fallback-top-1 still guarantees at least one block.
                min_rerank_score=cfg.min_rerank_score,
                compress_always=cfg.compress_always,
                compressor=self.compressor if cfg.enable_compression else None,
                compression_keep_ratio=cfg.compression_keep_ratio,
                compression_sentence_dedup_threshold=cfg.compression_sentence_dedup_threshold,
            )
            decisions.extend(outcome.decisions)
            dropped.extend(outcome.dropped)
            included, compressed = outcome.included, outcome.compressed
            prompt_blocks = outcome.prompt_blocks
            # tokens_out reflects compression (used_tokens), not raw block sizes.
            tracer.add("budget", before=selected, after=prompt_blocks, t0=t,
                       tokens_out=outcome.used_tokens, dropped=len(outcome.dropped))
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
            stages=tracer.records,
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


__all__ = ["TokenGate"]
