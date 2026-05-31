"""ContextPilot optimizer — the public entrypoint (CP-012).

Wires the pipeline together:
``dedup -> rank -> budget (compress-to-fit) -> build prompt -> audit`` and returns an
:class:`OptimizationResult`. Pure and deterministic for a given input + config.
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
from contextpilot.prompts.prompt_builder import build_prompt
from contextpilot.ranking.hybrid_ranker import rank_blocks
from contextpilot.utils.errors import ContextPilotError, InvalidBlockError, OptimizationError
from contextpilot.utils.logging import get_logger, log_event

_logger = get_logger("optimizer")


class ContextPilot:
    """Optimize candidate context blocks into a budgeted, audited prompt.

    Example::

        pilot = ContextPilot(max_prompt_tokens=4096, strategy="balanced")
        result = pilot.optimize(query="...", blocks=[...])
        result.final_prompt   # send to any LLM
        result.audit          # what was included / compressed / dropped, and why
    """

    def __init__(
        self,
        max_prompt_tokens: int = 4096,
        strategy: str = "balanced",
        *,
        config: OptimizerConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.config = config or OptimizerConfig(
            max_prompt_tokens=max_prompt_tokens,
            strategy=strategy,
            token_counter=token_counter,
        )
        self._counter = resolve_counter(self.config.token_counter or token_counter)

    def optimize(self, query: str, blocks: list[ContextBlock]) -> OptimizationResult:
        """Run the full optimization pipeline and return the result + audit."""
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

        # (1) exact deduplication
        if cfg.enable_dedup:
            kept, dup_dropped = deduplicate_exact(blocks)
        else:
            kept, dup_dropped = list(blocks), []

        dedup_decisions = [
            BlockDecision(
                b.block_id, DECISION_DROPPED, "exact duplicate",
                b.ensure_token_count(counter), 0, b.final_score,
            )
            for b in dup_dropped
        ]

        # (2-4) score + rank (multi-signal hybrid)
        ranked = rank_blocks(
            query, kept,
            semantic_weight=cfg.semantic_weight,
            keyword_weight=cfg.keyword_weight,
            recency_weight=cfg.recency_weight,
            source_priority_weight=cfg.source_priority_weight,
            token_efficiency_weight=cfg.token_efficiency_weight,
            source_priorities=cfg.source_priorities,
        )

        # (5-7) budget with compress-to-fit
        outcome = budget_blocks(
            ranked,
            query=query,
            budget_tokens=cfg.effective_budget,
            counter=counter,
            enable_compression=cfg.enable_compression,
        )

        # (8) build prompt from in-prompt blocks (ranked order, compressed substituted)
        final_prompt = build_prompt(query, outcome.prompt_blocks)
        final_prompt_tokens = counter.count(final_prompt)

        # (9) audit
        audit = build_audit_report(
            total_candidate_blocks=total_candidate_blocks,
            total_candidate_tokens=total_candidate_tokens,
            final_prompt_tokens=final_prompt_tokens,
            decisions=dedup_decisions + outcome.decisions,
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
            included_blocks=outcome.included,
            compressed_blocks=outcome.compressed,
            dropped_blocks=dup_dropped + outcome.dropped,
            audit=audit,
        )


__all__ = ["ContextPilot"]
