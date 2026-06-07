"""Run and compare pipeline recipes on the same candidate blocks (CP-041).

The core eval primitive: hold retrieval constant (the caller passes one candidate set) and
vary only the optimization recipe, so differences are attributable to the *pipeline*, not to
luck in retrieval. This is what answers "which recipe is best for my data?" and "where do my
token savings actually come from?" (the ablation).

No LLM is involved here — these measure token economics and per-stage contribution, which is
deterministic and fast. Quality metrics (coverage/answerability) attach later (Phase C).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any

from tokengate.budgeting.token_counter import TokenCounter, resolve_counter
from tokengate.core.block import TokenBlock
from tokengate.eval.recipe import RecipeConfig, get_recipe

if TYPE_CHECKING:
    from tokengate.models.base import EmbeddingModel, Reranker

# Stages that can be individually disabled for ablation -> the OptimizerConfig flag.
ABLATABLE_STAGES: dict[str, str] = {
    "reranker": "enable_reranker",
    "semantic_dedup": "enable_semantic_dedup",
    "mmr": "enable_mmr",
    "compression": "enable_compression",
    "adaptive_cutoff": "adaptive_cutoff",
}


@dataclass
class RecipeRunResult:
    """Token economics + per-stage contribution for one recipe on one query."""

    recipe: str
    final_prompt_tokens: int
    candidate_tokens: int
    tokens_saved: int
    tokens_saved_percent: float
    included: int
    compressed: int
    dropped: int
    latency_ms: float
    # stage name -> tokens removed at that stage (tokens_in - tokens_out), from the trace.
    stage_contributions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "final_prompt_tokens": self.final_prompt_tokens,
            "candidate_tokens": self.candidate_tokens,
            "tokens_saved": self.tokens_saved,
            "tokens_saved_percent": round(self.tokens_saved_percent, 1),
            "included": self.included,
            "compressed": self.compressed,
            "dropped": self.dropped,
            "latency_ms": round(self.latency_ms),
            "stage_contributions": dict(self.stage_contributions),
        }


@dataclass
class RecipeComparisonResult:
    query: str
    candidate_blocks: int
    candidate_tokens: int
    objective: str
    runs: list[RecipeRunResult] = field(default_factory=list)
    recommended: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "candidate_blocks": self.candidate_blocks,
            "candidate_tokens": self.candidate_tokens,
            "objective": self.objective,
            "recommended": self.recommended,
            "runs": [r.to_dict() for r in self.runs],
        }


@dataclass
class AblationResult:
    query: str
    base_recipe: str
    base_tokens: int
    # stage label -> {"tokens_without": int, "delta": int, "delta_percent": float}
    deltas: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "base_recipe": self.base_recipe,
            "base_tokens": self.base_tokens,
            "deltas": self.deltas,
        }


def _fresh_blocks(blocks: list[TokenBlock]) -> list[TokenBlock]:
    """Copy blocks and clear derived scores so each recipe run starts clean.

    Recipes run sequentially on the *same* candidate set; without this, one run's
    final_score/rerank_score would leak into the next. The input ``vector`` is preserved
    (it's a genuine input, reused to avoid re-embedding).
    """
    fresh: list[TokenBlock] = []
    for b in blocks:
        c = b.copy()
        c.semantic_score = None
        c.keyword_score = None
        c.final_score = None
        c.rerank_score = None
        fresh.append(c)
    return fresh


def run_recipe(
    query: str,
    blocks: list[TokenBlock],
    recipe: str | RecipeConfig,
    *,
    embedding_model: EmbeddingModel | None = None,
    reranker: Reranker | None = None,
    counter: TokenCounter | None = None,
    max_prompt_tokens: int = 4096,
) -> RecipeRunResult:
    """Run one recipe on ``blocks`` and return its token economics + stage contribution."""
    from tokengate.core.optimizer import TokenGate  # lazy: avoid import cycle

    counter = resolve_counter(counter)
    rc = get_recipe(recipe)
    config = rc.to_optimizer_config(max_prompt_tokens=max_prompt_tokens)
    gate = TokenGate(
        config=config, embedding_model=embedding_model, reranker=reranker,
        token_counter=counter, trace=True,
    )
    fresh = _fresh_blocks(blocks)
    t0 = perf_counter()
    result = gate.optimize(query, fresh)
    latency_ms = (perf_counter() - t0) * 1000.0

    audit = result.audit
    assert audit is not None  # trace=True always produces an audit
    stage_contrib = {s.stage: s.tokens_in - s.tokens_out for s in audit.stages}
    return RecipeRunResult(
        recipe=rc.name,
        final_prompt_tokens=audit.final_prompt_tokens,
        candidate_tokens=audit.total_candidate_tokens,
        tokens_saved=audit.tokens_saved,
        tokens_saved_percent=audit.tokens_saved_percent,
        included=audit.included_count,
        compressed=audit.compressed_count,
        dropped=audit.dropped_count,
        latency_ms=latency_ms,
        stage_contributions=stage_contrib,
    )


def _recommend(runs: list[RecipeRunResult], objective: str) -> str | None:
    """Deterministic recipe pick. Quality-based objectives arrive with the quality layer."""
    if not runs:
        return None
    if objective == "savings":
        return min(runs, key=lambda r: r.final_prompt_tokens).recipe
    if objective == "speed":
        return min(runs, key=lambda r: r.latency_ms).recipe
    # "balanced" (default): min-max normalize tokens + latency, equal weight, lower is better.
    toks = [r.final_prompt_tokens for r in runs]
    lats = [r.latency_ms for r in runs]
    t_lo, t_hi = min(toks), max(toks)
    l_lo, l_hi = min(lats), max(lats)

    def norm(v: float, lo: float, hi: float) -> float:
        return 0.0 if hi <= lo else (v - lo) / (hi - lo)

    return min(
        runs,
        key=lambda r: norm(r.final_prompt_tokens, t_lo, t_hi) + norm(r.latency_ms, l_lo, l_hi),
    ).recipe


def compare_recipes(
    query: str,
    blocks: list[TokenBlock],
    recipes: list[str | RecipeConfig],
    *,
    embedding_model: EmbeddingModel | None = None,
    reranker: Reranker | None = None,
    counter: TokenCounter | None = None,
    max_prompt_tokens: int = 4096,
    objective: str = "balanced",
) -> RecipeComparisonResult:
    """Run every recipe on the same candidate blocks and rank them by ``objective``."""
    counter = resolve_counter(counter)
    runs = [
        run_recipe(
            query, blocks, r, embedding_model=embedding_model, reranker=reranker,
            counter=counter, max_prompt_tokens=max_prompt_tokens,
        )
        for r in recipes
    ]
    cand_tokens = runs[0].candidate_tokens if runs else 0
    return RecipeComparisonResult(
        query=query,
        candidate_blocks=len(blocks),
        candidate_tokens=cand_tokens,
        objective=objective,
        runs=runs,
        recommended=_recommend(runs, objective),
    )


def ablation(
    query: str,
    blocks: list[TokenBlock],
    *,
    base: str | RecipeConfig = "full_tg",
    embedding_model: EmbeddingModel | None = None,
    reranker: Reranker | None = None,
    counter: TokenCounter | None = None,
    max_prompt_tokens: int = 4096,
) -> AblationResult:
    """Disable each stage in turn and measure the token delta — "where do savings come from?".

    A *positive* delta means disabling the stage made the prompt bigger, i.e. that stage was
    pulling its weight. A delta near zero means the stage contributed ~nothing.
    """
    counter = resolve_counter(counter)
    base_rc = get_recipe(base)
    base_run = run_recipe(
        query, blocks, base_rc, embedding_model=embedding_model, reranker=reranker,
        counter=counter, max_prompt_tokens=max_prompt_tokens,
    )
    deltas: dict[str, dict[str, float]] = {}
    for label, flag in ABLATABLE_STAGES.items():
        rc = base_rc.with_overrides(name=f"{base_rc.name}-no_{label}", **{flag: False})
        run = run_recipe(
            query, blocks, rc, embedding_model=embedding_model, reranker=reranker,
            counter=counter, max_prompt_tokens=max_prompt_tokens,
        )
        delta = run.final_prompt_tokens - base_run.final_prompt_tokens
        base_tok = base_run.final_prompt_tokens or 1
        deltas[label] = {
            "tokens_without": run.final_prompt_tokens,
            "delta": delta,
            "delta_percent": round(delta / base_tok * 100.0, 1),
        }
    return AblationResult(
        query=query, base_recipe=base_rc.name,
        base_tokens=base_run.final_prompt_tokens, deltas=deltas,
    )


__all__ = [
    "RecipeRunResult", "RecipeComparisonResult", "AblationResult",
    "run_recipe", "compare_recipes", "ablation", "ABLATABLE_STAGES",
]
