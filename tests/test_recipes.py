"""CP-040/041 tests: pipeline recipes, recipe comparison, and ablation (with fakes)."""

from __future__ import annotations

import pytest

from tokengate import TokenBlock, TokenGate
from tokengate.core.config import OptimizerConfig
from tokengate.eval import (
    RecipeConfig,
    ablation,
    compare_recipes,
    get_recipe,
    list_recipes,
    run_recipe,
)
from tokengate.eval.compare import ABLATABLE_STAGES
from tokengate.eval.report import ablation_to_markdown, comparison_to_markdown
from tokengate.models import FakeEmbeddingModel, FakeReranker
from tokengate.utils.errors import ConfigurationError

_EMB = FakeEmbeddingModel(dim=128)
_RER = FakeReranker()


def _blocks() -> list[TokenBlock]:
    # A duplicate-heavy, mixed set so dedup/rerank/budget all have something to do.
    return [
        TokenBlock(content="System: answer concisely.", block_type="system",
                   required=True, compressible=False, token_count=8),
        TokenBlock(content="The 2026 job search plan: update resume, contact recruiters, "
                           "apply weekly.",
                   source_id="plan.md", token_count=40),
        TokenBlock(content="The 2026 job search plan: update resume, contact recruiters, "
                           "apply weekly.",
                   source_id="plan_copy.md", token_count=40),
        TokenBlock(content="Resume tips: quantify impact, keep it one page, tailor keywords "
                           "per role.",
                   source_id="tips.md", token_count=40),
        TokenBlock(content="Grocery list: bananas, milk, eggs, bread, coffee, oats.",
                   source_id="groceries.txt", token_count=40),
        TokenBlock(content="Weather notes: rainy on Tuesday, sunny by the weekend in the bay.",
                   source_id="weather.txt", token_count=40),
    ]


# --- RecipeConfig --------------------------------------------------------

def test_builtin_recipes_exist_and_compile() -> None:
    assert set(list_recipes()) >= {"top_k_only", "reranker_only", "full_tg", "balanced"}
    for name in list_recipes():
        cfg = get_recipe(name).to_optimizer_config(max_prompt_tokens=512)
        assert isinstance(cfg, OptimizerConfig)


def test_invalid_recipe_override_fails() -> None:
    bad = RecipeConfig("bad", "invalid floor", base="balanced",
                       overrides={"relevance_floor": 5.0})
    with pytest.raises(ConfigurationError):
        bad.validate()


def test_unknown_recipe_name_raises() -> None:
    with pytest.raises(ConfigurationError):
        get_recipe("does_not_exist")


def test_recipe_serialization_roundtrip() -> None:
    r = get_recipe("dedup_focused")
    again = RecipeConfig.from_dict(r.to_dict())
    assert again.base == r.base
    assert again.overrides == r.overrides
    assert "OptimizerConfig" in r.to_python_snippet()


def test_top_k_only_disables_reranker_in_compiled_config() -> None:
    cfg = get_recipe("top_k_only").to_optimizer_config()
    assert cfg.enable_reranker is False
    assert cfg.enable_semantic_dedup is False
    assert cfg.enable_compression is False


# --- run / compare -------------------------------------------------------

def test_run_recipe_returns_result() -> None:
    res = run_recipe("job search resume", _blocks(), "balanced",
                     embedding_model=_EMB, reranker=_RER, max_prompt_tokens=300)
    assert res.recipe == "balanced"
    assert res.final_prompt_tokens > 0
    assert res.included >= 1


def test_disabled_stage_is_actually_disabled() -> None:
    # top_k_only must not run the reranker -> no "rerank" stage in the trace contributions.
    res = run_recipe("job search resume", _blocks(), "top_k_only",
                     embedding_model=_EMB, reranker=_RER, max_prompt_tokens=300)
    assert "rerank" not in res.stage_contributions
    # full_tg does run it.
    res2 = run_recipe("job search resume", _blocks(), "full_tg",
                      embedding_model=_EMB, reranker=_RER, max_prompt_tokens=300)
    assert "rerank" in res2.stage_contributions


def test_compare_recipes_runs_all_and_recommends() -> None:
    recipes = ["top_k_only", "reranker_only", "dedup_focused", "full_tg"]
    comp = compare_recipes("job search resume", _blocks(), recipes,
                           embedding_model=_EMB, reranker=_RER, max_prompt_tokens=300,
                           objective="savings")
    assert [r.recipe for r in comp.runs] == recipes
    assert comp.recommended in recipes
    # "savings" objective => recommended has the fewest final tokens.
    best = min(comp.runs, key=lambda r: r.final_prompt_tokens)
    assert comp.recommended == best.recipe
    assert "Recipe comparison" in comparison_to_markdown(comp)


def test_compare_reuses_blocks_without_mutating_caller_list() -> None:
    blocks = _blocks()
    before = [(b.block_id, b.final_score, b.rerank_score) for b in blocks]
    compare_recipes("job search", blocks, ["balanced", "top_k_only"],
                    embedding_model=_EMB, reranker=_RER, max_prompt_tokens=300)
    after = [(b.block_id, b.final_score, b.rerank_score) for b in blocks]
    assert before == after  # caller's blocks untouched (runs use fresh copies)


# --- ablation ------------------------------------------------------------

def test_ablation_reports_delta_per_stage() -> None:
    ab = ablation("job search resume", _blocks(), base="full_tg",
                  embedding_model=_EMB, reranker=_RER, max_prompt_tokens=300)
    assert set(ab.deltas) == set(ABLATABLE_STAGES)
    for d in ab.deltas.values():
        assert "delta" in d and "tokens_without" in d
    assert "Ablation" in ablation_to_markdown(ab)


# --- TokenGate API -------------------------------------------------------

def test_tokengate_recipe_constructor() -> None:
    gate = TokenGate(recipe="top_k_only", embedding_model=_EMB, reranker=_RER,
                     max_prompt_tokens=300)
    assert gate.config.enable_reranker is False


def test_tokengate_compare_recipes_method() -> None:
    gate = TokenGate(recipe="balanced", embedding_model=_EMB, reranker=_RER,
                     max_prompt_tokens=300)
    comp = gate.compare_recipes("job search", _blocks(), ["top_k_only", "full_tg"])
    assert len(comp.runs) == 2
