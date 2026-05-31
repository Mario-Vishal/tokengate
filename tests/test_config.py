"""CP-005 tests: OptimizerConfig validation, presets, effective budget."""

from __future__ import annotations

import pytest

from contextpilot import ConfigurationError, OptimizerConfig
from contextpilot.core.config import STRATEGY_BALANCED


def test_defaults_are_balanced() -> None:
    cfg = OptimizerConfig()
    assert cfg.strategy == STRATEGY_BALANCED
    assert cfg.max_prompt_tokens == 4096
    assert cfg.semantic_weight == 0.45
    assert cfg.keyword_weight == 0.25
    assert cfg.recency_weight == 0.10
    assert cfg.source_priority_weight == 0.10
    assert cfg.token_efficiency_weight == 0.10


def test_all_zero_weights_rejected() -> None:
    with pytest.raises(ConfigurationError):
        OptimizerConfig(
            semantic_weight=0.0, keyword_weight=0.0, recency_weight=0.0,
            source_priority_weight=0.0, token_efficiency_weight=0.0,
        )


def test_bad_source_priority_value_rejected() -> None:
    with pytest.raises(ConfigurationError):
        OptimizerConfig(source_priorities={"downloads": 1.5})


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_budget_rejected(bad: int) -> None:
    with pytest.raises(ConfigurationError):
        OptimizerConfig(max_prompt_tokens=bad)


def test_unknown_strategy_rejected() -> None:
    with pytest.raises(ConfigurationError):
        OptimizerConfig(strategy="nonsense-strategy")


def test_known_strategies_accepted() -> None:
    for s in ("speed", "balanced", "quality", "max_compression"):
        assert OptimizerConfig(strategy=s).strategy == s


def test_negative_weight_rejected() -> None:
    with pytest.raises(ConfigurationError):
        OptimizerConfig(semantic_weight=-0.1)


def test_semantic_and_keyword_zero_ok_when_others_positive() -> None:
    # Other signals still carry weight, so this is valid (not all-zero).
    cfg = OptimizerConfig(semantic_weight=0.0, keyword_weight=0.0)
    assert cfg.recency_weight > 0


@pytest.mark.parametrize("bad", [-0.01, 1.0, 1.5])
def test_safety_margin_bounds(bad: float) -> None:
    with pytest.raises(ConfigurationError):
        OptimizerConfig(safety_margin=bad)


def test_rerank_top_n_validation() -> None:
    assert OptimizerConfig(rerank_top_n=None).rerank_top_n is None
    assert OptimizerConfig(rerank_top_n=10).rerank_top_n == 10
    with pytest.raises(ConfigurationError):
        OptimizerConfig(rerank_top_n=0)


def test_effective_budget_applies_margin() -> None:
    cfg = OptimizerConfig(max_prompt_tokens=1000, safety_margin=0.1)
    assert cfg.effective_budget == 900


def test_effective_budget_floor_at_least_one() -> None:
    cfg = OptimizerConfig(max_prompt_tokens=1, safety_margin=0.99)
    assert cfg.effective_budget >= 1


def test_for_strategy_balanced_with_override() -> None:
    cfg = OptimizerConfig.for_strategy(STRATEGY_BALANCED, max_prompt_tokens=2048)
    assert cfg.max_prompt_tokens == 2048
    assert cfg.strategy == STRATEGY_BALANCED


def test_for_strategy_unknown_rejected() -> None:
    with pytest.raises(ConfigurationError):
        OptimizerConfig.for_strategy("nonsense-strategy")
