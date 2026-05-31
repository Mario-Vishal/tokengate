"""CP-005 tests: OptimizerConfig validation, presets, effective budget."""

from __future__ import annotations

import pytest

from contextpilot import ConfigurationError, OptimizerConfig
from contextpilot.core.config import STRATEGY_BALANCED


def test_defaults_are_balanced() -> None:
    cfg = OptimizerConfig()
    assert cfg.strategy == STRATEGY_BALANCED
    assert cfg.max_prompt_tokens == 4096
    assert cfg.semantic_weight == 0.6
    assert cfg.keyword_weight == 0.4


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_budget_rejected(bad: int) -> None:
    with pytest.raises(ConfigurationError):
        OptimizerConfig(max_prompt_tokens=bad)


def test_unknown_strategy_rejected() -> None:
    with pytest.raises(ConfigurationError):
        OptimizerConfig(strategy="quality")  # V2, not yet supported


def test_negative_weight_rejected() -> None:
    with pytest.raises(ConfigurationError):
        OptimizerConfig(semantic_weight=-0.1)


def test_both_weights_zero_rejected() -> None:
    with pytest.raises(ConfigurationError):
        OptimizerConfig(semantic_weight=0.0, keyword_weight=0.0)


@pytest.mark.parametrize("bad", [-0.01, 1.0, 1.5])
def test_safety_margin_bounds(bad: float) -> None:
    with pytest.raises(ConfigurationError):
        OptimizerConfig(safety_margin=bad)


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
        OptimizerConfig.for_strategy("speed")
