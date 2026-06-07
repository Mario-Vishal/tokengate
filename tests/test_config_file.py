"""CP-050 tests: TOML config-file loading + the compression-off default."""

from __future__ import annotations

from pathlib import Path

import pytest

from tokengate import ConfigurationError, OptimizerConfig
from tokengate.core.config import resolve_config_path

# --- compression is OFF by default (ships off; opt-in per strategy) ----------

def test_compression_off_by_default() -> None:
    assert OptimizerConfig().enable_compression is False
    assert OptimizerConfig.for_strategy("balanced").enable_compression is False
    assert OptimizerConfig.for_strategy("quality").enable_compression is False
    assert OptimizerConfig.for_strategy("speed").enable_compression is False
    # max_compression is the explicit opt-in lever.
    assert OptimizerConfig.for_strategy("max_compression").enable_compression is True


# --- from_dict: override only the provided keys ------------------------------

def test_from_dict_overrides_only_given_keys() -> None:
    cfg = OptimizerConfig.from_dict({"max_prompt_tokens": 8192, "mmr_lambda": 0.3})
    assert cfg.max_prompt_tokens == 8192
    assert cfg.mmr_lambda == 0.3
    # untouched keys keep balanced defaults
    assert cfg.semantic_weight == 0.45
    assert cfg.strategy == "balanced"


def test_from_dict_respects_strategy_key() -> None:
    cfg = OptimizerConfig.from_dict({"strategy": "quality", "rerank_top_n": 40})
    assert cfg.strategy == "quality"
    assert cfg.rerank_top_n == 40
    assert cfg.mmr_lambda == 0.7  # from the quality preset, not overridden


def test_from_dict_explicit_strategy_arg_wins() -> None:
    cfg = OptimizerConfig.from_dict({"strategy": "quality"}, strategy="speed")
    assert cfg.strategy == "speed"


def test_from_dict_unknown_key_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown config key"):
        OptimizerConfig.from_dict({"max_promt_tokens": 8192})  # typo


def test_from_dict_invalid_value_still_validated() -> None:
    with pytest.raises(ConfigurationError):
        OptimizerConfig.from_dict({"mmr_lambda": 1.5})


# --- from_file: TOML parsing -------------------------------------------------

def test_from_file_reads_tokengate_section(tmp_path: Path) -> None:
    p = tmp_path / "tokengate.toml"
    p.write_text(
        "[tokengate]\n"
        'strategy = "quality"\n'
        "max_prompt_tokens = 6000\n"
        "semantic_dedup_threshold = 0.88\n",
        encoding="utf-8",
    )
    cfg = OptimizerConfig.from_file(p)
    assert cfg.strategy == "quality"
    assert cfg.max_prompt_tokens == 6000
    assert cfg.semantic_dedup_threshold == 0.88


def test_from_file_reads_top_level_when_no_section(tmp_path: Path) -> None:
    p = tmp_path / "tg.toml"
    p.write_text("max_prompt_tokens = 1234\n", encoding="utf-8")
    assert OptimizerConfig.from_file(p).max_prompt_tokens == 1234


def test_from_file_source_priorities_table(tmp_path: Path) -> None:
    p = tmp_path / "tokengate.toml"
    p.write_text(
        "[tokengate.source_priorities]\n"
        'desktop = 0.9\n'
        'trash = 0.1\n',
        encoding="utf-8",
    )
    cfg = OptimizerConfig.from_file(p)
    assert cfg.source_priorities == {"desktop": 0.9, "trash": 0.1}


# --- load: resolution order --------------------------------------------------

def test_load_no_file_returns_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOKENGATE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)  # no tokengate.toml here
    cfg = OptimizerConfig.load()
    assert cfg.strategy == "balanced"
    assert cfg.enable_compression is False


def test_load_picks_up_local_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOKENGATE_CONFIG", raising=False)
    (tmp_path / "tokengate.toml").write_text(
        "[tokengate]\nmax_prompt_tokens = 4242\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert OptimizerConfig.load().max_prompt_tokens == 4242


def test_load_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "custom.toml"
    cfg_file.write_text("[tokengate]\nmax_prompt_tokens = 7777\n", encoding="utf-8")
    monkeypatch.setenv("TOKENGATE_CONFIG", str(cfg_file))
    assert OptimizerConfig.load().max_prompt_tokens == 7777


def test_load_explicit_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="config file not found"):
        OptimizerConfig.load(tmp_path / "nope.toml")


def test_env_var_missing_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENGATE_CONFIG", str(tmp_path / "ghost.toml"))
    with pytest.raises(ConfigurationError, match="TOKENGATE_CONFIG"):
        OptimizerConfig.load()


def test_resolve_config_path_none_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TOKENGATE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_config_path() is None
