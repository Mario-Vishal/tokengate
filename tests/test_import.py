"""CP-001 skeleton smoke test: the package and all subpackages import cleanly."""

from __future__ import annotations

import importlib

import tokengate


def test_version_exposed() -> None:
    assert isinstance(tokengate.__version__, str)
    assert tokengate.__version__


def test_all_subpackages_import() -> None:
    for sub in (
        "core",
        "deduplication",
        "ranking",
        "compression",
        "budgeting",
        "prompts",
        "audit",
        "utils",
    ):
        importlib.import_module(f"tokengate.{sub}")
