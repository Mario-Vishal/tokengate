"""CP-001 skeleton smoke test: the package and all subpackages import cleanly."""

from __future__ import annotations

import importlib

import contextpilot


def test_version_exposed() -> None:
    assert isinstance(contextpilot.__version__, str)
    assert contextpilot.__version__


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
        importlib.import_module(f"contextpilot.{sub}")
