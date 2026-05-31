"""CP-014: the neural-engine core dependencies are present and importable.

These are REQUIRED core deps (ADR-010), so they must import. The test documents the
hard requirement and fails loudly if the environment is missing them.
"""

from __future__ import annotations

import sys


def test_python_is_312() -> None:
    assert sys.version_info[:2] == (3, 12), "contextpilot targets Python 3.12 (ADR-012)"


def test_core_ml_deps_import() -> None:
    import numpy  # noqa: F401
    import sentence_transformers  # noqa: F401
    import torch  # noqa: F401


def test_torch_basic_tensor_op() -> None:
    import torch

    x = torch.tensor([1.0, 2.0, 3.0])
    assert float(x.sum()) == 6.0
