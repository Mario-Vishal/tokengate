"""Near-duplicate deduplication — see :mod:`contextpilot.deduplication.semantic`.

Kept as the name referenced in ARCHITECTURE.md; the implementation lives in
``semantic.py`` (CP-019). This module re-exports it.
"""

from __future__ import annotations

from contextpilot.deduplication.semantic import (
    SemanticDedupResult,
    deduplicate_semantic,
)

__all__ = ["SemanticDedupResult", "deduplicate_semantic"]
