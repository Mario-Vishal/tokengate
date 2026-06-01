"""ContextPilot — reusable, app-agnostic context optimization for LLM apps.

Public API is exported here as each component lands (see docs/TODO.md). During the
skeleton phase (CP-001) only the version is guaranteed importable; models and the
``ContextPilot`` optimizer are wired up in later tasks (CP-003 .. CP-012).
"""

from __future__ import annotations

from contextpilot.budgeting.token_counter import (
    HeuristicTokenCounter,
    TokenCounter,
)
from contextpilot.core.block import ContextBlock
from contextpilot.core.config import OptimizerConfig
from contextpilot.core.optimizer import ContextPilot
from contextpilot.core.result import (
    AuditReport,
    BlockDecision,
    OptimizationResult,
    StageRecord,
)
from contextpilot.utils.errors import (
    BudgetError,
    ConfigurationError,
    ContextPilotError,
    InvalidBlockError,
    OptimizationError,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # optimizer
    "ContextPilot",
    # models
    "ContextBlock",
    "OptimizationResult",
    "AuditReport",
    "BlockDecision",
    "StageRecord",
    "OptimizerConfig",
    # token counting
    "TokenCounter",
    "HeuristicTokenCounter",
    # errors
    "ContextPilotError",
    "InvalidBlockError",
    "BudgetError",
    "OptimizationError",
    "ConfigurationError",
]
