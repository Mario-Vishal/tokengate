"""TokenGate — reusable, app-agnostic context optimization for LLM apps.

Public API is exported here as each component lands (see docs/TODO.md). During the
skeleton phase (CP-001) only the version is guaranteed importable; models and the
``TokenGate`` optimizer are wired up in later tasks (CP-003 .. CP-012).
"""

from __future__ import annotations

from tokengate.budgeting.token_counter import (
    HeuristicTokenCounter,
    TokenCounter,
)
from tokengate.core.block import TokenBlock
from tokengate.core.config import OptimizerConfig
from tokengate.core.optimizer import TokenGate
from tokengate.core.result import (
    AuditReport,
    BlockDecision,
    OptimizationResult,
    StageRecord,
)
from tokengate.utils.errors import (
    BudgetError,
    ConfigurationError,
    InvalidBlockError,
    OptimizationError,
    TokenGateError,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # optimizer
    "TokenGate",
    # models
    "TokenBlock",
    "OptimizationResult",
    "AuditReport",
    "BlockDecision",
    "StageRecord",
    "OptimizerConfig",
    # token counting
    "TokenCounter",
    "HeuristicTokenCounter",
    # errors
    "TokenGateError",
    "InvalidBlockError",
    "BudgetError",
    "OptimizationError",
    "ConfigurationError",
]
