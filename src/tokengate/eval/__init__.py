"""TokenGate evaluation: pipeline recipes, recipe comparison, and ablation (CP-040/041).

This subpackage is the "context pipeline lab": run a named recipe, compare recipes on the
same candidate blocks, and ablate stages to see which actually produces the savings. It is
importable and dependency-light; optional baseline adapters (LLMLingua, LangChain, …) plug
in here without becoming core dependencies.
"""

from __future__ import annotations

from tokengate.eval.compare import (
    AblationResult,
    RecipeComparisonResult,
    RecipeRunResult,
    ablation,
    compare_recipes,
    run_recipe,
)
from tokengate.eval.recipe import (
    BUILTIN_RECIPES,
    RecipeConfig,
    get_recipe,
    list_recipes,
)
from tokengate.eval.report import (
    ablation_to_markdown,
    comparison_to_markdown,
    to_json,
)

__all__ = [
    "RecipeConfig",
    "BUILTIN_RECIPES",
    "get_recipe",
    "list_recipes",
    "RecipeRunResult",
    "RecipeComparisonResult",
    "AblationResult",
    "run_recipe",
    "compare_recipes",
    "ablation",
    "comparison_to_markdown",
    "ablation_to_markdown",
    "to_json",
]
