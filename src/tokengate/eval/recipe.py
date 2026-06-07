"""Pipeline recipes (CP-040).

A :class:`RecipeConfig` is a *named* combination of context-pipeline techniques and
thresholds. It is the developer-facing surface for "which optimization recipe do I run?" —
but it is deliberately **thin**: it compiles down to the one canonical
:class:`~tokengate.core.config.OptimizerConfig`, so there is a single source of truth and a
single validation path (no parallel config tree that can drift).

A recipe = a ``base`` strategy preset + a dict of ``OptimizerConfig`` field overrides. Built-in
recipes cover the common questions developers ask ("is reranking alone enough?",
"how much does dedup actually save?"). Custom recipes are just ``RecipeConfig`` instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from tokengate.core.config import OptimizerConfig
from tokengate.utils.errors import ConfigurationError


@dataclass(frozen=True)
class RecipeConfig:
    """A named pipeline recipe that compiles to an :class:`OptimizerConfig`."""

    name: str
    description: str
    base: str = "balanced"
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_optimizer_config(
        self, *, max_prompt_tokens: int | None = None, **extra: Any
    ) -> OptimizerConfig:
        """Compile to an ``OptimizerConfig`` (validates via its ``__post_init__``)."""
        merged: dict[str, Any] = {**self.overrides, **extra}
        if max_prompt_tokens is not None:
            merged["max_prompt_tokens"] = max_prompt_tokens
        try:
            return OptimizerConfig.for_strategy(self.base, **merged)
        except ConfigurationError as exc:
            raise ConfigurationError(f"recipe {self.name!r}: {exc}") from exc

    def validate(self) -> None:
        """Raise ``ConfigurationError`` if the recipe can't compile to a valid config."""
        self.to_optimizer_config()

    def with_overrides(self, *, name: str | None = None, **overrides: Any) -> RecipeConfig:
        """Return a new recipe with extra overrides merged in (for custom tuning/ablation)."""
        merged = {**self.overrides, **overrides}
        return replace(self, name=name or f"{self.name}+custom", overrides=merged)

    # --- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "base": self.base,
            "overrides": dict(self.overrides),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecipeConfig:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            base=data.get("base", "balanced"),
            overrides=dict(data.get("overrides", {})),
        )

    def to_yaml(self) -> str:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ConfigurationError("to_yaml requires PyYAML (pip install pyyaml)") from exc
        return str(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def to_python_snippet(self) -> str:
        """A copy-pasteable snippet that reconstructs this recipe's OptimizerConfig."""
        args = ", ".join(f"{k}={v!r}" for k, v in self.overrides.items())
        head = f'# Recipe: {self.name} — {self.description}'
        call = f'OptimizerConfig.for_strategy("{self.base}"{", " + args if args else ""})'
        return f"{head}\nfrom tokengate.core.config import OptimizerConfig\nconfig = {call}"


# --- built-in recipes -----------------------------------------------------
# Only recipes fully expressible with shipped stages are included. Safety/coverage
# recipes (safety_first, debug_full_trace) arrive with the quality layer (Phase C).

BUILTIN_RECIPES: dict[str, RecipeConfig] = {
    "top_k_only": RecipeConfig(
        "top_k_only",
        "Naive RAG baseline: hybrid rank + token budget only — "
        "no rerank, dedup, MMR, or compression.",
        base="balanced",
        overrides=dict(
            enable_reranker=False, enable_semantic_dedup=False, enable_mmr=False,
            enable_compression=False, adaptive_cutoff=False, relevance_floor=0.0,
        ),
    ),
    "reranker_only": RecipeConfig(
        "reranker_only",
        "Is reranking alone enough? Cross-encoder rerank + adaptive cutoff + budget; "
        "no dedup/MMR/compression.",
        base="balanced",
        overrides=dict(
            enable_semantic_dedup=False, enable_mmr=False, enable_compression=False,
        ),
    ),
    "dedup_focused": RecipeConfig(
        "dedup_focused",
        "Best for duplicate-heavy workspaces: rerank + semantic dedup + budget; "
        "no MMR/compression.",
        base="balanced",
        overrides=dict(enable_mmr=False, enable_compression=False),
    ),
    "diversity_focused": RecipeConfig(
        "diversity_focused",
        "Avoid many chunks from one source: rerank + dedup + MMR (diversity-weighted); "
        "no compression.",
        base="balanced",
        overrides=dict(enable_compression=False, mmr_lambda=0.3),
    ),
    "compression_focused": RecipeConfig(
        "compression_focused",
        "Small windows / huge chunks: full pipeline + aggressive compression + "
        "larger safety margin.",
        base="balanced",
        overrides=dict(enable_compression=True, compression_keep_ratio=0.3, safety_margin=0.10),
    ),
    "speed_first": RecipeConfig(
        "speed_first",
        "Fast approximate optimization: no cross-encoder, no compression, cheap fixed floor.",
        base="speed",
    ),
    "balanced": RecipeConfig(
        "balanced", "Default all-round profile (full pipeline).", base="balanced",
    ),
    "quality": RecipeConfig(
        "quality", "Deepest rerank, semantic-weighted, full diversity — best answers, slower.",
        base="quality",
    ),
    "full_tg": RecipeConfig(
        "full_tg", "Full TokenGate pipeline: every core stage on, compression on.",
        base="balanced", overrides=dict(enable_compression=True),
    ),
}


def get_recipe(recipe: str | RecipeConfig) -> RecipeConfig:
    """Resolve a recipe name or pass through a ``RecipeConfig``."""
    if isinstance(recipe, RecipeConfig):
        return recipe
    try:
        return BUILTIN_RECIPES[recipe]
    except KeyError:
        raise ConfigurationError(
            f"unknown recipe {recipe!r}; built-ins: {sorted(BUILTIN_RECIPES)}"
        ) from None


def list_recipes() -> list[str]:
    return list(BUILTIN_RECIPES)


__all__ = ["RecipeConfig", "BUILTIN_RECIPES", "get_recipe", "list_recipes"]
