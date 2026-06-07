"""Render recipe comparison + ablation results as Markdown / JSON (CP-041).

Plain text-rendering only — the dashboard reuses the same ``to_dict()`` payloads for its
charts, so there is one source of truth for the numbers.
"""

from __future__ import annotations

import json
from typing import Any

from tokengate.eval.compare import AblationResult, RecipeComparisonResult


def comparison_to_markdown(comp: RecipeComparisonResult) -> str:
    """A recipe comparison table (the "money shot")."""
    lines = [
        f"### Recipe comparison — {comp.query!r}",
        f"_{comp.candidate_blocks} candidate blocks · {comp.candidate_tokens} candidate tokens "
        f"· objective: {comp.objective}_",
        "",
        "| Recipe | Final tokens | Saved % | Incl | Compr | Dropped | Latency | Best? |",
        "|---|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for r in comp.runs:
        star = "✅" if r.recipe == comp.recommended else ""
        lines.append(
            f"| {r.recipe} | {r.final_prompt_tokens} | {r.tokens_saved_percent:.1f}% | "
            f"{r.included} | {r.compressed} | {r.dropped} | {r.latency_ms:.0f}ms | {star} |"
        )
    if comp.recommended:
        lines += ["", f"**Recommended ({comp.objective}): `{comp.recommended}`**"]
    return "\n".join(lines)


def ablation_to_markdown(ab: AblationResult) -> str:
    """Per-stage 'turn it off and see' table — answers where savings come from."""
    lines = [
        f"### Ablation — base `{ab.base_recipe}` ({ab.base_tokens} tokens) — {ab.query!r}",
        "",
        "| Stage disabled | Tokens without it | Δ tokens | Δ % | Carries weight? |",
        "|---|---:|---:|---:|:--:|",
    ]
    for stage, d in sorted(ab.deltas.items(), key=lambda kv: -kv[1]["delta"]):
        verdict = "yes" if d["delta"] > 0 else "~none"
        lines.append(
            f"| {stage} | {int(d['tokens_without'])} | +{int(d['delta'])} | "
            f"{d['delta_percent']:.1f}% | {verdict} |"
        )
    return "\n".join(lines)


def to_json(obj: RecipeComparisonResult | AblationResult, *, indent: int = 2) -> str:
    payload: dict[str, Any] = obj.to_dict()
    return json.dumps(payload, indent=indent, ensure_ascii=False)


__all__ = ["comparison_to_markdown", "ablation_to_markdown", "to_json"]
