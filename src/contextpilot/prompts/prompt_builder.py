"""Assemble the final prompt; stable sections first (CP-011).

Ordering matters for prompt caching: content that is stable across requests (system
instructions, reference material flagged ``cacheable``) is emitted first so an LLM's
prefix cache can be reused, followed by the per-query context blocks, then the query
itself. Output is fully deterministic for a given input order.
"""

from __future__ import annotations

from contextpilot.core.block import ContextBlock


def build_prompt(
    query: str,
    blocks: list[ContextBlock],
    *,
    context_header: str = "Context",
    query_label: str = "Question",
) -> str:
    """Build the final prompt string from the in-prompt blocks.

    ``blocks`` is the ordered set of blocks that survived budgeting (included +
    compressed). Cacheable blocks are placed first (in their given order); the rest go
    into a numbered context section; the query is appended last. Input order is
    otherwise preserved — callers control ranking.
    """
    cacheable = [b for b in blocks if b.cacheable]
    contextual = [b for b in blocks if not b.cacheable]

    sections: list[str] = []

    if cacheable:
        sections.append("\n\n".join(b.content.strip() for b in cacheable))

    if contextual:
        lines = [f"{context_header}:"]
        for i, block in enumerate(contextual, start=1):
            source = f" (source: {block.source_id})" if block.source_id else ""
            lines.append(f"[{i}]{source} {block.content.strip()}")
        sections.append("\n".join(lines))

    sections.append(f"{query_label}: {query.strip()}")

    return "\n\n".join(sections)


__all__ = ["build_prompt"]
