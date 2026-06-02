"""Auxiliary ranking signals (CP-017): recency, token efficiency, source priority.

Each returns one value per block, in input order, or ``None`` where the signal is
unavailable for that block (so the hybrid ranker can drop its weight and renormalize).
Recency and token-efficiency are *set-relative* (min-max over the candidates).
"""

from __future__ import annotations

from datetime import datetime

from tokengate.core.block import TokenBlock


def _parse_time(value: object) -> float | None:
    """Parse an ISO-8601 string or epoch number into a float timestamp, else None."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def _relative(values: list[float | None], *, invert: bool) -> list[float | None]:
    """Min-max scale present values to [0,1]; ``invert`` flips (small→high). None kept.

    All-equal present values map to 1.0 (equally good on this signal).
    """
    present = [v for v in values if v is not None]
    if not present:
        return [None] * len(values)
    lo, hi = min(present), max(present)
    span = hi - lo
    out: list[float | None] = []
    for v in values:
        if v is None:
            out.append(None)
        elif span == 0:
            out.append(1.0)
        else:
            scaled = (v - lo) / span
            out.append(1.0 - scaled if invert else scaled)
    return out


def recency_scores(blocks: list[TokenBlock]) -> list[float | None]:
    """Newer (larger ``metadata.modified_time``) → higher; None when unparseable."""
    times = [_parse_time(b.metadata.get("modified_time")) for b in blocks]
    return _relative(times, invert=False)


def token_efficiency_scores(blocks: list[TokenBlock]) -> list[float | None]:
    """Fewer tokens → higher (value per token); None when ``token_count`` is unset."""
    counts: list[float | None] = [
        float(b.token_count) if b.token_count is not None else None for b in blocks
    ]
    return _relative(counts, invert=True)


def source_priority_scores(
    blocks: list[TokenBlock], priorities: dict[str, float]
) -> list[float | None]:
    """Look up each block's ``source_id`` in ``priorities``; None when absent."""
    if not priorities:
        return [None] * len(blocks)
    return [
        priorities.get(b.source_id) if b.source_id is not None else None for b in blocks
    ]


__all__ = ["recency_scores", "token_efficiency_scores", "source_priority_scores"]
