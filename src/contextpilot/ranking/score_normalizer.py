"""Normalize heterogeneous scores to [0, 1] (CP-008).

Min-max scaling so lexical raw counts and other unbounded signals can be combined with
already-normalized semantic similarities. The mapping is relative to the candidate set
passed in one call.
"""

from __future__ import annotations


def normalize_min_max(values: list[float]) -> list[float]:
    """Scale ``values`` into ``[0, 1]`` via min-max.

    Edge cases:
    - empty input -> empty list.
    - all equal: ``1.0`` for every element if that common value is > 0 (equally
      relevant), else ``0.0`` (no signal).
    Results are clamped to ``[0, 1]`` to absorb any floating-point drift.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [1.0 if hi > 0 else 0.0] * len(values)
    span = hi - lo
    return [_clamp((v - lo) / span) for v in values]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


__all__ = ["normalize_min_max"]
