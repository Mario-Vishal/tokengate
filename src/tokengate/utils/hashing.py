"""Stable content hashing (CP-002).

Deterministic across processes and runs (unlike Python's salted ``hash()``), so it is
safe for deriving stable block ids and dedup keys.
"""

from __future__ import annotations

import hashlib

# Default length (hex chars) for short, human-friendly ids derived from content.
_DEFAULT_SHORT_LEN = 16


def hash_content(text: str) -> str:
    """Return the full SHA-256 hex digest of ``text`` (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, length: int = _DEFAULT_SHORT_LEN) -> str:
    """Return a truncated SHA-256 hex digest of ``text``.

    Used for compact, stable block ids. ``length`` is clamped to ``[1, 64]``.
    """
    length = max(1, min(length, 64))
    return hash_content(text)[:length]


def block_id_from_content(content: str, *, prefix: str = "blk") -> str:
    """Derive a stable block id from its content, e.g. ``"blk_1a2b3c..."``."""
    return f"{prefix}_{short_hash(content)}"


__all__ = ["hash_content", "short_hash", "block_id_from_content"]
