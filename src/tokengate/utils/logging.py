"""Structured logger factory (CP-002).

The library follows the standard convention of *not* configuring logging on import:
each logger gets a :class:`logging.NullHandler` so applications stay in control of
output. :func:`log_event` provides lightweight structured logging via the ``extra``
mechanism (key/value fields attached to the record).
"""

from __future__ import annotations

import logging
from typing import Any

_ROOT = "tokengate"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced library logger with a NullHandler attached.

    ``get_logger("ranking")`` -> logger ``"tokengate.ranking"``. Passing ``None``
    returns the root ``"tokengate"`` logger.
    """
    full = _ROOT if not name else f"{_ROOT}.{name}"
    logger = logging.getLogger(full)
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a structured log record: an ``event`` name plus arbitrary ``fields``.

    Fields are attached under ``record.event`` and ``record.fields`` so a structured
    (e.g. JSON) handler in the host application can serialize them.
    """
    logger.log(level, event, extra={"event": event, "fields": fields})


__all__ = ["get_logger", "log_event"]
