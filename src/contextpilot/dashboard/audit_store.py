"""AuditStore — SQLite-backed audit recorder for the ContextPilot dashboard (CP-029).

Thread-safe: each operation opens its own connection (sqlite3 is file-locking). All
audit data is stored as compact JSON so no schema migrations are needed when the audit
shape evolves. The store is append-only; the dashboard is read-only.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from collections.abc import Generator
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from contextpilot.core.result import OptimizationResult


class AuditStore:
    """Record ``optimize()`` calls and query them for the dashboard.

    The store is intentionally lightweight: one SQLite file, no ORM. Core library
    stays pure — this class lives under the optional ``[dashboard]`` extra.

    Example::

        store = AuditStore("audits.db")
        result = pilot.optimize(query, blocks)
        store.record("session-1", query, result,
                     config={"strategy": "balanced", "max_prompt_tokens": 4096})
        store.serve_dashboard()   # opens http://127.0.0.1:8080 in the browser
    """

    def __init__(self, path: str | Path = "contextpilot_audits.db") -> None:
        self.path = Path(path)
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    first_seen   REAL NOT NULL,
                    last_seen    REAL NOT NULL,
                    meta         TEXT
                );
                CREATE TABLE IF NOT EXISTS queries (
                    query_id     TEXT PRIMARY KEY,
                    session_id   TEXT NOT NULL REFERENCES sessions(session_id),
                    created_at   REAL NOT NULL,
                    query_text   TEXT NOT NULL,
                    audit_json   TEXT NOT NULL,
                    config_json  TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_queries_session
                    ON queries(session_id, created_at);
            """)

    def record(
        self,
        session_id: str,
        query: str,
        result: OptimizationResult,
        *,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Persist one ``optimize()`` call. Returns the generated ``query_id``."""
        now = time()
        query_id = uuid4().hex[:16]
        audit_dict = result.audit.to_dict() if result.audit else {}
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions VALUES (?, ?, ?, ?)",
                (session_id, now, now, None),
            )
            conn.execute(
                "UPDATE sessions SET last_seen=? WHERE session_id=?",
                (now, session_id),
            )
            conn.execute(
                "INSERT INTO queries VALUES (?, ?, ?, ?, ?, ?)",
                (query_id, session_id, now, query,
                 json.dumps(audit_dict),
                 json.dumps(config) if config else None),
            )
        return query_id

    def sessions(self) -> list[dict[str, Any]]:
        """Return session summaries (most-recent first)."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT
                    s.session_id, s.first_seen, s.last_seen,
                    COUNT(q.query_id) AS query_count,
                    COALESCE(SUM(json_extract(q.audit_json, '$.tokens_saved')), 0)
                        AS total_saved,
                    COALESCE(SUM(json_extract(q.audit_json, '$.total_candidate_tokens')), 0)
                        AS total_candidates,
                    COALESCE(AVG(json_extract(q.audit_json, '$.tokens_saved_percent')), 0.0)
                        AS avg_saved_pct
                FROM sessions s
                LEFT JOIN queries q ON q.session_id = s.session_id
                GROUP BY s.session_id
                ORDER BY s.last_seen DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def session_queries(self, session_id: str) -> list[dict[str, Any]]:
        """Return query summaries for a session (chronological)."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT
                    query_id, created_at, query_text,
                    json_extract(audit_json, '$.total_candidate_blocks')  AS candidates,
                    json_extract(audit_json, '$.final_prompt_tokens')     AS final_tokens,
                    json_extract(audit_json, '$.total_candidate_tokens')  AS candidate_tokens,
                    json_extract(audit_json, '$.tokens_saved')            AS tokens_saved,
                    json_extract(audit_json, '$.tokens_saved_percent')    AS saved_pct,
                    json_extract(audit_json, '$.included_count')          AS included,
                    json_extract(audit_json, '$.compressed_count')        AS compressed,
                    json_extract(audit_json, '$.dropped_count')           AS dropped,
                    json_extract(config_json, '$.strategy')               AS strategy,
                    json_extract(config_json, '$.max_prompt_tokens')      AS max_prompt_tokens
                FROM queries
                WHERE session_id=?
                ORDER BY created_at
            """, (session_id,)).fetchall()
        return [dict(r) for r in rows]

    def query_detail(self, query_id: str) -> dict[str, Any] | None:
        """Full record for one query, including the complete audit dict."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM queries WHERE query_id=?", (query_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["audit"] = json.loads(d.pop("audit_json"))
        raw_cfg = d.pop("config_json", None)
        d["config"] = json.loads(raw_cfg) if raw_cfg else None
        return d

    def global_stats(self) -> dict[str, Any]:
        """Aggregate stats across all sessions and queries."""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(DISTINCT session_id)                                AS total_sessions,
                    COUNT(*)                                                  AS total_queries,
                    COALESCE(SUM(json_extract(audit_json, '$.tokens_saved')), 0)
                        AS total_saved,
                    COALESCE(SUM(json_extract(audit_json, '$.total_candidate_tokens')), 0)
                        AS total_candidates,
                    COALESCE(AVG(json_extract(audit_json, '$.tokens_saved_percent')), 0.0)
                        AS avg_saved_pct
                FROM queries
            """).fetchone()
        return dict(row) if row else {}

    def serve_dashboard(
        self,
        port: int = 8080,
        host: str = "127.0.0.1",
        *,
        open_browser: bool = True,
    ) -> None:
        """Launch the dashboard HTTP server (blocking). Requires ``contextpilot[dashboard]``."""
        from contextpilot.dashboard.server import serve  # type: ignore[import]

        serve(self, port=port, host=host, open_browser=open_browser)


__all__ = ["AuditStore"]
