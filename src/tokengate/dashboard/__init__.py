"""TokenGate audit dashboard (optional extra: ``pip install tokengate[dashboard]``).

Records ``optimize()`` calls to SQLite and serves a self-hosted web dashboard
for deep audit inspection — sessions, queries, per-stage funnel, per-block decisions.

Quickstart::

    from tokengate import TokenGate
    from tokengate.dashboard import AuditStore

    store = AuditStore("audits.db")
    pilot = TokenGate(max_prompt_tokens=4096, strategy="balanced")

    result = pilot.optimize(query, blocks)
    store.record("my-session", query, result, config={"strategy": "balanced", "max_prompt_tokens": 4096})

    store.serve_dashboard(port=8080)   # blocks; opens browser automatically
    # or: python -m tokengate.dashboard --store audits.db --port 8080
"""

from tokengate.dashboard.audit_store import AuditStore

__all__ = ["AuditStore"]
