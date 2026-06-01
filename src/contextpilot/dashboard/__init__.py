"""ContextPilot audit dashboard (optional extra: ``pip install contextpilot[dashboard]``).

Records ``optimize()`` calls to SQLite and serves a self-hosted web dashboard
for deep audit inspection — sessions, queries, per-stage funnel, per-block decisions.

Quickstart::

    from contextpilot import ContextPilot
    from contextpilot.dashboard import AuditStore

    store = AuditStore("audits.db")
    pilot = ContextPilot(max_prompt_tokens=4096, strategy="balanced")

    result = pilot.optimize(query, blocks)
    store.record("my-session", query, result, config={"strategy": "balanced", "max_prompt_tokens": 4096})

    store.serve_dashboard(port=8080)   # blocks; opens browser automatically
    # or: python -m contextpilot.dashboard --store audits.db --port 8080
"""

from contextpilot.dashboard.audit_store import AuditStore

__all__ = ["AuditStore"]
