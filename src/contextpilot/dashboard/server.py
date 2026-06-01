"""FastAPI dashboard server (CP-029). Requires ``contextpilot[dashboard]``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_STATIC = Path(__file__).parent / "static"


def create_app(store: Any) -> Any:
    """Build the FastAPI app wired to *store*. Separated for testing."""
    from fastapi import FastAPI, HTTPException  # type: ignore[import]
    from fastapi.responses import HTMLResponse  # type: ignore[import]

    app = FastAPI(title="ContextPilot Dashboard", docs_url=None, redoc_url=None)

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        return store.global_stats()

    @app.get("/api/sessions")
    def sessions() -> list[dict[str, Any]]:
        return store.sessions()

    @app.get("/api/sessions/{session_id}")
    def session_queries(session_id: str) -> list[dict[str, Any]]:
        return store.session_queries(session_id)

    @app.get("/api/queries/{query_id}")
    def query_detail(query_id: str) -> dict[str, Any]:
        detail = store.query_detail(query_id)
        if not detail:
            raise HTTPException(status_code=404, detail="Query not found")
        return detail

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((_STATIC / "index.html").read_text(encoding="utf-8"))

    return app


def serve(
    store: Any,
    *,
    port: int = 8080,
    host: str = "127.0.0.1",
    open_browser: bool = True,
) -> None:
    """Run the dashboard server (blocking)."""
    import uvicorn  # type: ignore[import]

    app = create_app(store)
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(
            0.9, lambda: webbrowser.open(f"http://{host}:{port}")
        ).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


__all__ = ["create_app", "serve"]
