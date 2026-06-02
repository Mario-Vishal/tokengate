"""CLI entry: python -m tokengate.dashboard [--store PATH] [--port N] [--no-browser]"""

from __future__ import annotations

import argparse


def _beacon_paths() -> tuple[str, str]:
    """Return (beacon_db_path, audit_db_path) for the standard Beacon data dir."""
    import os
    import sys
    from pathlib import Path

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Beacon"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "Beacon"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = (Path(xdg) if xdg else Path.home() / ".local" / "share") / "Beacon"

    return str(base / "beacon.db"), str(base / "tokengate_audits.db")


def _default_store_path() -> str:
    """Return the Beacon audit DB path if it exists, else a local fallback."""
    from pathlib import Path
    _, audit_db = _beacon_paths()
    return audit_db if Path(audit_db).exists() else "tokengate_audits.db"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TokenGate audit dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m tokengate.dashboard --store ./audits.db --port 8080\n\n"
            "Requires: pip install tokengate[dashboard]"
        ),
    )
    parser.add_argument(
        "--store",
        default=None,
        metavar="PATH",
        help="SQLite audit store path (default: Beacon app-data dir, or ./tokengate_audits.db)",
    )
    parser.add_argument("--port", type=int, default=8080, metavar="N")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open the browser automatically")
    args = parser.parse_args()

    from tokengate.dashboard.audit_store import AuditStore

    store_path = args.store or _default_store_path()
    store = AuditStore(store_path)
    print(f"TokenGate Dashboard  http://{args.host}:{args.port}  (store: {store_path})")

    # Auto-sync any Beacon chats that aren't in the audit store yet.
    from pathlib import Path
    beacon_db, _ = _beacon_paths()
    if Path(beacon_db).exists():
        n = store.sync_from_beacon_db(beacon_db)
        if n > 0:
            print(f"  synced {n} new queries from Beacon")

    store.serve_dashboard(port=args.port, host=args.host, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
