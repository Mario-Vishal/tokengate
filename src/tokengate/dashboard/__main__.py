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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TokenGate audit dashboard — inspect what each optimize() call kept, "
                    "dropped, and why, across sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m tokengate.dashboard                      # ./tokengate_audits.db\n"
            "  python -m tokengate.dashboard --store ./audits.db --port 8080\n"
            "  python -m tokengate.dashboard --beacon-sync        # also pull Beacon chats\n\n"
            "Requires: pip install tokengate[dashboard]"
        ),
    )
    parser.add_argument(
        "--store",
        default="tokengate_audits.db",
        metavar="PATH",
        help="SQLite audit store path (default: ./tokengate_audits.db)",
    )
    parser.add_argument("--port", type=int, default=8080, metavar="N")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open the browser automatically")
    parser.add_argument(
        "--beacon-sync", action="store_true",
        help="Also import audits from the Beacon desktop app's local DB, if present "
             "(off by default — the dashboard is standalone).",
    )
    parser.add_argument(
        "--beacon-db", default=None, metavar="PATH",
        help="Explicit Beacon DB path to sync from (implies --beacon-sync).",
    )
    args = parser.parse_args()

    from pathlib import Path

    from tokengate.dashboard.audit_store import AuditStore

    store = AuditStore(args.store)
    print(f"TokenGate Dashboard  http://{args.host}:{args.port}  (store: {args.store})")

    # Beacon sync is opt-in: only when asked, so the standalone dashboard never depends
    # on Beacon being installed.
    if args.beacon_sync or args.beacon_db:
        beacon_db = args.beacon_db or _beacon_paths()[0]
        if Path(beacon_db).exists():
            n = store.sync_from_beacon_db(beacon_db)
            print(f"  synced {n} new queries from Beacon ({beacon_db})")
        else:
            print(f"  --beacon-sync: no Beacon DB found at {beacon_db}")

    store.serve_dashboard(port=args.port, host=args.host, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
