"""CLI entry: python -m contextpilot.dashboard [--store PATH] [--port N] [--no-browser]"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ContextPilot audit dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m contextpilot.dashboard --store ./audits.db --port 8080\n\n"
            "Requires: pip install contextpilot[dashboard]"
        ),
    )
    parser.add_argument(
        "--store",
        default="contextpilot_audits.db",
        metavar="PATH",
        help="SQLite audit store path (default: contextpilot_audits.db)",
    )
    parser.add_argument("--port", type=int, default=8080, metavar="N")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open the browser automatically")
    args = parser.parse_args()

    from contextpilot.dashboard.audit_store import AuditStore

    store = AuditStore(args.store)
    print(f"ContextPilot Dashboard → http://{args.host}:{args.port}")
    store.serve_dashboard(port=args.port, host=args.host, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
