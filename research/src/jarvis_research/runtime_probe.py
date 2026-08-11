"""Fixed in-container probes used by the live qualification runner."""

from __future__ import annotations

import socket
import sys


def direct_route_blocked() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 80), timeout=1.0):
            return False
    except OSError:
        return True


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args == ["direct-route-blocked"]:
        return 0 if direct_route_blocked() else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
