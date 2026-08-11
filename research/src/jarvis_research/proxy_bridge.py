"""Secret-reading SearXNG-only bridge that injects proxy authentication."""

from __future__ import annotations

import argparse
import os
import socket
import socketserver
import threading
from pathlib import Path

MAX_HEADER = 16 * 1024


class BridgeHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        upstream: socket.socket | None = None
        try:
            raw = bytearray()
            self.request.settimeout(5)
            while b"\r\n\r\n" not in raw and len(raw) <= MAX_HEADER:
                chunk = self.request.recv(4096)
                if not chunk:
                    return
                raw.extend(chunk)
            if len(raw) > MAX_HEADER or b"\r\n\r\n" not in raw or b"\nProxy-Authorization:" in bytes(raw):
                return
            head, tail = bytes(raw).split(b"\r\n\r\n", 1)
            upstream = socket.create_connection((self.server.guard_host, self.server.guard_port), timeout=3)  # type: ignore[attr-defined]
            upstream.sendall(head + b"\r\nProxy-Authorization: Bearer " + self.server.token.encode("ascii") + b"\r\n\r\n" + tail)  # type: ignore[attr-defined]
            self._duplex(upstream)
        except OSError:
            return
        finally:
            if upstream is not None:
                upstream.close()

    def _duplex(self, upstream: socket.socket) -> None:
        stop = threading.Event()

        def relay(source: socket.socket, target: socket.socket) -> None:
            try:
                while not stop.is_set():
                    data = source.recv(64 * 1024)
                    if not data:
                        break
                    target.sendall(data)
            except OSError:
                pass
            finally:
                stop.set()

        worker = threading.Thread(target=relay, args=(upstream, self.request), daemon=True)
        worker.start()
        relay(self.request, upstream)
        worker.join(timeout=1)


class BridgeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], *, guard_host: str, guard_port: int, token: str) -> None:
        self.guard_host = guard_host
        self.guard_port = guard_port
        self.token = token
        super().__init__(address, BridgeHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis-searx-proxy-bridge")
    parser.add_argument("--bind", default="0.0.0.0:8081")
    args = parser.parse_args(argv)
    host, raw_port = args.bind.rsplit(":", 1)
    token = Path(os.environ.get("JARVIS_EGRESS_PROXY_CREDENTIALS_FILE", "/run/secrets/egress_proxy_credentials")).read_text().strip()
    if not 32 <= len(token.encode()) <= 512 or not token.isascii():
        raise ValueError("bridge credential is invalid")
    server = BridgeServer((host, int(raw_port)), guard_host="egress-guard", guard_port=8080, token=token)
    try:
        server.serve_forever(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
