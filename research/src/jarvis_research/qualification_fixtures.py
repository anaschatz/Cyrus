"""Deterministic, offline HTTP/search/DNS fixtures for the qualification profile."""

from __future__ import annotations

import argparse
import gzip
import ipaddress
import json
import socket
import socketserver
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

FIXTURE_IPV4 = "93.184.216.10"
_LONG_EN = ("The verified qualification document contains deterministic English evidence for exact extraction, language eligibility, passage ranking, and bounded corpus construction. " * 5).encode()
_LONG_EL = ("Το επαληθευμένο δοκιμαστικό έγγραφο περιέχει σταθερά ελληνικά τεκμήρια για ακριβή εξαγωγή, γλωσσική επιλεξιμότητα και κατάταξη αποσπασμάτων. " * 5).encode()


class FixtureHTTPHandler(BaseHTTPRequestHandler):
    server_version = "JARVISFixture/1"
    sys_version = ""

    def log_message(self, *_: object) -> None:
        return

    def do_GET(self) -> None:
        host = self.headers.get("Host", "").partition(":")[0].casefold()
        if self.path == "/robots.txt":
            return self._robots(host)
        if self.path == "/redirect-valid":
            return self._reply(302, b"", {"Location": "http://allowed.fixture.test/doc"})
        if self.path == "/redirect-private":
            return self._reply(302, b"", {"Location": "http://private-v4.fixture.test/doc"})
        if self.path == "/mime-pdf":
            return self._reply(200, b"%PDF-fixture", {"Content-Type": "application/pdf"})
        if self.path == "/plain":
            return self._reply(200, _LONG_EN, {"Content-Type": "text/plain"})
        if self.path == "/xhtml":
            return self._reply(200, b"<html><body><p>" + _LONG_EN + b"</p></body></html>", {"Content-Type": "application/xhtml+xml"})
        if self.path == "/greek":
            return self._html(_LONG_EL, date="2026-08-10T10:00:00Z")
        if self.path == "/unsupported":
            return self._html(("Ceci est un document en langue française qui ne doit jamais devenir une preuve éligible. " * 6).encode())
        if self.path == "/short":
            return self._html(b"Too short.")
        if self.path == "/date-conflict":
            metadata = b"<meta property='article:published_time' content='2026-08-10T10:00:00Z'><meta name='datePublished' content='2026-08-09T10:00:00Z'>"
            return self._html(_LONG_EN, metadata=metadata)
        if self.path == "/date-jsonld":
            metadata = b"<script type='application/ld+json'>{\"datePublished\":\"2026-08-10T10:00:00Z\"}</script>"
            return self._html(_LONG_EN, metadata=metadata)
        if self.path == "/gzip-limit":
            body = gzip.compress(b"x" * (2 * 1024 * 1024 + 1))
            return self._reply(200, body, {"Content-Type": "text/plain", "Content-Encoding": "gzip"})
        if self.path == "/body-limit":
            return self._reply(200, b"x" * (2 * 1024 * 1024 + 1), {"Content-Type": "text/plain"})
        if self.path == "/delayed":
            time.sleep(46)
            return self._html(_LONG_EN)
        if self.path.startswith("/transient-"):
            status = int(self.path.rsplit("-", 1)[1])
            key = (host, self.path)
            with self.server.state_lock:  # type: ignore[attr-defined]
                count = self.server.counts.get(key, 0)  # type: ignore[attr-defined]
                self.server.counts[key] = count + 1  # type: ignore[attr-defined]
            if count == 0:
                return self._reply(status, b"")
            return self._html(_LONG_EN)
        return self._html(_LONG_EN, date="2026-08-10T10:00:00Z")

    def _robots(self, host: str) -> None:
        if host.startswith("robots404"):
            return self._reply(404, b"")
        if host.startswith("robots410"):
            return self._reply(410, b"")
        if host.startswith("robots401"):
            return self._reply(401, b"")
        if host.startswith("robots403"):
            return self._reply(403, b"")
        if host.startswith("robots429"):
            return self._reply(429, b"")
        if host.startswith("robots500"):
            return self._reply(500, b"")
        if host.startswith("robots-malformed"):
            return self._reply(200, b"not-a-directive")
        if host.startswith("robots-oversized"):
            return self._reply(200, b"#" * (128 * 1024 + 1))
        if host.startswith("robots-timeout"):
            time.sleep(4)
            return self._reply(200, b"User-agent: *\nAllow: /\n")
        if host.startswith("robots-disallow"):
            return self._reply(200, b"User-agent: *\nDisallow: /doc\n")
        return self._reply(200, b"User-agent: *\nAllow: /\n")

    def _html(self, body: bytes, *, date: str | None = None, metadata: bytes = b"") -> None:
        date_value = b"" if date is None else f"<meta property='article:published_time' content='{date}'>".encode()
        value = b"<html><head><title>Qualification fixture</title>" + date_value + metadata + b"</head><body><main><p>" + body + b"</p></main></body></html>"
        self._reply(200, value, {"Content-Type": "text/html; charset=utf-8"})

    def _reply(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass


class FixtureHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int]) -> None:
        self.counts: dict[tuple[str, str], int] = {}
        self.state_lock = threading.Lock()
        super().__init__(address, FixtureHTTPHandler)


class SearchFixtureHandler(BaseHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        return

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            fields = parse_qs(self.rfile.read(min(length, 4096)).decode("utf-8"))
            query = fields.get("q", [""])[0]
        except (ValueError, UnicodeDecodeError):
            query = ""
        route = _CASE_ROUTES.get(query, "/doc")
        host = _CASE_HOSTS.get(query, "allowed.fixture.test")
        value = {"results": [{"url": f"http://{host}{route}", "title": "Fixture", "content": "ranking only", "engine": "qualification"}]}
        body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


_CASE_HOSTS = {
    "robots_disallow": "robots-disallow.fixture.test", "robots_404": "robots404.fixture.test",
    "robots_410": "robots410.fixture.test", "robots_fail_closed": "robots-malformed.fixture.test",
    "ipv4_private_reserved": "private-v4.fixture.test", "ipv6_private_reserved": "private-v6.fixture.test",
    "mixed_dns_answers": "mixed.fixture.test", "dns_rebinding": "rebind.fixture.test",
}
_CASE_ROUTES = {
    "redirect_revalidation": "/redirect-private", "mime_limit": "/mime-pdf",
    "compressed_body_limit": "/body-limit", "decompressed_body_limit": "/gzip-limit",
    "single_transient_retry": "/transient-503", "deadline": "/delayed", "cancellation": "/delayed",
}


class DNSFixtureHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        data, transport = self.request
        response = self.server.answer(data)  # type: ignore[attr-defined]
        if response:
            transport.sendto(response, self.client_address)


class DNSFixtureServer(socketserver.ThreadingUDPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int]) -> None:
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()
        super().__init__(address, DNSFixtureHandler)

    def answer(self, query: bytes) -> bytes:
        try:
            name, offset = _decode_name(query, 12)
            qtype, qclass = struct.unpack("!HH", query[offset:offset + 4])
        except (ValueError, struct.error):
            return b""
        question = query[12:offset + 4]
        addresses = self._addresses(name.casefold(), qtype)
        answers = []
        for address in addresses:
            packed = ipaddress.ip_address(address).packed
            answers.append(b"\xc0\x0c" + struct.pack("!HHIH", qtype, qclass, 0, len(packed)) + packed)
        flags = 0x8180 if answers else 0x8183
        return query[:2] + struct.pack("!HHHHH", flags, 1, len(answers), 0, 0) + question + b"".join(answers)

    def _addresses(self, name: str, qtype: int) -> tuple[str, ...]:
        if name == "private-v4.fixture.test" and qtype == 1:
            return ("10.0.0.1",)
        if name == "private-v6.fixture.test" and qtype == 28:
            return ("fd00::1",)
        if name == "mixed.fixture.test" and qtype == 1:
            return (FIXTURE_IPV4, "127.0.0.1")
        if name == "rebind.fixture.test" and qtype == 1:
            with self._lock:
                count = self._counts.get(name, 0)
                self._counts[name] = count + 1
            return (FIXTURE_IPV4,) if count == 0 else ("127.0.0.1",)
        if name.endswith(".fixture.test") and qtype == 1:
            return (FIXTURE_IPV4,)
        return ()


def _decode_name(value: bytes, offset: int) -> tuple[str, int]:
    labels: list[str] = []
    while offset < len(value):
        size = value[offset]
        offset += 1
        if size == 0:
            return ".".join(labels), offset
        if size > 63 or offset + size > len(value):
            raise ValueError("dns_name_invalid")
        labels.append(value[offset:offset + size].decode("ascii"))
        offset += size
    raise ValueError("dns_name_invalid")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis-research-fixture")
    parser.add_argument("kind", choices=("http", "search", "dns"))
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    if args.kind == "dns":
        server = DNSFixtureServer((args.bind, args.port or 53))
    elif args.kind == "search":
        server = ThreadingHTTPServer((args.bind, args.port or 8080), SearchFixtureHandler)
    else:
        server = FixtureHTTPServer((args.bind, args.port or 80))
    try:
        server.serve_forever(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
