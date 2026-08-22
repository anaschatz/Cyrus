"""Authenticated, resolving, pinned-IP HTTP/CONNECT egress proxy."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import select
import socket
import socketserver
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlsplit

from jarvis_orchestrator.domain import canonical_json
from jarvis_orchestrator.research import validate_public_url
from jarvis_orchestrator.research_security import validate_resolved_addresses

MAX_HEADER_BYTES = 16 * 1024
MAX_REQUEST_BODY_BYTES = 64 * 1024
MAX_TUNNEL_BYTES = 4 * 1024 * 1024
TUNNEL_IDLE_SECONDS = 10.0
CONNECT_SECONDS = 3.0
ALLOWED_PORTS = frozenset({80, 443})
_HOP = frozenset({"connection", "proxy-connection", "proxy-authorization", "keep-alive", "te", "trailer", "transfer-encoding", "upgrade"})


class GuardError(Exception):
    def __init__(self, code: str, status: int = 403) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


@dataclass(frozen=True, slots=True)
class PinnedTargetV1:
    host: str
    port: int
    family: int
    address: str


@dataclass(frozen=True, slots=True)
class GuardConfigV1:
    bind_host: str
    bind_port: int
    bearer_token: str
    internal_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = ()

    @classmethod
    def from_environment(cls, env: dict[str, str] | None = None) -> "GuardConfigV1":
        values = dict(os.environ if env is None else env)
        bind = values.get("JARVIS_EGRESS_BIND", "127.0.0.1:8080")
        try:
            host, raw_port = bind.rsplit(":", 1)
            port = int(raw_port)
        except (ValueError, TypeError) as exc:
            raise ValueError("egress bind is invalid") from exc
        if host not in {"127.0.0.1", "::1", "0.0.0.0"} or not 1 <= port <= 65535:
            raise ValueError("egress bind is invalid")
        token_path = Path(values.get("JARVIS_EGRESS_PROXY_CREDENTIALS_FILE", "/run/secrets/egress_proxy_credentials"))
        token = token_path.read_text(encoding="utf-8").strip()
        if not 32 <= len(token.encode("utf-8")) <= 512 or any(ch in token for ch in "\r\n"):
            raise ValueError("egress bearer token is invalid")
        raw_networks = values.get("JARVIS_EGRESS_INTERNAL_NETWORKS", "")
        try:
            networks = tuple(ipaddress.ip_network(item.strip(), strict=True) for item in raw_networks.split(",") if item.strip())
        except ValueError as exc:
            raise ValueError("egress internal networks are invalid") from exc
        return cls(host, port, token, networks)


class GuardPolicyV1:
    def __init__(
        self,
        config: GuardConfigV1,
        *,
        resolver: Callable[[str, int], Iterable[tuple[int, str]]] | None = None,
        connector: Callable[[PinnedTargetV1], socket.socket] | None = None,
        audit: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.config = config
        self._resolver = resolver or self._system_resolve
        self._connector = connector or self._system_connect
        self._audit = audit or (lambda event: logging.getLogger("jarvis_egress.audit").info(canonical_json(event)))

    def authenticated(self, value: str | None) -> bool:
        return value is not None and hmac.compare_digest(value, "Bearer " + self.config.bearer_token)

    def target_for_url(self, value: str) -> tuple[str, PinnedTargetV1]:
        try:
            safe = validate_public_url(value)
        except ValueError as exc:
            raise GuardError("target_invalid") from exc
        parsed = urlsplit(safe)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return safe, self.resolve(parsed.hostname or "", port)

    def target_for_connect(self, authority: str) -> PinnedTargetV1:
        if any(character.isspace() or ord(character) < 33 for character in authority) or "@" in authority:
            raise GuardError("connect_authority_invalid")
        try:
            parsed = urlsplit("//" + authority)
            host, port = parsed.hostname, parsed.port
        except ValueError as exc:
            raise GuardError("connect_authority_invalid") from exc
        if not host or port not in ALLOWED_PORTS or parsed.username is not None:
            raise GuardError("connect_authority_invalid")
        try:
            validate_public_url(f"https://{authority}/")
        except ValueError as exc:
            raise GuardError("connect_authority_invalid") from exc
        return self.resolve(host, port)

    def resolve(self, host: str, port: int) -> PinnedTargetV1:
        if port not in ALLOWED_PORTS:
            raise GuardError("target_port_forbidden")
        try:
            records = tuple(self._resolver(host, port))
            validated = validate_resolved_addresses(address for _, address in records)
        except (OSError, ValueError) as exc:
            code = "dns_answer_forbidden" if isinstance(exc, ValueError) else "dns_resolution_failed"
            raise GuardError(code) from exc
        by_address = {ipaddress.ip_address(address).compressed: family for family, address in records}
        for address in validated:
            parsed = ipaddress.ip_address(address)
            if any(parsed in network for network in self.config.internal_networks):
                raise GuardError("dns_container_internal")
        chosen = validated[0]
        return PinnedTargetV1(host, port, by_address[chosen], chosen)

    def connect(self, target: PinnedTargetV1) -> socket.socket:
        try:
            return self._connector(target)
        except OSError as exc:
            raise GuardError("upstream_connect_failed", 502) from exc

    def emit(self, code: str, started: float, target: PinnedTargetV1 | None, byte_count: int = 0) -> None:
        identity = "" if target is None else hashlib.sha256(f"{target.host}:{target.port}".encode()).hexdigest()
        self._audit({"bytes": byte_count, "code": code, "duration_ms": int((time.monotonic()-started)*1000), "target_sha256": identity})

    @staticmethod
    def _system_resolve(host: str, port: int) -> Iterable[tuple[int, str]]:
        seen: set[tuple[int, str]] = set()
        for family, _, _, _, sockaddr in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
            seen.add((family, sockaddr[0]))
        return tuple(sorted(seen, key=lambda item: (item[0], item[1])))

    @staticmethod
    def _system_connect(target: PinnedTargetV1) -> socket.socket:
        outgoing = socket.socket(target.family, socket.SOCK_STREAM)
        outgoing.settimeout(CONNECT_SECONDS)
        sockaddr = (target.address, target.port, 0, 0) if target.family == socket.AF_INET6 else (target.address, target.port)
        outgoing.connect(sockaddr)
        outgoing.settimeout(TUNNEL_IDLE_SECONDS)
        return outgoing


def parse_headers(raw: bytes) -> tuple[str, str, str, dict[str, str]]:
    if len(raw) > MAX_HEADER_BYTES or b"\r\n\r\n" not in raw:
        raise GuardError("request_headers_invalid", 400)
    try:
        lines = raw.split(b"\r\n\r\n", 1)[0].decode("ascii", errors="strict").split("\r\n")
        method, target, version = lines[0].split(" ")
    except (UnicodeDecodeError, ValueError) as exc:
        raise GuardError("request_line_invalid", 400) from exc
    if method not in {"GET", "HEAD", "POST", "CONNECT"} or version not in {"HTTP/1.0", "HTTP/1.1"}:
        raise GuardError("request_method_invalid", 405)
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line:
            raise GuardError("request_headers_invalid", 400)
        name, value = line.split(":", 1)
        key = name.strip().casefold()
        if not key or key in headers or any(ord(ch) < 32 for ch in value):
            raise GuardError("request_headers_invalid", 400)
        headers[key] = value.strip()
    return method, target, version, headers


class EgressGuardHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        started = time.monotonic()
        target: PinnedTargetV1 | None = None
        try:
            raw = self._read_headers()
            method, request_target, _, headers = parse_headers(raw)
            if not self.server.policy.authenticated(headers.get("proxy-authorization")):  # type: ignore[attr-defined]
                raise GuardError("proxy_authentication_required", 407)
            try:
                content_length = int(headers.get("content-length", "0"))
            except ValueError as exc:
                raise GuardError("request_body_forbidden", 400) from exc
            if (
                headers.get("transfer-encoding") is not None
                or not 0 <= content_length <= MAX_REQUEST_BODY_BYTES
                or (method != "POST" and content_length != 0)
            ):
                raise GuardError("request_body_forbidden", 400)
            if method == "CONNECT":
                target = self.server.policy.target_for_connect(request_target)  # type: ignore[attr-defined]
                upstream = self.server.policy.connect(target)  # type: ignore[attr-defined]
                self.request.sendall(b"HTTP/1.1 200 Connection Established\r\nConnection: close\r\n\r\n")
                count = self._tunnel(upstream)
            else:
                safe, target = self.server.policy.target_for_url(request_target)  # type: ignore[attr-defined]
                upstream = self.server.policy.connect(target)  # type: ignore[attr-defined]
                parsed = urlsplit(safe)
                path = (parsed.path or "/") + ("?" + parsed.query if parsed.query else "")
                outgoing = [f"{method} {path} HTTP/1.1", f"Host: {parsed.hostname}" + (f":{parsed.port}" if parsed.port else "")]
                for name, value in headers.items():
                    if name not in _HOP and name != "host":
                        outgoing.append(f"{name}: {value}")
                outgoing.extend(("Connection: close", "", ""))
                head, buffered = raw.split(b"\r\n\r\n", 1)
                del head
                body = bytearray(buffered[:content_length])
                while len(body) < content_length:
                    chunk = self.request.recv(content_length - len(body))
                    if not chunk:
                        raise GuardError("request_body_incomplete", 400)
                    body.extend(chunk)
                upstream.sendall("\r\n".join(outgoing).encode("ascii") + bytes(body))
                count = self._relay_response(upstream)
            self.server.policy.emit("proxy_complete", started, target, count)  # type: ignore[attr-defined]
        except GuardError as exc:
            self._error(exc.status, exc.code)
            self.server.policy.emit(exc.code, started, target)  # type: ignore[attr-defined]
        except OSError:
            self._error(502, "proxy_io_failed")
            self.server.policy.emit("proxy_io_failed", started, target)  # type: ignore[attr-defined]

    def _read_headers(self) -> bytes:
        value = bytearray()
        self.request.settimeout(3.0)
        while b"\r\n\r\n" not in value:
            chunk = self.request.recv(min(4096, MAX_HEADER_BYTES + 1 - len(value)))
            if not chunk:
                break
            value.extend(chunk)
            if len(value) > MAX_HEADER_BYTES:
                break
        return bytes(value)

    def _relay_response(self, upstream: socket.socket) -> int:
        total = 0
        try:
            while True:
                chunk = upstream.recv(64 * 1024)
                if not chunk:
                    return total
                total += len(chunk)
                self.request.sendall(chunk)
        finally:
            upstream.close()

    def _tunnel(self, upstream: socket.socket) -> int:
        sockets = (self.request, upstream)
        total = 0
        try:
            while total <= MAX_TUNNEL_BYTES:
                readable, _, _ = select.select(sockets, (), (), TUNNEL_IDLE_SECONDS)
                if not readable:
                    raise GuardError("tunnel_idle_timeout", 504)
                for source in readable:
                    chunk = source.recv(min(64 * 1024, MAX_TUNNEL_BYTES + 1 - total))
                    if not chunk:
                        return total
                    total += len(chunk)
                    if total > MAX_TUNNEL_BYTES:
                        raise GuardError("tunnel_byte_limit", 413)
                    (upstream if source is self.request else self.request).sendall(chunk)
            raise GuardError("tunnel_byte_limit", 413)
        finally:
            upstream.close()

    def _error(self, status: int, code: str) -> None:
        body = canonical_json({"error": {"code": code}}).encode()
        reason = {400: "Bad Request", 403: "Forbidden", 405: "Method Not Allowed", 407: "Proxy Authentication Required", 413: "Content Too Large", 502: "Bad Gateway", 504: "Gateway Timeout"}.get(status, "Forbidden")
        try:
            self.request.sendall(f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
        except OSError:
            pass


class EgressGuardServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], policy: GuardPolicyV1) -> None:
        self.policy = policy
        super().__init__(address, EgressGuardHandler)


def serve(config: GuardConfigV1 | None = None) -> None:
    runtime = config or GuardConfigV1.from_environment()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server = EgressGuardServer((runtime.bind_host, runtime.bind_port), GuardPolicyV1(runtime))
    try:
        server.serve_forever(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cyrus-egress-guard")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args(argv)
    config = GuardConfigV1.from_environment()
    if args.check_config:
        return 0
    serve(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
