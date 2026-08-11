"""Fixed in-container probes used by the live qualification runner."""

from __future__ import annotations

import json
import hashlib
import os
import socket
import sys
import threading
import time
import urllib.parse
import urllib.error
import urllib.request

from jarvis_research.adapters import DirectHTTPFetcher
from jarvis_orchestrator.research_security import RetryBudgetV1


def direct_route_blocked() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 80), timeout=1.0):
            return False
    except OSError:
        return True


def _proxy_exchange(request: bytes, *, read_limit: int = 64 * 1024, timeout: float = 5.0) -> bytes:
    with socket.create_connection(("egress-guard", 8080), timeout=3) as connection:
        connection.settimeout(timeout)
        connection.sendall(request)
        value = bytearray()
        while len(value) < read_limit:
            try:
                chunk = connection.recv(min(8192, read_limit - len(value)))
            except socket.timeout:
                break
            if not chunk:
                break
            value.extend(chunk)
        return bytes(value)


def _status(value: bytes) -> int:
    try:
        return int(value.split(b"\r\n", 1)[0].split()[1])
    except (IndexError, ValueError):
        return 0


def _egress_matrix() -> bool:
    """Exercise authenticated HTTP/CONNECT and address policy with typed output only."""

    try:
        token = open(os.environ["JARVIS_EGRESS_PROXY_CREDENTIALS_FILE"], encoding="utf-8").read().strip()

        def http(target: str, credential: str = token) -> bytes:
            host = target.split("/", 3)[2]
            return _proxy_exchange(
                (
                    f"GET {target} HTTP/1.1\r\nHost: {host}\r\n"
                    f"Proxy-Authorization: Bearer {credential}\r\nConnection: close\r\n\r\n"
                ).encode("ascii")
            )

        checks = {
            "authenticated_http": _status(http("http://allowed.fixture.test/doc")) == 200,
            "public_ipv4": _status(http("http://allowed.fixture.test/plain")) == 200,
            "public_ipv6": _status(http("http://public-v6.fixture.test/doc")) == 200,
            "http_auth_rejected": _status(http("http://allowed.fixture.test/doc", "invalid")) == 407,
            "private_ipv4_rejected": _status(http("http://private-v4.fixture.test/doc")) == 403,
            "private_ipv6_rejected": _status(http("http://private-v6.fixture.test/doc")) == 403,
            "mixed_dns_rejected": _status(http("http://mixed.fixture.test/doc")) == 403,
            "port_rejected": _status(http("http://allowed.fixture.test:81/doc")) == 403,
            "environment_routes_disabled": not any(
                name in os.environ
                for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
            ),
        }
        # A second resolution must fail even if the first rebinding answer was public.
        http("http://rebind.fixture.test/doc")
        checks["dns_rebinding_rejected"] = _status(http("http://rebind.fixture.test/doc")) == 403

        def connect(credential: str = token) -> tuple[socket.socket, bytes]:
            connection = socket.create_connection(("egress-guard", 8080), timeout=3)
            connection.settimeout(12)
            connection.sendall(
                (
                    "CONNECT tunnel.fixture.test:443 HTTP/1.1\r\nHost: tunnel.fixture.test:443\r\n"
                    f"Proxy-Authorization: Bearer {credential}\r\n\r\n"
                ).encode("ascii")
            )
            header = bytearray()
            while b"\r\n\r\n" not in header and len(header) < 16 * 1024:
                header.extend(connection.recv(4096))
            return connection, bytes(header)

        connection, header = connect()
        checks["authenticated_connect"] = _status(header) == 200
        connection.close()
        connection, header = connect("invalid")
        checks["connect_auth_rejected"] = _status(header) == 407
        checks["no_target_leakage"] = b"fixture.test" not in header
        connection.close()

        connection, header = connect()
        byte_limited = False
        try:
            connection.sendall(b"x" * (4 * 1024 * 1024 + 1))
            byte_limited = not connection.recv(1)
        except OSError:
            byte_limited = True
        finally:
            connection.close()
        checks["connect_byte_limit"] = _status(header) == 200 and byte_limited

        connection, header = connect()
        started = time.monotonic()
        connection.sendall(b"idle")
        try:
            idle_value = connection.recv(4096)
        except (OSError, socket.timeout):
            idle_value = b""
        checks["connect_idle_timeout"] = _status(header) == 200 and time.monotonic() - started >= 9.5 and (
            not idle_value or b"tunnel_idle_timeout" in idle_value
        )
        connection.close()
        sys.stdout.write(json.dumps(checks, sort_keys=True, separators=(",", ":")))
        return all(checks.values())
    except (KeyError, OSError, ValueError):
        return False


def _single_retry_token() -> bool:
    """Prove one status retry mutates the shared run-level token."""

    async def run() -> bool:
        token = open(os.environ["JARVIS_EGRESS_PROXY_CREDENTIALS_FILE"], encoding="utf-8").read().strip()
        budget = RetryBudgetV1()
        fetcher = DirectHTTPFetcher.with_egress_proxy(
            "http://egress-guard:8080",
            proxy_headers={"Proxy-Authorization": "Bearer " + token},
            retry_budget=budget,
        )
        try:
            result = await fetcher.fetch("http://allowed.fixture.test/transient-503")
            return budget.used and fetcher.network_retry_count == 1 and bool(result.body)
        finally:
            await fetcher.aclose()

    try:
        import asyncio

        return asyncio.run(run())
    except (OSError, RuntimeError, ValueError):
        return False


def searxng_json_usable() -> bool:
    """Exercise the real private SearXNG API without emitting query or results."""

    body = urllib.parse.urlencode(
        {"q": "!qf JARVIS qualification", "format": "json", "language": "en", "safesearch": "1"}
    ).encode("ascii")
    request = urllib.request.Request(
        "http://searxng:8080/search",
        data=body,
        headers={"Accept": "application/json", "X-Forwarded-For": "127.0.0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            raw = response.read(256 * 1024 + 1)
            value = json.loads(raw)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        response.status == 200
        and response.headers.get_content_type() == "application/json"
        and len(raw) <= 256 * 1024
        and isinstance(value, dict)
        and isinstance(value.get("results"), list)
        and len(value["results"]) > 0
    )


def _research_case() -> bool:
    """Run one manifest case without putting query, URL, token, or response on argv/logs."""

    try:
        spec = json.loads(sys.stdin.buffer.read(64 * 1024))
        if set(spec) != {"request", "idempotency_key", "expected", "cancel_after_ms"}:
            return False
        token = open(os.environ["JARVIS_RESEARCH_BEARER_TOKEN_FILE"], encoding="utf-8").read().strip()
        raw = json.dumps(spec["request"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        request = urllib.request.Request(
            "http://127.0.0.1:8770/v1/research",
            data=raw,
            method="POST",
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
                "Idempotency-Key": spec["idempotency_key"],
            },
        )
        result: dict[str, object] = {}

        def perform() -> None:
            try:
                with urllib.request.urlopen(request, timeout=50) as response:
                    body = response.read(1024 * 1024)
                    result.update(status=response.status, body=body)
            except urllib.error.HTTPError as exc:
                result.update(status=exc.code, body=exc.read(64 * 1024))
            except OSError:
                result.update(status=0, body=b"")

        worker = threading.Thread(target=perform, daemon=True)
        worker.start()
        cancel_ms = int(spec["cancel_after_ms"])
        if cancel_ms:
            worker.join(cancel_ms / 1000)
            passed = worker.is_alive()
            summary = {"cancelled": passed, "effect_sha256": ""}
        else:
            worker.join(52)
            expected = spec["expected"]
            status = int(result.get("status", 0))
            body = result.get("body", b"")
            if not isinstance(body, bytes) or not isinstance(expected, dict):
                return False
            source_count = -1
            languages: list[str] = []
            provenances: list[str] = []
            groups = -1
            retries = -1
            if status == 200:
                value = json.loads(body)
                corpus = value["corpus"]
                sources = corpus["sources"]
                source_count = len(sources)
                languages = sorted(source["language_code"] for source in sources)
                provenances = sorted(source["date_provenance"] for source in sources)
                groups = len({source["independence_group_id"] for source in sources})
                retries = corpus["network_retry_count"]
            passed = (
                status == expected["http"]
                and source_count == expected["source_count"]
                and languages == expected["language_codes"]
                and provenances == expected["date_provenance"]
                and groups == expected["independence_group_count"]
                and (expected["retry_count"] is None or retries == expected["retry_count"])
            )
            summary = {
                "http": status,
                "source_count": source_count,
                "language_count": len(languages),
                "independence_group_count": groups,
                "retry_count": retries,
                "effect_sha256": hashlib.sha256(body).hexdigest(),
            }
        sys.stdout.write(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return passed
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _qualification_identity() -> bool:
    try:
        token = open(os.environ["JARVIS_RESEARCH_BEARER_TOKEN_FILE"], encoding="utf-8").read().strip()
        request = urllib.request.Request(
            "http://127.0.0.1:8770/v1/qualification",
            headers={"Authorization": "Bearer " + token},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            value = json.loads(response.read(64 * 1024))
        sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
        return response.status == 200 and isinstance(value, dict) and len(value) == 8
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _api_auth_boundary() -> bool:
    try:
        token = open(os.environ["JARVIS_RESEARCH_BEARER_TOKEN_FILE"], encoding="utf-8").read().strip()
        statuses: list[int] = []
        for authorization in ("Bearer invalid", "Bearer " + token):
            request = urllib.request.Request(
                "http://127.0.0.1:8770/v1/qualification",
                headers={"Authorization": authorization},
            )
            try:
                with urllib.request.urlopen(request, timeout=3) as response:
                    statuses.append(response.status)
            except urllib.error.HTTPError as exc:
                statuses.append(exc.code)
        return statuses == [401, 200]
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args == ["direct-route-blocked"]:
        return 0 if direct_route_blocked() else 1
    if args == ["searxng-json-usable"]:
        return 0 if searxng_json_usable() else 1
    if args == ["research-case"]:
        return 0 if _research_case() else 1
    if args == ["qualification-identity"]:
        return 0 if _qualification_identity() else 1
    if args == ["api-auth-boundary"]:
        return 0 if _api_auth_boundary() else 1
    if args == ["egress-matrix"]:
        return 0 if _egress_matrix() else 1
    if args == ["single-retry-token"]:
        return 0 if _single_retry_token() else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
