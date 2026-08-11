from __future__ import annotations

import ipaddress
import socket
import threading
import unittest

import _bootstrap  # noqa: F401

from jarvis_research.egress_guard import EgressGuardServer, GuardConfigV1, GuardError, GuardPolicyV1, parse_headers
from jarvis_research.qualification_fixtures import DNSFixtureServer, FIXTURE_IPV4, _decode_name


class EgressGuardTests(unittest.TestCase):
    def config(self, networks=()) -> GuardConfigV1:
        return GuardConfigV1("127.0.0.1", 8080, "g" * 32, tuple(ipaddress.ip_network(item) for item in networks))

    def test_proxy_auth_and_strict_request_parser(self) -> None:
        policy = GuardPolicyV1(self.config(), resolver=lambda *_: ((socket.AF_INET, "93.184.216.34"),))
        self.assertTrue(policy.authenticated("Bearer " + "g" * 32))
        self.assertFalse(policy.authenticated("Bearer wrong"))
        method, target, version, headers = parse_headers(
            b"GET http://example.com/path HTTP/1.1\r\nHost: example.com\r\nProxy-Authorization: Bearer token\r\n\r\n"
        )
        self.assertEqual((method, target, version), ("GET", "http://example.com/path", "HTTP/1.1"))
        self.assertIn("proxy-authorization", headers)
        for invalid in (b"PUT http://example.com/ HTTP/1.1\r\nHost: x\r\n\r\n", b"GET /relative HTTP/2\r\nHost: x\r\n\r\n"):
            with self.assertRaises(GuardError):
                parse_headers(invalid)

    def test_resolution_rejects_private_ipv4_ipv6_mixed_and_internal(self) -> None:
        forbidden = (
            ((socket.AF_INET, "10.0.0.1"),),
            ((socket.AF_INET6, "fd00::1"),),
            ((socket.AF_INET, "93.184.216.34"), (socket.AF_INET, "127.0.0.1")),
        )
        for records in forbidden:
            with self.subTest(records=records):
                policy = GuardPolicyV1(self.config(), resolver=lambda *_, records=records: records)
                with self.assertRaisesRegex(GuardError, "dns_answer_forbidden"):
                    policy.resolve("example.com", 443)
        policy = GuardPolicyV1(
            self.config(("93.184.216.0/24",)), resolver=lambda *_: ((socket.AF_INET, "93.184.216.34"),)
        )
        with self.assertRaisesRegex(GuardError, "dns_container_internal"):
            policy.resolve("example.com", 443)

    def test_resolution_is_repeated_and_pinned_for_rebinding(self) -> None:
        calls = 0

        def resolver(*_):
            nonlocal calls
            calls += 1
            return ((socket.AF_INET, "93.184.216.34" if calls == 1 else "127.0.0.1"),)

        policy = GuardPolicyV1(self.config(), resolver=resolver)
        first = policy.resolve("rebind.example.com", 443)
        self.assertEqual(first.address, "93.184.216.34")
        with self.assertRaisesRegex(GuardError, "dns_answer_forbidden"):
            policy.resolve("rebind.example.com", 443)
        self.assertEqual(calls, 2)

    def test_connect_and_url_targets_allow_only_public_80_443(self) -> None:
        policy = GuardPolicyV1(self.config(), resolver=lambda *_: ((socket.AF_INET, "93.184.216.34"),))
        self.assertEqual(policy.target_for_connect("example.com:443").port, 443)
        self.assertEqual(policy.target_for_url("http://example.com/a")[1].port, 80)
        for value in ("example.com:22", "user@example.com:443", "127.0.0.1:443"):
            with self.assertRaises(GuardError):
                policy.target_for_connect(value)

    def test_controlled_dns_fixture_answers_are_deterministic(self) -> None:
        server = object.__new__(DNSFixtureServer)
        server._counts = {}
        server._lock = threading.Lock()
        self.assertEqual(server._addresses("allowed.fixture.test", 1), (FIXTURE_IPV4,))
        self.assertEqual(server._addresses("mixed.fixture.test", 1), (FIXTURE_IPV4, "127.0.0.1"))
        self.assertEqual(server._addresses("rebind.fixture.test", 1), (FIXTURE_IPV4,))
        self.assertEqual(server._addresses("rebind.fixture.test", 1), ("127.0.0.1",))
        encoded = b"\x07example\x03com\x00"
        self.assertEqual(_decode_name(encoded, 0), ("example.com", len(encoded)))

    def test_live_http_proxy_and_connect_tunnel_inject_no_target_data_into_errors(self) -> None:
        upstream_requests = []

        def connector(_target):
            proxy_end, fixture_end = socket.socketpair()

            def fixture():
                data = fixture_end.recv(4096)
                upstream_requests.append(data)
                if data.startswith(b"GET "):
                    fixture_end.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                else:
                    fixture_end.sendall(data)
                fixture_end.close()

            threading.Thread(target=fixture, daemon=True).start()
            return proxy_end

        audit = []
        policy = GuardPolicyV1(
            self.config(), resolver=lambda *_: ((socket.AF_INET, "93.184.216.34"),),
            connector=connector, audit=audit.append,
        )
        server = EgressGuardServer(("127.0.0.1", 0), policy)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        port = server.server_address[1]
        try:
            with socket.create_connection(("127.0.0.1", port)) as client:
                client.sendall(b"GET http://example.com/secret?q=hidden HTTP/1.1\r\nHost: example.com\r\nProxy-Authorization: Bearer " + b"g" * 32 + b"\r\n\r\n")
                response = client.recv(4096)
            self.assertIn(b"200 OK", response)
            self.assertTrue(upstream_requests[0].startswith(b"GET /secret?q=hidden HTTP/1.1"))
            self.assertNotIn(b"Proxy-Authorization", upstream_requests[0])
            with socket.create_connection(("127.0.0.1", port)) as client:
                client.sendall(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\nProxy-Authorization: Bearer " + b"g" * 32 + b"\r\n\r\n")
                self.assertIn(b"200 Connection Established", client.recv(4096))
                client.sendall(b"ping")
                self.assertEqual(client.recv(4), b"ping")
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=1)
        self.assertNotIn("example.com", str(audit))
        self.assertNotIn("secret", str(audit))


if __name__ == "__main__":
    unittest.main()
