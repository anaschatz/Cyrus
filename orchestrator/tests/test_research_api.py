from __future__ import annotations

import json
import socket
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_orchestrator.domain import canonical_json
from jarvis_orchestrator.research import ResearchCorpusV1, ResearchRequestV1, SearchFacetV1, SearchPlanV1
from jarvis_research.api import APIError, ResearchApplicationV1, ResearchHTTPServer, parse_research_request, validate_post_envelope
from jarvis_research.config import ASSET_ROOT, QualificationIdentityV1, RuntimeConfigV1, file_sha256


def plan() -> SearchPlanV1:
    return SearchPlanV1(
        "Current verified research question", (SearchFacetV1("f1", "verified result"),),
        ("current verified result",), "en", "current", "Europe/Athens",
    )


def request_bytes() -> bytes:
    return canonical_json(asdict(ResearchRequestV1("task-1", "conversation-1", 1, plan()))).encode()


class FakeEngine:
    def __init__(self) -> None:
        self.calls = 0

    async def research(self, request):
        self.calls += 1
        return ResearchCorpusV1(request.plan.plan_hash, (), {}, (), ())


class ResearchAPITests(unittest.TestCase):
    @unittest.skipUnless(socket.has_ipv6, "IPv6 is unavailable")
    def test_api_can_bind_the_supported_ipv6_loopback_address(self) -> None:
        app = ResearchApplicationV1(self.config(), engine=FakeEngine(), audit=lambda _: None)
        with ResearchHTTPServer(("::1", 0), app) as server:
            self.assertEqual(server.socket.family, socket.AF_INET6)

    def config(self) -> RuntimeConfigV1:
        identity = QualificationIdentityV1(
            "research-corpus/1", "a" * 64, "b" * 64, "c" * 64,
            "jarvis-language-script/1", file_sha256(ASSET_ROOT / "language-detector-v1.json"),
            "authority-registry/1", "",
        )
        return RuntimeConfigV1("127.0.0.1", 8770, "t" * 32, "proxy-token", identity)

    def test_strict_request_rejects_unknown_duplicate_nfc_and_browser(self) -> None:
        parsed = parse_research_request(request_bytes())
        self.assertEqual(parsed.plan.plan_hash, plan().plan_hash)
        value = json.loads(request_bytes())
        value["unknown"] = True
        with self.assertRaises(APIError):
            parse_research_request(canonical_json(value).encode())
        duplicate = request_bytes()[:-1] + b',"epoch":1}'
        with self.assertRaises(APIError):
            parse_research_request(duplicate)
        value = json.loads(request_bytes())
        value["plan"]["canonical_query"] = "Cafe\u0301"
        with self.assertRaises(APIError):
            parse_research_request(canonical_json(value).encode())
        value = json.loads(request_bytes())
        value["plan"]["browser_fallback_allowed"] = True
        with self.assertRaises(APIError):
            parse_research_request(canonical_json(value).encode())

    def test_auth_qualification_idempotency_and_sanitized_audit(self) -> None:
        audit = []
        engine = FakeEngine()
        app = ResearchApplicationV1(self.config(), engine=engine, audit=audit.append)
        self.assertFalse(app.authenticated(None))
        self.assertFalse(app.authenticated("Bearer wrong"))
        self.assertFalse(app.authenticated("Bearer café"))
        self.assertTrue(app.authenticated("Bearer " + "t" * 32))
        self.assertEqual(set(json.loads(app.qualification())), {
            "contract_version", "model_sha256", "profile_sha256", "qualification_sha256",
            "language_detector_version", "language_detector_sha256", "authority_registry_version",
            "browser_qualification_sha256",
        })
        first = app.execute(request_bytes(), "same-key")
        second = app.execute(request_bytes(), "same-key")
        self.assertEqual(first, second)
        self.assertEqual(engine.calls, 1)
        changed = json.loads(request_bytes())
        changed["task_id"] = "task-2"
        with self.assertRaisesRegex(APIError, "idempotency_conflict"):
            app.execute(canonical_json(changed).encode(), "same-key")
        log = canonical_json(audit)
        self.assertNotIn("Current verified", log)
        self.assertNotIn("current verified result", log)
        self.assertIn("request_sha256", log)

    def test_deeply_nested_json_is_a_client_error(self) -> None:
        with self.assertRaisesRegex(APIError, "request_json_invalid") as caught:
            parse_research_request(b'[' * 2000 + b'0' + b']' * 2000)
        self.assertEqual(caught.exception.status, 400)

    def test_concurrent_identical_requests_share_one_engine_result(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        duplicate = threading.Event()

        class BlockingEngine(FakeEngine):
            async def research(self, request):
                self.calls += 1
                if self.calls > 1:
                    duplicate.set()
                entered.set()
                if not release.wait(2):
                    raise AssertionError("fixture release timed out")
                return ResearchCorpusV1(request.plan.plan_hash, (), {}, ())

        engine = BlockingEngine()
        app = ResearchApplicationV1(self.config(), engine=engine, audit=lambda _: None)
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(app.execute, request_bytes(), "same-key")
            self.assertTrue(entered.wait(1))
            second = pool.submit(app.execute, request_bytes(), "same-key")
            try:
                self.assertFalse(duplicate.wait(0.1))
                changed = json.loads(request_bytes())
                changed["task_id"] = "task-2"
                with self.assertRaisesRegex(APIError, "idempotency_conflict"):
                    app.execute(canonical_json(changed).encode(), "same-key")
            finally:
                release.set()
            self.assertEqual(first.result(2), second.result(2))
        self.assertEqual(engine.calls, 1)

    def test_fractional_page_limit_is_rejected_before_retrieval(self) -> None:
        value = json.loads(request_bytes())
        value["plan"]["page_limit"] = 1.5
        with self.assertRaises(APIError) as caught:
            parse_research_request(canonical_json(value).encode())
        self.assertEqual(caught.exception.status, 400)

    def test_content_type_body_and_idempotency_bounds(self) -> None:
        self.assertEqual(
            validate_post_envelope(
                content_type="application/json", transfer_encoding=None,
                content_length=str(len(request_bytes())), idempotency_key="bounded-key",
            ),
            (len(request_bytes()), "bounded-key"),
        )
        invalid = (
            {"content_type": "application/json; charset=utf-8", "transfer_encoding": None, "content_length": "1", "idempotency_key": "k"},
            {"content_type": "application/json", "transfer_encoding": "chunked", "content_length": "1", "idempotency_key": "k"},
            {"content_type": "application/json", "transfer_encoding": None, "content_length": str(32 * 1024 + 1), "idempotency_key": "k"},
            {"content_type": "application/json", "transfer_encoding": None, "content_length": "1", "idempotency_key": "x" * 129},
        )
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(APIError):
                validate_post_envelope(**item)

    def test_qualification_file_is_exact_and_digest_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "identity.json"
            value = asdict(self.config().qualification)
            path.write_text(canonical_json(value), encoding="utf-8")
            self.assertEqual(QualificationIdentityV1.load(path), self.config().qualification)
            value["extra"] = "forbidden"
            path.write_text(canonical_json(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fields"):
                QualificationIdentityV1.load(path)


if __name__ == "__main__":
    unittest.main()
