from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from jarvis_orchestrator.research import DateProvenance
from jarvis_orchestrator.research_security import (
    BrowserLimitsV1,
    RetryBudgetV1,
    RobotsDecision,
    browser_fallback_eligible,
    language_eligibility,
    publication_date_from_html,
    reject_sensitive_text,
    robots_decision,
    validate_resolved_addresses,
)


class ResearchSecurityTests(unittest.TestCase):
    def test_mixed_ipv4_or_ipv6_dns_answer_fails_closed(self) -> None:
        self.assertEqual(validate_resolved_addresses(["93.184.216.34"]), ("93.184.216.34",))
        forbidden = (
            ["93.184.216.34", "127.0.0.1"],
            ["10.0.0.1"],
            ["169.254.1.1"],
            ["::1"],
            ["fe80::1"],
            ["fd00::1"],
            ["::"],
        )
        for answer in forbidden:
            with self.subTest(answer=answer), self.assertRaisesRegex(ValueError, "forbidden"):
                validate_resolved_addresses(answer)

    def test_robots_policy_is_explicit_and_fail_closed(self) -> None:
        body = b"User-agent: *\nDisallow: /private\nAllow: /private/public\n"
        self.assertEqual(robots_decision(status=200, body=body, path="/"), RobotsDecision.ALLOW)
        self.assertEqual(robots_decision(status=200, body=body, path="/private/x"), RobotsDecision.DISALLOW)
        self.assertEqual(robots_decision(status=200, body=body, path="/private/public/x"), RobotsDecision.ALLOW)
        self.assertEqual(robots_decision(status=404, body=b"", path="/"), RobotsDecision.ALLOW)
        for status in (None, 401, 403, 429, 500):
            self.assertEqual(robots_decision(status=status, body=b"", path="/"), RobotsDecision.FAIL_CLOSED)
        self.assertEqual(robots_decision(status=200, body=b"malformed", path="/"), RobotsDecision.FAIL_CLOSED)
        self.assertEqual(
            robots_decision(status=200, body=b"x" * (128 * 1024 + 1), path="/"),
            RobotsDecision.FAIL_CLOSED,
        )

    def test_one_retry_token_is_global_and_only_pre_body_transient(self) -> None:
        budget = RetryBudgetV1()
        self.assertFalse(budget.consume(status=503, body_bytes_received=1, remaining_seconds=5))
        self.assertTrue(budget.consume(status=503, delay_seconds=0.5, remaining_seconds=5))
        self.assertFalse(budget.consume(error_code="connect_reset", remaining_seconds=5))
        self.assertFalse(RetryBudgetV1().consume(status=404, remaining_seconds=5))
        self.assertFalse(RetryBudgetV1().consume(status=429, delay_seconds=0.6, remaining_seconds=5))

    def test_browser_and_language_bounds_are_frozen(self) -> None:
        limits = BrowserLimitsV1()
        self.assertEqual((limits.page_max_requests, limits.run_max_bytes, limits.tunnel_max_bytes), (40, 12 * 1024 * 1024, 4 * 1024 * 1024))
        for blocked in ("image", "media", "font", "websocket", "service_worker", "download"):
            self.assertFalse(limits.resource_allowed(blocked))
        self.assertTrue(limits.resource_allowed("document"))
        self.assertTrue(browser_fallback_eligible(normalized_characters=399, non_boilerplate_blocks=2))
        self.assertFalse(browser_fallback_eligible(normalized_characters=400, non_boilerplate_blocks=2))
        greek = "Α" * 200
        self.assertEqual(language_eligibility(sample=greek, detected_code="el", confidence=0.80), (True, "el-GR"))
        self.assertEqual(language_eligibility(sample="x" * 199, detected_code="en", confidence=1), (False, None))
        self.assertEqual(language_eligibility(sample="x" * 200, detected_code="fr", confidence=1), (False, None))

    def test_date_metadata_never_uses_visible_text_or_conflicting_values(self) -> None:
        value = '<html><meta property="article:published_time" content="2026-08-10T10:00:00+03:00"><body>Published yesterday</body></html>'
        self.assertEqual(
            publication_date_from_html(value),
            ("2026-08-10T07:00:00.000Z", DateProvenance.ARTICLE_PUBLISHED_TIME),
        )
        conflict = value.replace("</html>", '<script type="application/ld+json">{"datePublished":"2026-08-09T10:00:00+03:00"}</script></html>')
        self.assertEqual(publication_date_from_html(conflict), (None, DateProvenance.CONFLICT))
        self.assertEqual(publication_date_from_html("<p>Published 2026-08-10</p>"), (None, DateProvenance.NONE))

    def test_sensitive_patterns_are_rejected_before_search(self) -> None:
        with self.assertRaisesRegex(ValueError, "sensitive"):
            reject_sensitive_text("token eyJabcdefghij.abcdefghij.abcdefghij")


if __name__ == "__main__":
    unittest.main()
