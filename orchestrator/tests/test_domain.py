from __future__ import annotations

import unittest
import uuid
from dataclasses import replace

from jarvis_orchestrator.domain import OrchestratorError, RouteDecision, require_uuid4


class DomainBoundaryTests(unittest.TestCase):
    def test_uuid_requires_text_and_preserves_existing_case_normalization(self):
        value = str(uuid.uuid4())
        self.assertEqual(require_uuid4(value.upper()), value)
        for invalid in (None, value.encode(), True, 1, [], {}, "", str(uuid.uuid1())):
            with self.subTest(value=invalid), self.assertRaises(OrchestratorError) as caught:
                require_uuid4(invalid)
            self.assertEqual(caught.exception.status, 400)

    def test_route_validation_rejects_nontext_tokens_and_invalid_unicode(self):
        valid = RouteDecision("local", "", "", "none", "default")
        self.assertEqual(valid.validate(), valid)
        for field in ("route", "response_language", "tool_name"):
            for invalid in ([], {}, None, 1):
                with self.subTest(field=field, value=invalid), self.assertRaisesRegex(
                    OrchestratorError, "route_validation_failed"
                ):
                    replace(valid, **{field: invalid}).validate()
        for route, field in (("memory_local", "memory_query"), ("research_local", "research_query")):
            with self.subTest(route=route), self.assertRaisesRegex(OrchestratorError, "route_validation_failed"):
                replace(valid, route=route, **{field: "invalid-\ud800"}).validate()


if __name__ == "__main__":
    unittest.main()
