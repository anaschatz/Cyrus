from __future__ import annotations

import unittest

from jarvis_inference.contracts import Capabilities, normalize_request
from jarvis_inference.engine import GenerationEvent, PreparedPrompt
from jarvis_inference.gateway import GatewayHandler


class UsageTests(unittest.TestCase):
    def test_accounting_uses_prepared_ids_and_backend_count(self) -> None:
        request = normalize_request(
            {"model": "jarvis-local", "messages": [{"role": "user", "content": "x"}], "max_completion_tokens": 2},
            alias="jarvis-local", capabilities=Capabilities(), context_window=16,
        )
        prepared = PreparedPrompt((1, 2, 3), "prompt", request)
        usage = GatewayHandler._usage(prepared, GenerationEvent("final", finish_reason="stop", completion_tokens=2))
        self.assertEqual(usage, {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})

    def test_boolean_and_over_limit_accounting_fail_closed(self) -> None:
        request = normalize_request(
            {"model": "jarvis-local", "messages": [{"role": "user", "content": "x"}], "max_completion_tokens": 2},
            alias="jarvis-local", capabilities=Capabilities(), context_window=16,
        )
        prepared = PreparedPrompt((1,), "prompt", request)
        for count in (True, 3):
            with self.subTest(count=count), self.assertRaises(Exception):
                GatewayHandler._usage(prepared, GenerationEvent("final", finish_reason="stop", completion_tokens=count))
