from __future__ import annotations

import json
import io
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jarvis_inference.contracts import Capabilities, normalize_request
from jarvis_inference.engine import EngineFailure, EngineProtocolFailure, GenerationEvent, PreparedPrompt
from jarvis_inference.errors import ApplianceError
from jarvis_inference.gateway import GatewayHandler, Production, effective_configuration
from jarvis_inference.strict_json import canonical_bytes


class EffectiveConfigurationTests(unittest.TestCase):
    def test_slow_body_cannot_extend_its_total_deadline(self) -> None:
        now = [0.0]

        class Drip(io.BytesIO):
            def read(self, size=-1):
                result = super().read(size)
                now[0] += 4 * len(result)
                return result

            def read1(self, size=-1):
                return self.read(min(size, 1))

        handler = Mock(rfile=Drip(b"{}    "))
        handler.headers.get_all.return_value = ["6"]
        with patch("jarvis_inference.gateway.time.monotonic", side_effect=lambda: now[0]):
            with self.assertRaises(ApplianceError) as caught:
                GatewayHandler._read_body(handler, 0.0)
        self.assertEqual(caught.exception.status, 408)
        self.assertTrue(handler.close_connection)

    def test_unproven_engine_teardown_stops_admission(self) -> None:
        engine = Mock()
        engine.cancel.side_effect = EngineFailure("cancel failed")
        engine.close.side_effect = EngineFailure("exit unproven")
        production = Production(None, "test", None)
        production.engine = engine
        production.ready_state = True
        cancel = threading.Event()
        GatewayHandler._cancel_generation(Mock(production=production), engine, cancel, 100)
        self.assertTrue(cancel.is_set())
        self.assertTrue(production.stop.is_set())
        self.assertFalse(production.is_ready())

    def test_malformed_backend_event_fields_are_protocol_errors(self) -> None:
        for event in (
            GenerationEvent([]),
            GenerationEvent("final", finish_reason={}, completion_tokens=0),
            GenerationEvent("tool", tool_name=["tool"], tool_arguments="{}"),
            GenerationEvent("tool", tool_name="tool", tool_arguments={}),
        ):
            with self.subTest(event=event), self.assertRaises(EngineProtocolFailure):
                GatewayHandler._validate_event(event)

    def test_qualification_identity_survives_canonical_json_round_trip(self) -> None:
        configuration = effective_configuration()
        self.assertEqual(json.loads(canonical_bytes(configuration)), configuration)

    def test_qualification_cancel_rejects_nonstring_phase(self) -> None:
        for phase in ([], {}, None, True):
            handler = Mock(
                path="/jarvis/qualification/cancel",
                production=SimpleNamespace(inject_qualification_cancel=Mock()),
            )
            handler._read_body.return_value = {"phase": phase}
            with self.subTest(phase=phase):
                GatewayHandler.do_POST(handler)
                self.assertEqual(handler._error.call_args.args[0].status, 400)
                handler.production.inject_qualification_cancel.assert_not_called()


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
