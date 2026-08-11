from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jarvis_inference.macos_fixture import (
    FixtureFailure,
    QualificationClient,
    _memory_drift,
    _phys_footprint_bytes,
    _read_identity,
    _stage,
    _write_evidence,
    main,
)


class MacosFixtureTests(unittest.TestCase):
    def test_loopback_url_is_exact(self) -> None:
        client = QualificationClient("http://127.0.0.1:1234/v1", "secret", "jarvis-local")
        self.assertEqual(client.port, 1234)
        for value in (
            "https://127.0.0.1:1234/v1",
            "http://localhost:1234/v1",
            "http://127.0.0.1:1234/other",
        ):
            with self.subTest(value=value), self.assertRaises(FixtureFailure):
                QualificationClient(value, "secret", "jarvis-local")

    def test_nonstream_empty_text_is_a_valid_stability_sample(self) -> None:
        client = QualificationClient("http://127.0.0.1:1234/v1", "secret", "jarvis-local")
        response = {
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
        }
        with patch.object(client, "json_request", return_value=(200, response)):
            self.assertEqual(client.completion("sample"), "")

    def test_tool_and_structured_helpers_validate_public_shapes(self) -> None:
        client = QualificationClient("http://127.0.0.1:1234/v1", "secret", "jarvis-local")
        tool = {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_" + "a" * 32,
                        "type": "function",
                        "function": {"name": "record_index", "arguments": '{"index":7}'},
                    }],
                },
            }],
        }
        structured = {
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": '{"accepted":true}'},
            }],
        }
        with patch.object(client, "json_request", side_effect=[(200, tool), (200, structured)]):
            client.tool_call("sample", index=7, named=True)
            self.assertEqual(client.structured("sample", {
                "type": "object",
                "properties": {"accepted": {"type": "boolean"}},
                "required": ["accepted"],
                "additionalProperties": False,
            }), {"accepted": True})

    def test_failure_diagnostic_contains_only_stage_and_kind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            diagnostic = Path(temporary) / "diagnostic.json"
            environment = {"JARVIS_QUAL_DIAGNOSTIC": str(diagnostic)}
            _stage("json_schema")
            with patch.dict(os.environ, environment, clear=False), patch(
                "jarvis_inference.macos_fixture.run", side_effect=FixtureFailure("content-free")
            ):
                self.assertEqual(main(), 2)
            self.assertEqual(json.loads(diagnostic.read_text()), {
                "kind": "fixture_failure",
                "stage": "json_schema",
            })
            _stage("bootstrap")

    def test_memory_drift_is_bounded_and_deterministic(self) -> None:
        self.assertEqual(_memory_drift([1_000_000] * 200), (0.0, 0.0))
        final, projected = _memory_drift([1_000_000 + index * 1_000 for index in range(200)])
        self.assertGreater(final, 0.05)
        self.assertGreater(projected, 0.05)
        with self.assertRaises(FixtureFailure):
            _memory_drift([1] * 199)

    def test_phys_footprint_parser_is_strict(self) -> None:
        result = type("Result", (), {
            "returncode": 0,
            "stdout": "Auxiliary data:\n    phys_footprint: 123456 B\n    phys_footprint_peak: 999999 B\n",
        })()
        with patch("jarvis_inference.macos_fixture.subprocess.run", return_value=result) as invoked:
            self.assertEqual(_phys_footprint_bytes({"engine": {"pid": 42}}), 123456)
        self.assertEqual(invoked.call_args.args[0], [
            "/usr/bin/footprint", "--noCategories", "--format", "bytes", "--pid", "42",
        ])
        malformed = type("Result", (), {"returncode": 0, "stdout": "rss: 123456 B\n"})()
        with patch("jarvis_inference.macos_fixture.subprocess.run", return_value=malformed):
            with self.assertRaises(FixtureFailure):
                _phys_footprint_bytes({"engine": {"pid": 42}})

    def test_identity_comes_only_from_current_in_progress_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "alias": "jarvis-local",
                "capabilities": {
                    "text": True,
                    "stream": True,
                    "tools": True,
                    "json_object": True,
                    "json_schema": True,
                },
                "qualification_report": "qualification-report.json",
            }))
            report = root / "qualification-report.json"
            report.write_text(json.dumps({
                "status": "in_progress",
                "challenge": "c" * 64,
                "identity": {"manifest_digest": "d" * 64},
            }))
            alias, identity, capabilities = _read_identity(manifest, "c" * 64)
            self.assertEqual(alias, "jarvis-local")
            self.assertEqual(identity["manifest_digest"], "d" * 64)
            self.assertTrue(capabilities["tools"])
            self.assertTrue(capabilities["json_schema"])
            with self.assertRaises(FixtureFailure):
                _read_identity(manifest, "e" * 64)

    def test_evidence_write_is_fresh_and_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence.json"
            _write_evidence(output, {"ready": True})
            self.assertEqual(json.loads(output.read_text()), {"ready": True})
            with self.assertRaises(FixtureFailure):
                _write_evidence(output, {"ready": False})


if __name__ == "__main__":
    unittest.main()
