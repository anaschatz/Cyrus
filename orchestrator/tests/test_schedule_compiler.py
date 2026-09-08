from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import httpx

from jarvis_orchestrator.jcs import jcs_bytes
from jarvis_orchestrator.schedule_compiler import (
    INPUT_JSON_SCHEMA, OUTPUT_JSON_SCHEMA, PROMPT_TEMPLATE_SHA256,
    CompilerQualification, LocalScheduleCompiler, load_compiler_qualification,
)
from jarvis_orchestrator.scheduler_contract import (
    PackagedTimezoneProvider, ScheduleContractError, compiler_qualification_digest,
)
from test_scheduler_v5 import proposal_input, proposal_output


def qualification_document():
    inputs = dict.fromkeys(("compiler_artifact_sha256", "model_artifact_sha256", "model_profile_sha256",
                            "qualification_report_sha256", "runtime_sha256", "tokenizer_sha256"), "a" * 64)
    inputs.update(prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
                  input_schema_sha256=hashlib.sha256(jcs_bytes(INPUT_JSON_SCHEMA)).hexdigest(),
                  output_schema_sha256=hashlib.sha256(jcs_bytes(OUTPUT_JSON_SCHEMA)).hexdigest())
    return {"qualification_inputs": inputs, "qualification_sha256": compiler_qualification_digest(inputs)}


class ScheduleCompilerBoundaryTests(unittest.TestCase):
    def test_compiler_imports_with_only_the_standalone_orchestrator_package(self):
        source = Path(__file__).resolve().parents[1] / "src"
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c",
             "import sys; sys.path.insert(0, sys.argv[1]); "
             "import jarvis_orchestrator.schedule_compiler; "
             "assert 'jarvis_voice' not in sys.modules", str(source)],
            capture_output=True, text=True, timeout=10, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_qualification_checks_declared_runtime_inputs_without_rewriting_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"

            def load(value):
                raw = json.dumps(value).encode()
                path.write_bytes(raw)
                return load_compiler_qualification(path, hashlib.sha256(raw).hexdigest())

            value = qualification_document()
            loaded = load(value)
            self.assertEqual(loaded.artifact, value)
            for field in ("prompt_template_sha256", "input_schema_sha256", "output_schema_sha256"):
                invalid = qualification_document()
                invalid["qualification_inputs"][field] = "b" * 64
                with self.subTest(field=field), self.assertRaises(ScheduleContractError):
                    load(invalid)

    def test_qualification_rejects_ambiguous_or_malformed_input_with_typed_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            valid = json.dumps(qualification_document())
            for raw in (valid[:-1] + ', "qualification_sha256":' + json.dumps(qualification_document()["qualification_sha256"]) + '}',
                        " " * (1024 * 1024) + valid, '[' * 1500 + '0' + ']' * 1500,
                        json.dumps({"qualification_inputs": None, "qualification_sha256": "a" * 64})):
                path.write_text(raw)
                with self.subTest(size=len(raw)), self.assertRaises(ScheduleContractError):
                    load_compiler_qualification(path, hashlib.sha256(raw.encode()).hexdigest())
            for invalid in (None, 1, [], ""):
                with self.subTest(pin=invalid), self.assertRaises(ScheduleContractError):
                    load_compiler_qualification(path, invalid)

    def compile_response(self, response_factory, *, timeout_seconds=1):
        request_id = str(uuid.uuid4())
        incoming = proposal_input(request_id, "2030-01-10T12:00:00.000Z")
        outgoing = proposal_output(request_id)
        compiler = LocalScheduleCompiler(
            base_url="http://127.0.0.1:8765/v1", bearer_token="test-token", model_alias="jarvis-local",
            qualification=CompilerQualification("a" * 64, {}), timeout_seconds=timeout_seconds,
            zones=PackagedTimezoneProvider(),
        )
        client_type = httpx.AsyncClient
        with patch("httpx.AsyncClient", side_effect=lambda **kwargs: client_type(
            transport=httpx.MockTransport(lambda request: response_factory(request, outgoing)), **kwargs
        )):
            return compiler.compile(incoming)

    @staticmethod
    def response_value(outgoing):
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(outgoing)},
                             "finish_reason": "stop"}]}

    def test_complete_response_compiles_but_truncated_or_ambiguous_outputs_do_not(self):
        result = self.compile_response(lambda request, outgoing: httpx.Response(200, json=self.response_value(outgoing)))
        self.assertEqual(result.first_due_at_utc, "2030-01-10T12:20:00.000Z")
        for defect in ("length", "duplicate_inner", "duplicate_outer", "multiple_choices", "nontext_content", "deep_json"):
            def response(request, outgoing):
                value = self.response_value(outgoing)
                if defect == "length":
                    value["choices"][0]["finish_reason"] = "length"
                elif defect == "duplicate_inner":
                    content = json.dumps(outgoing)
                    value["choices"][0]["message"]["content"] = content[:-1] + ', "operation":"CREATE_JOB"}'
                elif defect == "multiple_choices":
                    value["choices"] *= 2
                elif defect == "nontext_content":
                    value["choices"][0]["message"]["content"] = outgoing
                elif defect == "deep_json":
                    value["choices"][0]["message"]["content"] = '[' * 1500 + '0' + ']' * 1500
                raw = json.dumps(value)
                if defect == "duplicate_outer":
                    raw = raw[:-1] + ', "choices":' + json.dumps(value["choices"]) + '}'
                return httpx.Response(200, content=raw, headers={"Content-Type": "application/json"})
            with self.subTest(defect=defect), self.assertRaises(ScheduleContractError):
                self.compile_response(response)

    def test_one_deadline_covers_slow_headers_and_trickling_body(self):
        async def slow_headers(request, outgoing):
            await asyncio.sleep(10)
            return httpx.Response(200, json=self.response_value(outgoing))

        class Trickle(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                for _ in range(1000):
                    await asyncio.sleep(0.005)
                    yield b" "

            async def aclose(self):
                self.closed = True

        body = Trickle()
        for factory in (slow_headers, lambda request, outgoing: httpx.Response(
            200, stream=body, headers={"Content-Type": "application/json"}
        )):
            started = time.monotonic()
            with self.assertRaisesRegex(ScheduleContractError, "compiler_timeout"):
                self.compile_response(factory, timeout_seconds=0.03)
            self.assertLess(time.monotonic() - started, 1)
        self.assertTrue(body.closed)

    def test_invalid_configuration_and_transport_errors_have_typed_results(self):
        valid = dict(base_url="http://127.0.0.1:8765/v1", bearer_token="test-token", model_alias="jarvis-local",
                     qualification=CompilerQualification("a" * 64, {}), timeout_seconds=1,
                     zones=PackagedTimezoneProvider())
        for field, value in (("base_url", "http://[invalid/v1"), ("base_url", None),
                             ("base_url", "http://127.0.0.1:0/v1"), ("bearer_token", "bad\r\ntoken"),
                             ("bearer_token", "\ud800"), ("timeout_seconds", True),
                             ("timeout_seconds", "1"), ("timeout_seconds", float("nan"))):
            with self.subTest(field=field), self.assertRaises(ScheduleContractError):
                LocalScheduleCompiler(**{**valid, field: value})

        def unavailable(request, outgoing):
            raise httpx.ConnectError("synthetic failure", request=request)

        for factory in (unavailable, lambda request, outgoing: httpx.Response(503)):
            with self.assertRaisesRegex(ScheduleContractError, "compiler_unavailable"):
                self.compile_response(factory)

    def test_oversized_response_stops_reading_and_closes_stream(self):
        class Body(httpx.AsyncByteStream):
            closed = False
            exhausted = False

            async def __aiter__(self):
                for _ in range(128):
                    yield b" " * 16384
                self.exhausted = True

            async def aclose(self):
                self.closed = True

        body = Body()
        with self.assertRaises(ScheduleContractError):
            self.compile_response(lambda request, outgoing: httpx.Response(
                200, stream=body, headers={"Content-Type": "application/json"},
            ))
        self.assertTrue(body.closed)
        self.assertFalse(body.exhausted)


if __name__ == "__main__":
    unittest.main()
