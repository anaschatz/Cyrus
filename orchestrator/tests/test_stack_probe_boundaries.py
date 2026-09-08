from __future__ import annotations

import asyncio
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import httpx
import _bootstrap  # noqa: F401
import test_private_stack as fixtures
from jarvis_orchestrator import macos_beta_probes as probes, stack_cli
from jarvis_orchestrator.config import OrchestratorConfig


def frame(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def text_frame(text="ready", finish=None, index=0):
    return frame(
        {
            "choices": [
                {"index": index, "delta": {"content": text}, "finish_reason": finish}
            ]
        }
    )


STOP = frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
USAGE = frame(
    {
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
)
DONE = b"data: [DONE]\n\n"
SSE_HEADERS = {"content-type": "text/event-stream; charset=utf-8"}


class StackProbeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PrivateStackConfigTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.config = OrchestratorConfig.load(self.fixture.orchestrator)

    def run_http(self, action, body=b"", *, status=200, headers=None, stream=None):
        requests, responses = [], []
        original_client = httpx.AsyncClient

        def respond(request):
            requests.append(request)
            response = httpx.Response(
                status,
                headers=headers,
                **({"stream": stream} if stream else {"content": body}),
            )
            responses.append(response)
            return response

        with patch.object(
            httpx,
            "AsyncClient",
            side_effect=lambda **kwargs: original_client(
                transport=httpx.MockTransport(respond), **kwargs
            ),
        ) as client:
            result = action()
        self.assertEqual(len(requests), 1)
        self.assertTrue(responses[0].is_closed)
        self.assertFalse(client.call_args.kwargs["trust_env"])
        self.assertFalse(client.call_args.kwargs["follow_redirects"])
        return result, requests[0]

    def test_ready_probes_require_exact_bool_and_strict_bounded_json(self):
        for action in (
            lambda: stack_cli._readyz(self.config),
            lambda: probes.orchestrator_http_ready(self.config),
        ):
            for raw in (
                b'{"ready":1}',
                b'{"ready":false,"ready":true}',
                b"[" * 2000 + b"0" + b"]" * 2000,
                b" " * (16 * 1024) + b'{"ready":true}',
            ):
                with self.subTest(size=len(raw)):
                    result, _ = self.run_http(action, raw)
                    self.assertFalse(result)
            result, request = self.run_http(action, b'{"ready":true}')
            self.assertTrue(result)
            self.assertEqual(str(request.url), "http://127.0.0.1:8767/readyz")

    def test_beta_session_rejects_duplicate_and_deep_json(self):
        for raw in (
            b"[" * 2000 + b"0" + b"]" * 2000,
            b'{"core_ready":false,"core_ready":true}',
        ):
            with self.subTest(size=len(raw)):
                result, request = self.run_http(
                    lambda: stack_cli._beta_session(self.config), raw
                )
                self.assertIsNone(result)
                self.assertEqual(
                    request.headers["authorization"],
                    "Bearer " + self.config.control_bearer_token,
                )

    def test_warmup_requires_complete_ordered_frames_and_accounting(self):
        invalid = (
            text_frame() + STOP + DONE,
            USAGE + text_frame() + STOP + DONE,
            text_frame() + STOP + STOP + USAGE + DONE,
            text_frame() + STOP + USAGE + USAGE + DONE,
            text_frame() + STOP + USAGE + DONE.rstrip(),
            text_frame() + STOP + USAGE + DONE + text_frame(),
            text_frame(index=False) + STOP + USAGE + DONE,
            STOP + text_frame() + USAGE + DONE,
            text_frame()
            + STOP
            + frame(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": True,
                        "total_tokens": 2,
                    },
                }
            )
            + DONE,
            text_frame()
            + STOP
            + frame(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 17,
                        "total_tokens": 18,
                    },
                }
            )
            + DONE,
            text_frame()
            + STOP
            + frame(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 99,
                    },
                }
            )
            + DONE,
            b"data: " + b"[" * 2000 + b"0" + b"]" * 2000 + b"\n\n" + DONE,
            b" " * (256 * 1024 + 1),
        )
        for raw in invalid:
            with self.subTest(size=len(raw)):
                result, _ = self.run_http(
                    lambda: probes.inference_warmup(None, self.config),
                    raw,
                    headers=SSE_HEADERS,
                )
                self.assertFalse(result)
        valid = text_frame() + STOP + USAGE + DONE
        for raw in (valid, valid.replace(b"\n", b"\r\n")):
            result, request = self.run_http(
                lambda: probes.inference_warmup(None, self.config),
                raw,
                headers=SSE_HEADERS,
            )
            self.assertTrue(result)
            self.assertEqual(request.method, "POST")
            self.assertEqual(json.loads(request.content)["max_completion_tokens"], 16)

    def test_warmup_rejects_redirects_and_encoded_responses(self):
        for status, headers in (
            (307, {"location": "https://example.invalid/"}),
            (200, {**SSE_HEADERS, "content-encoding": "gzip"}),
        ):
            with self.subTest(status=status):
                result, _ = self.run_http(
                    lambda: probes.inference_warmup(None, self.config),
                    headers=headers,
                    status=status,
                )
                self.assertFalse(result)

    def test_total_deadlines_cancel_trickling_response_and_close_stream(self):
        class SlowStream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield b" "
                await asyncio.sleep(10)

            async def aclose(self):
                self.closed = True

        original_timeout = asyncio.timeout
        for action, headers, expected in (
            (lambda: probes.inference_warmup(None, self.config), SSE_HEADERS, False),
            (lambda: stack_cli._readyz(self.config), {}, False),
            (lambda: stack_cli._beta_session(self.config), {}, None),
        ):
            stream = SlowStream()
            with patch.object(
                asyncio, "timeout", side_effect=lambda seconds: original_timeout(0.02)
            ):
                result, _ = self.run_http(action, headers=headers, stream=stream)
            self.assertEqual(result, expected)
            self.assertTrue(stream.closed)

    @unittest.skipUnless(os.name == "posix", "POSIX symlink input")
    def test_cli_init_does_not_resolve_away_a_configured_symlink(self):
        linked = self.fixture.root / "linked-runner.py"
        linked.symlink_to(self.fixture.runner)
        output = self.fixture.root / "created-stack.json"
        rendered = io.StringIO()
        with redirect_stdout(rendered):
            result = stack_cli.main(
                [
                    "init",
                    "--output",
                    str(output),
                    "--voice-config",
                    str(self.fixture.voice),
                    "--orchestrator-config",
                    str(self.fixture.orchestrator),
                    "--inference-runner",
                    str(linked),
                    "--inference-manifest",
                    str(self.fixture.manifest),
                    "--inference-qualification-report",
                    str(self.fixture.report),
                ]
            )
        self.assertEqual(result, 2)
        self.assertFalse(output.exists())
        self.assertNotIn(str(self.fixture.root), rendered.getvalue())


if __name__ == "__main__":
    unittest.main()
