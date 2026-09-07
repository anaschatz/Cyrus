from __future__ import annotations

import ctypes
import hashlib
import io
import json
import unittest
from ctypes import wintypes
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jarvis_inference.strict_json import canonical_bytes
from jarvis_inference.windows_primary_metrics import (
    PrimaryQualificationClient,
    PrimaryRunnerFailure,
    windows_process_working_set_bytes,
)


def _tokenization() -> dict[str, object]:
    return {
        "token_count": 3,
        "token_ids_sha256": hashlib.sha256(
            canonical_bytes([7, 11, 13])
        ).hexdigest(),
    }


def _metric(serial: int = 1) -> dict[str, object]:
    tokenization = _tokenization()
    return {
        "generation_serial": serial,
        "prompt_token_count": 3,
        "prompt_token_ids_sha256": tokenization["token_ids_sha256"],
        "accepted_token_count": 2,
        "accepted_token_ids_sha256": hashlib.sha256(
            canonical_bytes([17, 19])
        ).hexdigest(),
        "request_ttft_seconds": 0.75,
        "backend_prompt_seconds": 0.5,
        "backend_decode_seconds": 0.5,
        "backend_decode_tokens_per_second": 4.0,
        "request_to_engine_final_seconds": 1.25,
    }


class WindowsPrimaryMetricClientTests(unittest.TestCase):
    def test_buffered_finish_reason_rejects_nonstrings_with_a_typed_error(self) -> None:
        for reason in ("stop", "length", [], {}, None, True):
            with self.subTest(reason=reason):
                document = {
                    "choices": [{"message": {"content": "yes"}, "finish_reason": reason}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                }
                response = Mock(status=200)
                response.read.return_value = canonical_bytes(document)
                connection = Mock()
                connection.getresponse.return_value = response
                client = self.client()
                client.clock = Mock(side_effect=[1.0, 2.0])
                with patch.object(client, "_connection", return_value=connection):
                    if isinstance(reason, str):
                        self.assertEqual(client._observe_buffered({}, 3)["finish_reason"], reason)
                    else:
                        with self.assertRaises(PrimaryRunnerFailure):
                            client._observe_buffered({}, 3)
                connection.close.assert_called_once()

    def test_stream_metrics_require_ordered_complete_frames_and_integer_usage(
        self,
    ) -> None:
        def frame(event):
            return b"data: " + json.dumps(event).encode("utf-8") + b"\n"

        content = frame(
            {"choices": [{"delta": {"content": "yes"}, "finish_reason": None}]}
        )
        terminal = frame({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        usage = frame(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            }
        )
        baseline = [content, terminal, usage, b"data: [DONE]\n"]
        tool_stream = [
            frame(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "lookup",
                                            "arguments": "{}",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            ),
            frame({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            usage,
            baseline[-1],
        ]
        invalid = [
            [content, usage, terminal, baseline[-1]],
            [content, terminal, terminal, usage, baseline[-1]],
            [content, terminal, content, usage, baseline[-1]],
            [content, terminal, usage, usage, baseline[-1]],
            baseline[:-1] + [baseline[-1].rstrip()],
            [b"data: " + b"[" * 2000 + b"0" + b"]" * 2000 + b"\n"],
            [
                content,
                terminal,
                frame(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 1,
                            "total_tokens": 4.0,
                        },
                    }
                ),
                baseline[-1],
            ],
        ]
        for delta, reason in (
            ({"content": 4}, None),
            ({"tool_calls": [4]}, None),
            ({}, []),
        ):
            invalid.append(
                [
                    content,
                    frame({"choices": [{"delta": delta, "finish_reason": reason}]}),
                    terminal,
                    usage,
                    baseline[-1],
                ]
            )
        for index, lines in enumerate([baseline, tool_stream] + invalid):
            with self.subTest(index=index):
                response = Mock(readline=io.BytesIO(b"".join(lines)).readline)
                response.status = 200
                response.getheader.return_value = "text/event-stream; charset=utf-8"
                connection = Mock()
                connection.getresponse.return_value = response
                client = self.client()
                client.clock = Mock(side_effect=[1.0, 2.0])
                with patch.object(client, "_connection", return_value=connection):
                    if index < 2:
                        self.assertEqual(
                            client._observe_stream({}, 3)["completion_tokens"], 1
                        )
                    else:
                        with self.assertRaises(PrimaryRunnerFailure):
                            client._observe_stream({}, 3)
                connection.close.assert_called_once()

    def test_process_memory_preserves_pointer_width_and_closes_on_failure(self) -> None:
        handle = (1 << 40) + 123
        opened = ctypes.CFUNCTYPE(
            wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD
        )(lambda *args: handle)
        # Emulate an unconfigured DLL function's default C-int result.
        opened.restype = ctypes.c_int
        closed = []
        close_handle = ctypes.CFUNCTYPE(wintypes.BOOL, wintypes.HANDLE)(
            lambda value: closed.append(value) or 1
        )
        received = []

        def sample(value, counters, size):
            received.append(value)
            counters._obj.WorkingSetSize = 3_000_000_000
            return success

        memory_info = Mock(side_effect=sample)
        kernel32 = SimpleNamespace(OpenProcess=opened, CloseHandle=close_handle)
        psapi = SimpleNamespace(GetProcessMemoryInfo=memory_info)

        for success in (1, 0):
            with (
                self.subTest(success=success),
                patch("jarvis_inference.windows_primary_metrics.os.name", "nt"),
                patch.object(
                    ctypes,
                    "windll",
                    SimpleNamespace(kernel32=kernel32, psapi=psapi),
                    create=True,
                ),
                patch.object(
                    ctypes,
                    "WinDLL",
                    side_effect=lambda name, **kwargs: (
                        kernel32 if name == "kernel32" else psapi
                    ),
                    create=True,
                ),
            ):
                if success:
                    self.assertEqual(
                        windows_process_working_set_bytes(42), 3_000_000_000
                    )
                else:
                    with self.assertRaises(PrimaryRunnerFailure):
                        windows_process_working_set_bytes(42)
                self.assertEqual(received[-1], handle)
                self.assertEqual(closed[-1], handle)

    def client(self) -> PrimaryQualificationClient:
        return PrimaryQualificationClient(
            "http://127.0.0.1:8766/v1", "k" * 48
        )

    def test_arm_public_request_and_exact_one_shot_metric_are_cross_bound(self) -> None:
        client = self.client()
        calls: list[tuple[str, str, object]] = []

        def request(method, path, payload=None):
            calls.append((method, path, payload))
            if path.endswith("/arm"):
                return 200, {"generation_serial": 1}
            return 200, _metric()

        public = {
            "end_to_end_seconds": 1.5,
            "completion_tokens": 2,
            "finish_reason": "stop",
        }
        with patch.object(client, "json_request", side_effect=request), patch.object(
            client, "_observe_buffered", return_value=public
        ):
            observed = client.observe(
                {"stream": False}, _tokenization()  # type: ignore[arg-type]
            )

        self.assertEqual(
            [path for _method, path, _payload in calls],
            [
                "/jarvis/qualification/generation/arm",
                "/jarvis/qualification/generation/metric",
            ],
        )
        self.assertEqual(calls[1][2], {"expected_serial": 1})
        self.assertEqual(observed["generation_serial"], 1)
        self.assertEqual(observed["ttft_seconds"], 0.75)
        self.assertEqual(observed["accepted_decode_tokens"], 2)
        serialized = canonical_bytes(observed).decode("ascii")
        self.assertNotIn('"content"', serialized)
        self.assertNotIn('"token_ids"', serialized)

    def test_serial_gap_extra_content_and_impossible_timings_fail_closed(self) -> None:
        client = self.client()
        with patch.object(
            client,
            "json_request",
            return_value=(200, {"generation_serial": 2}),
        ):
            with self.assertRaisesRegex(PrimaryRunnerFailure, "serial"):
                client._arm_generation()

        extra = _metric()
        extra["content"] = "must never escape"
        with patch.object(client, "json_request", return_value=(200, extra)):
            with self.assertRaisesRegex(PrimaryRunnerFailure, "shape"):
                client._consume_generation_metric(1)

        impossible = _metric()
        impossible["backend_prompt_seconds"] = 2.0
        impossible["backend_decode_seconds"] = 2.0
        impossible["backend_decode_tokens_per_second"] = 1.0
        impossible["request_to_engine_final_seconds"] = 3.0
        with patch.object(client, "json_request", return_value=(200, impossible)):
            with self.assertRaisesRegex(PrimaryRunnerFailure, "inconsistent"):
                client._consume_generation_metric(1)


if __name__ == "__main__":
    unittest.main()
