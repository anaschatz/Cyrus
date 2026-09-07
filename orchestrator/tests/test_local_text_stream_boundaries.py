from __future__ import annotations

import asyncio
import json
import unittest

import httpx
import _bootstrap  # noqa: F401
from jarvis_orchestrator.domain import ModelRequest, OrchestratorError
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.pydantic_adapter import _DirectLocalTextStreamPort
from test_model_stream_ownership import Store


def frame(value):
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


def content(text="ok", index=0, finish=None):
    return frame(
        {
            "choices": [
                {"index": index, "delta": {"content": text}, "finish_reason": finish}
            ]
        }
    )


def usage(count=2, total=5):
    return frame(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": count,
                "total_tokens": total,
            },
        }
    )


STOP = frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
DONE = b"data: [DONE]\n\n"
VALID = content() + STOP + usage() + DONE


class LocalTextStreamBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def port(self, body, *, headers=None, stream=None):
        store = Store()
        responses = []

        def respond(request):
            response = httpx.Response(
                200,
                headers=headers or {"content-type": "text/event-stream"},
                stream=stream if stream is not None else httpx.ByteStream(body),
            )
            responses.append(response)
            return response

        port = BudgetedModelPort(
            store,
            _DirectLocalTextStreamPort(
                base_url="http://127.0.0.1:9/v1",
                bearer_token="synthetic",
                timeout_seconds=1,
                http_transport=httpx.MockTransport(respond),
            ),
        )
        return (
            port.stream(ModelRequest("task", 1, "final", "system", (), 16)),
            store,
            responses,
        )

    async def reject(self, body, **kwargs):
        stream, store, responses = self.port(body, **kwargs)
        try:
            with self.assertRaisesRegex(OrchestratorError, "backend_failure"):
                async for _ in stream:
                    pass
        finally:
            await stream.aclose()
        self.assertEqual(store.reconciled, [None])
        self.assertTrue(responses[0].is_closed)

    async def test_requires_complete_ordered_terminal_and_usage(self):
        for raw in (
            b"",
            content(),
            content() + STOP,
            content() + STOP + DONE,
            content() + STOP + usage(),
            VALID.rstrip(),
            usage() + VALID,
            content() + STOP + STOP + usage() + DONE,
            content() + STOP + usage() + usage() + DONE,
            VALID + content(),
            content() + STOP + content() + usage() + DONE,
            content(finish="tool_calls") + usage() + DONE,
        ):
            with self.subTest(raw=raw):
                await self.reject(raw)

    async def test_rejects_invalid_token_accounting(self):
        for count, total in (
            (True, 4),
            (-1, 2),
            (1.5, 4.5),
            ("2", 5),
            (None, 3),
            (17, 20),
            (2, 6),
        ):
            with self.subTest(count=count, total=total):
                await self.reject(content() + STOP + usage(count, total) + DONE)

    async def test_rejects_malformed_choice_json_and_nontext_deltas(self):
        for event in (
            content(index=True),
            content(index=1),
            content(text=[]),
            frame({"choices": [{"index": 0, "delta": {"tool_calls": []}}]}),
            b'data: {"choices":[],"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n',
            b"data: " + b"[" * 2000 + b"0" + b"]" * 2000 + b"\n\n",
            b'data: {"choices":[{"index":0,"delta":{"content":"\\ud800"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}],"x":NaN}\n\n',
        ):
            with self.subTest(size=len(event)):
                await self.reject(event + STOP + usage() + DONE)

    async def test_bounds_unterminated_frames_and_rejects_wrong_media(self):
        await self.reject(b"data: " + b"x" * (524288 + 1))
        await self.reject(VALID, headers={"content-type": "application/json"})
        await self.reject(
            VALID,
            headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
        )

    async def test_valid_fragmented_stream_remains_incremental_and_closes(self):
        for separator in (b"\n", b"\r\n"):

            class Stream(httpx.AsyncByteStream):
                closed = False

                async def __aiter__(self):
                    first = content("α").replace(b"\n", separator)
                    for byte in first:
                        yield bytes([byte])
                    yield (STOP + usage() + DONE).replace(b"\n", separator)

                async def aclose(self):
                    self.closed = True

            response_stream = Stream()
            stream, store, responses = self.port(b"", stream=response_stream)
            chunks = [chunk async for chunk in stream]
            self.assertEqual("".join(c.text for c in chunks), "α")
            self.assertEqual(store.reconciled, [2])
            self.assertTrue(response_stream.closed)
            self.assertTrue(responses[0].is_closed)

    async def test_frame_limit_fails_before_waiting_for_unbounded_peer(self):
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"data: " + b"x" * (524_288 + 2)
                await asyncio.Event().wait()

        await asyncio.wait_for(self.reject(b"", stream=Stream()), 0.5)

    async def test_length_finish_and_zero_usage_follow_appliance_contract(self):
        for count, finish in ((0, "stop"), (16, "length")):
            stream, store, _ = self.port(
                content(finish=finish) + usage(count, 3 + count) + DONE
            )
            chunks = [chunk async for chunk in stream]
            self.assertEqual(chunks[-1].usage.output_tokens, count)
            self.assertEqual(store.reconciled, [count])

    async def test_first_text_does_not_wait_for_completion(self):
        release = asyncio.Event()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield content()
                await release.wait()
                yield STOP + usage() + DONE

        stream, store, _ = self.port(b"", stream=Stream())
        try:
            first = await asyncio.wait_for(anext(stream), 0.5)
            self.assertEqual(first.text, "ok")
        finally:
            release.set()
            await stream.aclose()
        self.assertEqual(store.reconciled, [None])


if __name__ == "__main__":
    unittest.main()
