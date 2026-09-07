"""The actual SDK transport must keep model input on the pinned loopback route."""

import asyncio

import httpx
import pytest
from openai import APIStatusError

from anamnesis.vllm_runtime import OpenAIExternalVllmClient


REQUEST = {
    "model": "synthetic-model",
    "messages": [{"role": "user", "content": "synthetic data only"}],
    "chat_template_kwargs": {"enable_thinking": False},
}


def _client():
    return OpenAIExternalVllmClient(
        base_url="http://127.0.0.1:18002/v1",
        api_key="synthetic-test-key",
        request_timeout_seconds=5,
    )


@pytest.mark.parametrize(
    "variable", ["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"]
)
def test_loopback_completion_ignores_environment_proxy(monkeypatch, variable):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv(variable, "http://127.0.0.1:19999")

    async def run():
        client = _client()
        http = client._client._client
        transports = []

        async def respond(transport, request):
            transports.append(transport)
            assert set(request.extensions["timeout"].values()) == {5}
            assert str(request.url) == "http://127.0.0.1:18002/v1/chat/completions"
            return httpx.Response(
                200,
                json={
                    "id": "synthetic",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "synthetic-model",
                    "choices": [],
                },
            )

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
        try:
            await client.complete(REQUEST)
            assert transports == [http._transport]
            assert client._client.max_retries == 0
        finally:
            await client.aclose()
        assert http.is_closed

    asyncio.run(run())


@pytest.mark.parametrize("status", [307, 308])
def test_loopback_completion_does_not_follow_redirect(monkeypatch, status):
    async def run():
        client = _client()
        seen = []

        def respond(request):
            seen.append(str(request.url))
            if len(seen) == 1:
                return httpx.Response(
                    status,
                    headers={
                        "Location": "https://outside.example.invalid/capture",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "id": "synthetic",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "synthetic-model",
                    "choices": [],
                },
            )

        transport = httpx.MockTransport(respond)
        monkeypatch.setattr(
            client._client._client, "_transport_for_url", lambda url: transport
        )
        try:
            with pytest.raises(APIStatusError) as error:
                await client.complete(REQUEST)
            assert error.value.status_code == status
            assert seen == ["http://127.0.0.1:18002/v1/chat/completions"]
        finally:
            await client.aclose()

    asyncio.run(run())
