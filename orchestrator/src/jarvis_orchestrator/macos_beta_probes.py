"""Bounded loopback readiness probes for the foreground macOS beta."""

from __future__ import annotations

import asyncio
import json
import urllib.parse

import httpx

from .config import OrchestratorConfig
from .inference import InferenceControlClient, parse_inference_json
from .llm_context import ENGLISH_ONLY_SYSTEM_PROMPT, SYSTEM_PROMPT
from .stack_config import PrivateStackConfig


def inference_warmup(
    _stack: PrivateStackConfig,
    config: OrchestratorConfig,
) -> bool:
    """Pay the one-time local generation cost before Voice becomes ready."""

    try:
        parsed = urllib.parse.urlparse(config.inference_base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port is None
            or not 1 <= parsed.port <= 65535
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path.rstrip("/") != "/v1"
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            return False
        body = json.dumps(
            {
                "model": config.model_alias,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            ENGLISH_ONLY_SYSTEM_PROMPT
                            if config.response_language == "en"
                            else SYSTEM_PROMPT
                        ),
                    },
                    {"role": "user", "content": "Reply with only this word: ready"},
                ],
                "max_completion_tokens": 16,
                "temperature": 0,
                "top_p": 1,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        async def request_body() -> bytes:
            async with asyncio.timeout(10.0):
                async with httpx.AsyncClient(
                    timeout=10.0, trust_env=False, follow_redirects=False
                ) as client:
                    async with client.stream(
                        "POST",
                        f"http://127.0.0.1:{parsed.port}/v1/chat/completions",
                        content=body,
                        headers={
                            "Authorization": f"Bearer {config.inference_bearer_token}",
                            "Content-Type": "application/json",
                            "Accept": "text/event-stream",
                            "Accept-Encoding": "identity",
                        },
                    ) as response:
                        if (
                            response.status_code != 200
                            or response.headers.get("content-type")
                            != "text/event-stream; charset=utf-8"
                            or response.headers.get(
                                "content-encoding", "identity"
                            ).lower()
                            != "identity"
                        ):
                            raise ValueError
                        received = bytearray()
                        async for chunk in response.aiter_bytes(chunk_size=4096):
                            received.extend(chunk)
                            if len(received) > 256 * 1024:
                                raise ValueError
                        return bytes(received)

        return _warmup_response_ready(asyncio.run(request_body()))
    except (OSError, ValueError, httpx.HTTPError):
        return False


def _warmup_response_ready(raw: bytes) -> bool:
    frames = raw.replace(b"\r\n", b"\n").split(b"\n\n")
    if frames[-1] or len(frames) < 2 or frames[-2] != b"data: [DONE]":
        return False
    parts: list[str] = []
    finished = False
    usage_seen = False
    for frame in frames[:-2]:
        if not frame:
            continue
        if not frame.startswith(b"data: ") or b"\n" in frame or len(frame) > 64 * 1024:
            return False
        event = parse_inference_json(frame[6:], maximum_bytes=64 * 1024)
        choices = event.get("choices")
        if choices == []:
            usage = event.get("usage")
            if not finished or usage_seen or not isinstance(usage, dict):
                return False
            counts = [
                usage.get(name)
                for name in ("prompt_tokens", "completion_tokens", "total_tokens")
            ]
            if (
                any(type(value) is not int or value < 0 for value in counts)
                or not 1 <= counts[1] <= 16
                or counts[2] != counts[0] + counts[1]
            ):
                return False
            usage_seen = True
            continue
        if (
            finished
            or not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
        ):
            return False
        choice = choices[0]
        delta = choice.get("delta")
        if (
            type(choice.get("index")) is not int
            or choice["index"] != 0
            or not isinstance(delta, dict)
            or set(delta) - {"role", "content"}
            or ("role" in delta and delta["role"] != "assistant")
        ):
            return False
        content = delta.get("content")
        if content is not None:
            if not isinstance(content, str):
                return False
            parts.append(content)
            if len("".join(parts).encode("utf-8")) > 64:
                return False
        finish = choice.get("finish_reason")
        if finish is not None:
            if finish != "stop":
                return False
            finished = True
    return finished and usage_seen and "".join(parts).strip().casefold() == "ready"


def orchestrator_http_ready(config: OrchestratorConfig) -> bool:
    return InferenceControlClient(
        f"http://127.0.0.1:{config.control_port}/v1",
        config.control_bearer_token,
        timeout_seconds=2.0,
    ).ready()


__all__ = ["inference_warmup", "orchestrator_http_ready"]
