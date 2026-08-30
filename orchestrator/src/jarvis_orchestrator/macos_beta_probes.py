"""Bounded loopback readiness probes for the foreground macOS beta."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from .config import OrchestratorConfig
from .llm_context import ENGLISH_ONLY_SYSTEM_PROMPT, SYSTEM_PROMPT
from .stack_config import PrivateStackConfig

UrlOpen = Callable[..., Any]


def inference_warmup(
    _stack: PrivateStackConfig,
    config: OrchestratorConfig,
    *,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> bool:
    """Pay the one-time local generation cost before Voice becomes ready."""

    try:
        parsed = urllib.parse.urlparse(config.inference_base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port is None
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
        request = urllib.request.Request(
            f"http://127.0.0.1:{parsed.port}/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {config.inference_bearer_token}",
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "Host": f"127.0.0.1:{parsed.port}",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        parts: list[str] = []
        finish_reason: str | None = None
        done = False
        total_bytes = 0
        with urlopen(request, timeout=10.0) as response:
            if (
                response.status != 200
                or response.headers.get("Content-Type")
                != "text/event-stream; charset=utf-8"
            ):
                return False
            while total_bytes <= 256 * 1024:
                line = response.readline(64 * 1024 + 1)
                if not line:
                    break
                total_bytes += len(line)
                if len(line) > 64 * 1024 or total_bytes > 256 * 1024:
                    return False
                if not line.startswith(b"data: "):
                    if line.strip():
                        return False
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                event = json.loads(payload.decode("utf-8", errors="strict"))
                if not isinstance(event, dict):
                    return False
                choices = event.get("choices")
                if choices == []:
                    continue
                if (
                    not isinstance(choices, list)
                    or len(choices) != 1
                    or not isinstance(choices[0], dict)
                ):
                    return False
                choice = choices[0]
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    return False
                content = delta.get("content")
                if content is not None:
                    if not isinstance(content, str):
                        return False
                    parts.append(content)
                    if len("".join(parts).encode("utf-8")) > 64:
                        return False
                candidate = choice.get("finish_reason")
                if candidate is not None:
                    if not isinstance(candidate, str):
                        return False
                    finish_reason = candidate
        return (
            done
            and finish_reason == "stop"
            and "".join(parts).strip().casefold() == "ready"
        )
    except (
        OSError,
        TimeoutError,
        UnicodeError,
        ValueError,
        urllib.error.URLError,
    ):
        return False


def orchestrator_http_ready(
    config: OrchestratorConfig,
    *,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> bool:
    request = urllib.request.Request(
        f"http://127.0.0.1:{config.control_port}/readyz",
        headers={"Host": f"127.0.0.1:{config.control_port}"},
    )
    try:
        with urlopen(request, timeout=2.0) as response:
            body = response.read(4 * 1024 + 1)
        return (
            response.status == 200
            and len(body) <= 4 * 1024
            and json.loads(body.decode("utf-8", errors="strict"))
            == {"ready": True}
        )
    except (
        OSError,
        TimeoutError,
        UnicodeError,
        ValueError,
        urllib.error.URLError,
    ):
        return False


__all__ = ["inference_warmup", "orchestrator_http_ready"]
