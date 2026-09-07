from __future__ import annotations

import time
from typing import Any

from .errors import ApplianceError
from .limits import SSE_PAYLOAD_BYTES
from .primary_profile import PRIMARY_SYSTEM_PROMPT
from .profile_preflight import (
    Fixture,
    LoopbackClient,
    PreflightFailure,
    score_output,
)
from .strict_json import canonical_bytes, parse_body


class PrimaryQualityClient(LoopbackClient):
    """Primary-only SSE scorer without mutating the frozen Qwen scorer."""

    def stream(self, fixture: Fixture) -> dict[str, Any]:
        body = canonical_bytes(
            {
                "model": "jarvis-local",
                "messages": [
                    {"role": "system", "content": PRIMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": fixture.prompt},
                ],
                "max_completion_tokens": fixture.max_tokens,
                "temperature": 0,
                "top_p": 1,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        connection = self._connection()
        started = time.perf_counter()
        first_content: float | None = None
        parts: list[str] = []
        usage: dict[str, Any] | None = None
        finish_reason: str | None = None
        done = False
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers=self._headers(body),
            )
            response = connection.getresponse()
            if (
                response.status != 200
                or response.getheader("Content-Type")
                != "text/event-stream; charset=utf-8"
            ):
                raise PreflightFailure("primary quality stream request failed")
            while True:
                raw = response.readline(SSE_PAYLOAD_BYTES + 8)
                if not raw:
                    break
                if len(raw) > SSE_PAYLOAD_BYTES + 7 or not raw.endswith(b"\n"):
                    raise PreflightFailure("invalid primary quality SSE frame size")
                if not raw.startswith(b"data: "):
                    if raw.strip():
                        raise PreflightFailure("invalid primary quality SSE framing")
                    continue
                payload = raw[6:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                try:
                    event = parse_body(payload)
                except (ApplianceError, RecursionError) as exc:
                    raise PreflightFailure("invalid primary quality SSE JSON") from exc
                if not isinstance(event, dict):
                    raise PreflightFailure("invalid primary quality SSE event")
                choices = event.get("choices")
                if choices == []:
                    candidate = event.get("usage")
                    if finish_reason is None or usage is not None or not isinstance(candidate, dict):
                        raise PreflightFailure("invalid primary quality usage ordering")
                    usage = candidate
                    continue
                if (
                    finish_reason is not None
                    or not isinstance(choices, list)
                    or len(choices) != 1
                    or not isinstance(choices[0], dict)
                ):
                    raise PreflightFailure("invalid primary quality SSE choice")
                choice = choices[0]
                delta = choice.get("delta")
                if not isinstance(delta, dict) or set(delta) - {"role", "content"} or delta.get("role", "assistant") != "assistant":
                    raise PreflightFailure("invalid primary quality SSE delta")
                content = delta.get("content")
                if content is not None and not isinstance(content, str):
                    raise PreflightFailure("invalid primary quality text content")
                if isinstance(content, str) and content:
                    if first_content is None:
                        first_content = time.perf_counter()
                    parts.append(content)
                candidate_reason = choice.get("finish_reason")
                if candidate_reason is not None:
                    if candidate_reason not in ("stop", "length"):
                        raise PreflightFailure("invalid primary quality finish reason")
                    finish_reason = candidate_reason
        finally:
            ended = time.perf_counter()
            connection.close()
        if first_content is None or finish_reason is None or not done or not isinstance(usage, dict):
            raise PreflightFailure("incomplete primary quality SSE response")
        completion_tokens = usage.get("completion_tokens")
        if (set(usage) != {"prompt_tokens", "completion_tokens", "total_tokens"}
                or any(type(value) is not int or value < 0 for value in usage.values())
                or not 1 <= completion_tokens <= fixture.max_tokens
                or usage["total_tokens"] != usage["prompt_tokens"] + completion_tokens):
            raise PreflightFailure("invalid primary completion token accounting")
        text = "".join(parts).strip()
        score = score_output(fixture, text, finish_reason)
        decode_seconds = max(0.000001, ended - first_content)
        return {
            "case_id": fixture.case_id,
            "language": fixture.language,
            "category": fixture.category,
            "protocol_pass": True,
            **score,
            "finish_reason": finish_reason,
            "ttft_seconds": first_content - started,
            "end_to_end_seconds": ended - started,
            "completion_tokens": completion_tokens,
            "decode_tokens_per_second": completion_tokens / decode_seconds,
            "output_utf8_bytes": len(text.encode("utf-8")),
        }
