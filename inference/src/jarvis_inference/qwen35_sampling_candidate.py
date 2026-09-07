"""One-shot, candidate-only Qwen3.5 9B official-sampling diagnostic.

This module intentionally has no production gateway or model-spawn integration.
It can only interrogate an already running, authenticated IPv4-loopback raw
llama-server and write content-free evidence for the frozen challenger cell.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import stat
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

from .limits import BACKEND_EVENT_BYTES, BODY_BYTES, SSE_PAYLOAD_BYTES
from .profile_preflight import FALLBACK_SYSTEM_PROMPT, Fixture, PreflightFailure, score_output
from .profile_quality import (
    FROZEN_CORPUS_SHA256,
    QUALITY_SUITE_VERSION,
    load_corpus,
    run_quality_gate,
)
from .strict_json import canonical_bytes, parse_body


EXPERIMENT_ID = "qwen35-9b-official-nonthinking-sampling-v1"
REPORT_SCHEMA_VERSION = "jarvis-qwen35-official-sampling-candidate/1"
MARKER_SCHEMA_VERSION = "jarvis-one-shot-generation-marker/1"
FROZEN_PREREGISTRATION_SHA256 = "4c28f5527ff5b3a9c968667792048de798fc5232bb907ea3eab6177797836d88"
FROZEN_SYSTEM_PROMPT_SHA256 = "05dbfe802df53da6c8660425016b9ea20fef0c745111c38701845d5c422cf4f2"
PUBLIC_MODEL_ALIAS = "jarvis-local"
WARMUPS = 2
GENERATION_DISPATCHES = 82

TREATMENT_SAMPLING: dict[str, int | float] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    "seed": 101,
}
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
EFFECTIVE_REQUEST_PROFILE = {
    "model": PUBLIC_MODEL_ALIAS,
    "sampling": TREATMENT_SAMPLING,
    "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
    "n": 1,
    "stream": True,
    "stream_options": {"include_usage": True},
    "tools": "absent",
    "reasoning_output": "forbidden",
}
SCORER_CODE_BUNDLE = {
    "aggregate_sha256": "ed1682eed182da50cf819bf1bc923a4fc1f728e252009ffb0736c27318f6286c",
    "files": [
        {
            "name": "limits.py",
            "sha256": "327f592cf7fa1f4368bb1a0e57062cc3d321353eb9d74bf7b6f5af689810ee7e",
            "size_bytes": 822,
        },
        {
            "name": "profile_preflight.py",
            "sha256": "b22a9659cf4e7c8f8a9629333b5e12545f91daa42402f4b0c21ccb17fc4f16e1",
            "size_bytes": 15_197,
        },
        {
            "name": "profile_quality.py",
            "sha256": "a7c5cf30101102117ca2d4486f63f65d3c66e7913f08db385c5a303f1abe0a8f",
            "size_bytes": 11_748,
        },
        {
            "name": "strict_json.py",
            "sha256": "a9875b61825fbd6032eceb4d93ea59109aacd595c91a0f022e4f48fcc2c879f1",
            "size_bytes": 5_778,
        },
    ],
}

EXPECTED_PREREGISTRATION: dict[str, Any] = {
    "schema_version": 1,
    "experiment_id": EXPERIMENT_ID,
    "status": "preregistered_unrun",
    "candidate_only": True,
    "authoritative_qualification": False,
    "public_model_alias": PUBLIC_MODEL_ALIAS,
    "suite_version": QUALITY_SUITE_VERSION,
    "corpus_sha256": FROZEN_CORPUS_SHA256,
    "system_prompt_sha256": FROZEN_SYSTEM_PROMPT_SHA256,
    "scorer_code_bundle": SCORER_CODE_BUNDLE,
    "model": {
        "description": "Qwen3.5 9B Ollama-format GGUF Q4_K_M",
        "sha256": "dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c",
        "size_bytes": 6_594_462_816,
    },
    "runner": {
        "description": "llama-server build 9840",
        "sha256": "b9bdd68928c150482049100d20bd446ccee5569ff3db386135855e0cb8fea66a",
        "size_bytes": 11_496_832,
    },
    "fixed_runtime": {
        "context_tokens": 8192,
        "slots": 1,
        "reasoning": "off",
        "native_chat_template": True,
        "offline": True,
        "authenticated_ipv4_loopback": True,
    },
    "baseline_sampling": {"temperature": 0, "top_p": 1},
    "treatment_sampling": TREATMENT_SAMPLING,
    "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
    "execution": {
        "warmups": WARMUPS,
        "generation_dispatches": GENERATION_DISPATCHES,
        "retries": 0,
        "repairs": 0,
        "max_runs": 1,
        "rerun_after_first_generation_dispatch": False,
        "tuning_after_result": False,
        "raw_prompt_persistence": False,
        "raw_completion_persistence": False,
    },
    "acceptance": {
        "greek_passes_at_least": 34,
        "english_passes_at_least": 34,
        "language_gap_at_most": 0.1,
        "protocol_completions": 80,
        "critical_failures": 0,
    },
    "decision_rule": (
        "Pass only if every original frozen threshold passes; otherwise reject this exact sampling "
        "profile without rerun or tuning on this corpus."
    ),
}


class CandidateFailure(PreflightFailure):
    """Sanitized failure for the isolated candidate harness."""


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class RenderedTokenIdentity:
    case_id: str
    input_sha256: str
    rendered_prompt_sha256: str
    token_ids_sha256: str
    token_count: int

    def public(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "input_sha256": self.input_sha256,
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
            "token_ids_sha256": self.token_ids_sha256,
            "token_count": self.token_count,
        }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_small_regular(path: Path, *, maximum: int = BODY_BYTES) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CandidateFailure("candidate input is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1 or before.st_size > maximum:
            raise CandidateFailure("candidate input is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                raise CandidateFailure("candidate input changed during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CandidateFailure("candidate input changed during read")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise CandidateFailure("candidate input changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def hash_artifact(path: Path, *, expected_size: int, expected_sha256: str) -> ArtifactIdentity:
    if expected_size < 1 or not _is_sha256(expected_sha256):
        raise CandidateFailure("invalid frozen artifact identity")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CandidateFailure("candidate artifact is unavailable") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise CandidateFailure("candidate artifact size mismatch")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise CandidateFailure("candidate artifact changed during hashing")
    finally:
        os.close(descriptor)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise CandidateFailure("candidate artifact digest mismatch")
    return ArtifactIdentity(observed, expected_size)


def load_preregistration(path: Path) -> str:
    raw = _read_small_regular(path)
    digest = _sha256(raw)
    if digest != FROZEN_PREREGISTRATION_SHA256:
        raise CandidateFailure("candidate preregistration digest mismatch")
    try:
        document = parse_body(raw)
    except Exception as exc:
        raise CandidateFailure("candidate preregistration is invalid") from exc
    if canonical_bytes(document) != canonical_bytes(EXPECTED_PREREGISTRATION):
        raise CandidateFailure("candidate preregistration contract mismatch")
    return digest


def scorer_code_bundle_identity() -> dict[str, Any]:
    root = Path(__file__).parent
    observed: list[dict[str, Any]] = []
    for specification in SCORER_CODE_BUNDLE["files"]:
        name = specification["name"]
        if not isinstance(name, str) or Path(name).name != name:
            raise CandidateFailure("invalid scorer source identity")
        identity = hash_artifact(
            root / name,
            expected_size=specification["size_bytes"],
            expected_sha256=specification["sha256"],
        )
        observed.append({"name": name, "sha256": identity.sha256, "size_bytes": identity.size_bytes})
    aggregate = _sha256(canonical_bytes(observed))
    if aggregate != SCORER_CODE_BUNDLE["aggregate_sha256"]:
        raise CandidateFailure("scorer source bundle digest mismatch")
    return {"aggregate_sha256": aggregate, "files": observed}


def read_private_bearer(path: Path) -> str:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CandidateFailure("private bearer configuration is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
        or metadata.st_mode & 0o077
    ):
        raise CandidateFailure("private bearer configuration permissions are unsafe")
    try:
        value = parse_body(_read_small_regular(path))
    except Exception as exc:
        raise CandidateFailure("private bearer configuration is invalid") from exc
    bearer = value.get("inference_bearer_token")
    if (
        not isinstance(bearer, str)
        or not 32 <= len(bearer.encode("utf-8")) <= 4096
        or "\r" in bearer
        or "\n" in bearer
    ):
        raise CandidateFailure("private bearer configuration has no safe inference token")
    return bearer


def _private_parent_descriptor(path: Path) -> int:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise CandidateFailure("candidate destination must be an absolute file path")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.parent, flags)
    except OSError as exc:
        raise CandidateFailure("candidate destination parent is unavailable") from exc
    parent = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(parent.st_mode)
        or (hasattr(os, "getuid") and parent.st_uid != os.getuid())
        or parent.st_mode & 0o077
    ):
        os.close(descriptor)
        raise CandidateFailure("candidate destination parent permissions are unsafe")
    try:
        os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return descriptor
    except OSError:
        os.close(descriptor)
        raise CandidateFailure("candidate destination state is unsafe") from None
    else:
        os.close(descriptor)
        raise CandidateFailure("candidate destination already exists")


def _assert_private_destination(path: Path) -> None:
    descriptor = _private_parent_descriptor(path)
    os.close(descriptor)


def _write_private_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value) + b"\n"
    parent = _private_parent_descriptor(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=parent)
        except OSError as exc:
            raise CandidateFailure("candidate evidence could not be created") from exc
        try:
            written = 0
            while written < len(payload):
                count = os.write(descriptor, payload[written:])
                if count <= 0:
                    raise CandidateFailure("candidate evidence write was incomplete")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent)
    finally:
        os.close(parent)


class OneShotMarker:
    """Create the immutable no-rerun marker before the first generation call."""

    def __init__(self, path: Path, preregistration_sha256: str):
        self.path = path
        self.preregistration_sha256 = preregistration_sha256
        self.armed = False

    def arm(self) -> None:
        if self.armed:
            return
        _write_private_exclusive(self.path, {
            "schema_version": MARKER_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "preregistration_sha256": self.preregistration_sha256,
            "generation_dispatch_started": True,
            "rerun_permitted": False,
        })
        self.armed = True


def _json_content_type(value: str | None) -> bool:
    if value is None:
        return False
    parts = [part.strip().casefold() for part in value.split(";")]
    return parts == ["application/json"] or parts == ["application/json", "charset=utf-8"]


def _sse_content_type(value: str | None) -> bool:
    if value is None:
        return False
    parts = [part.strip().casefold() for part in value.split(";")]
    return parts == ["text/event-stream"] or parts == ["text/event-stream", "charset=utf-8"]


class OfficialSamplingCandidateClient:
    """Strict raw llama-server client used only by the preregistered cell."""

    def __init__(
        self,
        base_url: str,
        bearer: str,
        *,
        before_first_generation: Callable[[], None],
    ):
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port is None
            or parsed.path not in {"/v1", "/v1/"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise CandidateFailure("candidate endpoint must be exact IPv4 loopback /v1")
        if (
            not 32 <= len(bearer.encode("utf-8")) <= 4096
            or "\r" in bearer
            or "\n" in bearer
        ):
            raise CandidateFailure("candidate bearer is invalid")
        self.host = "127.0.0.1"
        self.port = parsed.port
        self.bearer = bearer
        self.before_first_generation = before_first_generation
        self.generation_dispatches = 0
        self._rendered: dict[str, RenderedTokenIdentity] = {}
        self._rendered_order: list[str] = []

    def _connection(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=180.0)

    def _headers(self, body: bytes | None = None, *, accept: str = "application/json") -> dict[str, str]:
        result = {
            "Authorization": "Bearer " + self.bearer,
            "Host": f"127.0.0.1:{self.port}",
            "Accept": accept,
        }
        if body is not None:
            result.update({"Content-Type": "application/json", "Content-Length": str(len(body))})
        return result

    def _json_request(self, method: str, path: str, value: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if value is None else canonical_bytes(value)
        connection = self._connection()
        try:
            connection.request(method, path, body=body, headers=self._headers(body))
            response = connection.getresponse()
            raw = response.read(BODY_BYTES + 1)
            if (
                response.status != 200
                or not _json_content_type(response.getheader("Content-Type"))
                or len(raw) > BODY_BYTES
            ):
                raise CandidateFailure("candidate JSON endpoint failed")
            try:
                return parse_body(raw)
            except Exception as exc:
                raise CandidateFailure("candidate JSON endpoint returned invalid JSON") from exc
        except OSError as exc:
            raise CandidateFailure("candidate loopback endpoint is unavailable") from exc
        finally:
            connection.close()

    def readiness(self) -> None:
        value = self._json_request("GET", "/health")
        if value.get("status") != "ok":
            raise CandidateFailure("candidate server is not ready")

    def capabilities(self) -> None:
        value = self._json_request("GET", "/v1/models")
        data = value.get("data")
        if value.get("object") != "list" or not isinstance(data, list) or len(data) != 1:
            raise CandidateFailure("candidate model registry is invalid")
        item = data[0]
        if not isinstance(item, dict) or item.get("id") != PUBLIC_MODEL_ALIAS:
            raise CandidateFailure("candidate public model alias mismatch")

    @staticmethod
    def _messages(fixture: Fixture) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
            {"role": "user", "content": fixture.prompt},
        ]

    def prepare_rendered_tokens(self, fixtures: Iterable[Fixture]) -> None:
        if self._rendered:
            raise CandidateFailure("rendered-token preflight may run only once")
        self.readiness()
        self.capabilities()
        for fixture in fixtures:
            if fixture.case_id in self._rendered:
                raise CandidateFailure("rendered-token fixture identifiers must be unique")
            messages = self._messages(fixture)
            input_sha256 = _sha256(canonical_bytes(messages))
            applied = self._json_request("POST", "/apply-template", {
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            })
            if set(applied) != {"prompt"} or not isinstance(applied["prompt"], str):
                raise CandidateFailure("candidate template response is invalid")
            rendered = applied["prompt"]
            expected_suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            if not rendered or not rendered.endswith(expected_suffix):
                raise CandidateFailure("candidate template did not apply non-thinking mode")
            tokenized = self._json_request("POST", "/tokenize", {
                "content": rendered,
                "add_special": True,
                "parse_special": True,
                "with_pieces": False,
            })
            if set(tokenized) != {"tokens"}:
                raise CandidateFailure("candidate tokenizer response is invalid")
            tokens = tokenized["tokens"]
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(type(token) is not int or token < 0 for token in tokens)
            ):
                raise CandidateFailure("candidate tokenizer returned invalid token IDs")
            identity = RenderedTokenIdentity(
                case_id=fixture.case_id,
                input_sha256=input_sha256,
                rendered_prompt_sha256=_sha256(rendered.encode("utf-8")),
                token_ids_sha256=_sha256(canonical_bytes(tokens)),
                token_count=len(tokens),
            )
            self._rendered[fixture.case_id] = identity
            self._rendered_order.append(fixture.case_id)

    def rendered_token_identity(self) -> dict[str, Any]:
        if not self._rendered_order:
            raise CandidateFailure("rendered-token identity is unavailable")
        cases = [self._rendered[case_id].public() for case_id in self._rendered_order]
        return {
            "unique_fixture_count": len(cases),
            "aggregate_sha256": _sha256(canonical_bytes(cases)),
            "cases": cases,
        }

    @staticmethod
    def _usage(
        value: Any,
        *,
        expected_prompt_tokens: int,
        max_completion_tokens: int,
    ) -> dict[str, int]:
        if not isinstance(value, dict) or set(value) != {
            "prompt_tokens", "completion_tokens", "total_tokens",
        }:
            raise CandidateFailure("candidate stream usage is invalid")
        if any(type(value[name]) is not int for name in value):
            raise CandidateFailure("candidate stream usage is invalid")
        prompt = value["prompt_tokens"]
        completion = value["completion_tokens"]
        total = value["total_tokens"]
        if (
            prompt != expected_prompt_tokens
            or not 1 <= completion <= max_completion_tokens
            or total != prompt + completion
        ):
            raise CandidateFailure("candidate stream usage does not match rendered tokens")
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}

    @staticmethod
    def _validate_event_envelope(event: dict[str, Any], stream_id: str | None) -> str:
        allowed = {
            "id", "object", "created", "model", "system_fingerprint", "choices", "usage", "timings",
        }
        if set(event) - allowed:
            raise CandidateFailure("candidate stream event has unknown fields")
        if event.get("object") != "chat.completion.chunk" or event.get("model") != PUBLIC_MODEL_ALIAS:
            raise CandidateFailure("candidate stream identity mismatch")
        event_id = event.get("id")
        if not isinstance(event_id, str) or not event_id or (stream_id is not None and event_id != stream_id):
            raise CandidateFailure("candidate stream event ID mismatch")
        created = event.get("created")
        if type(created) is not int or created < 0:
            raise CandidateFailure("candidate stream timestamp is invalid")
        fingerprint = event.get("system_fingerprint")
        if fingerprint is not None and not isinstance(fingerprint, str):
            raise CandidateFailure("candidate stream fingerprint is invalid")
        timings = event.get("timings")
        if timings is not None and not isinstance(timings, dict):
            raise CandidateFailure("candidate stream timings are invalid")
        return event_id

    def _read_sse(
        self,
        response: http.client.HTTPResponse,
        fixture: Fixture,
        *,
        started: float,
        expected_prompt_tokens: int,
    ) -> dict[str, Any]:
        if response.status != 200 or not _sse_content_type(response.getheader("Content-Type")):
            raise CandidateFailure("candidate stream response failed")
        first_content: float | None = None
        parts: list[str] = []
        output_bytes = 0
        usage: dict[str, int] | None = None
        finish_reason: str | None = None
        done = False
        stream_id: str | None = None
        finished = False
        while True:
            raw = response.readline(BACKEND_EVENT_BYTES + 1)
            if not raw:
                break
            if len(raw) > BACKEND_EVENT_BYTES or not raw.endswith(b"\n"):
                raise CandidateFailure("candidate stream event size or framing is invalid")
            if not raw.strip() or raw.startswith(b":"):
                continue
            if not raw.startswith(b"data: "):
                raise CandidateFailure("candidate stream framing is invalid")
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                done = True
                break
            try:
                event = parse_body(payload)
            except Exception as exc:
                raise CandidateFailure("candidate stream event is invalid JSON") from exc
            stream_id = self._validate_event_envelope(event, stream_id)
            choices = event.get("choices")
            if not isinstance(choices, list):
                raise CandidateFailure("candidate stream choices are invalid")
            if choices and event.get("usage") is not None:
                raise CandidateFailure("candidate stream mixed usage with content")
            if not choices:
                if usage is not None or not finished:
                    raise CandidateFailure("candidate stream usage ordering is invalid")
                usage = self._usage(
                    event.get("usage"),
                    expected_prompt_tokens=expected_prompt_tokens,
                    max_completion_tokens=fixture.max_tokens,
                )
                continue
            if len(choices) != 1 or usage is not None or finished:
                raise CandidateFailure("candidate stream choice ordering is invalid")
            choice = choices[0]
            if not isinstance(choice, dict) or set(choice) - {"index", "delta", "finish_reason", "logprobs"}:
                raise CandidateFailure("candidate stream choice shape is invalid")
            if type(choice.get("index")) is not int or choice["index"] != 0 or choice.get("logprobs") is not None:
                raise CandidateFailure("candidate stream choice metadata is invalid")
            delta = choice.get("delta")
            if not isinstance(delta, dict) or set(delta) - {
                "role", "content", "reasoning_content", "reasoning", "tool_calls", "function_call",
            }:
                raise CandidateFailure("candidate stream delta shape is invalid")
            role = delta.get("role")
            if role is not None and role != "assistant":
                raise CandidateFailure("candidate stream role is invalid")
            if delta.get("reasoning_content") not in (None, "") or delta.get("reasoning") not in (None, ""):
                raise CandidateFailure("candidate emitted reasoning output")
            tool_calls = delta.get("tool_calls")
            if tool_calls is not None and tool_calls != [] and tool_calls != ():
                raise CandidateFailure("candidate emitted tool output")
            function_call = delta.get("function_call")
            if function_call is not None and function_call != {} and function_call != ():
                raise CandidateFailure("candidate emitted tool output")
            content = delta.get("content")
            if content is not None and not isinstance(content, str):
                raise CandidateFailure("candidate stream content is invalid")
            if content:
                encoded = content.encode("utf-8")
                output_bytes += len(encoded)
                if output_bytes > SSE_PAYLOAD_BYTES:
                    raise CandidateFailure("candidate stream content is oversized")
                if first_content is None:
                    first_content = time.perf_counter()
                parts.append(content)
            reason = choice.get("finish_reason")
            if reason is not None:
                if reason not in ("stop", "length") or finish_reason is not None:
                    raise CandidateFailure("candidate stream finish reason is invalid")
                finish_reason = reason
                finished = True
        ended = time.perf_counter()
        if not done or not finished or usage is None or first_content is None:
            raise CandidateFailure("candidate stream is incomplete")
        text = "".join(parts).strip()
        lowered = text.casefold()
        if "<think" in lowered or "</think" in lowered or "<tool_call" in lowered:
            raise CandidateFailure("candidate emitted a forbidden reasoning or tool channel")
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
            "completion_tokens": usage["completion_tokens"],
            "decode_tokens_per_second": usage["completion_tokens"] / decode_seconds,
            "output_utf8_bytes": len(text.encode("utf-8")),
        }

    def stream(self, fixture: Fixture) -> dict[str, Any]:
        identity = self._rendered.get(fixture.case_id)
        messages = self._messages(fixture)
        if identity is None or identity.input_sha256 != _sha256(canonical_bytes(messages)):
            raise CandidateFailure("candidate fixture lacks rendered-token identity")
        if self.generation_dispatches >= GENERATION_DISPATCHES:
            raise CandidateFailure("candidate generation dispatch budget is exhausted")
        body = canonical_bytes({
            "model": PUBLIC_MODEL_ALIAS,
            "messages": messages,
            "max_completion_tokens": fixture.max_tokens,
            **TREATMENT_SAMPLING,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "n": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
        })
        self.before_first_generation()
        connection = self._connection()
        started = time.perf_counter()
        try:
            self.generation_dispatches += 1
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers=self._headers(body, accept="text/event-stream"),
            )
            response = connection.getresponse()
            return self._read_sse(
                response,
                fixture,
                started=started,
                expected_prompt_tokens=identity.token_count,
            )
        except OSError as exc:
            raise CandidateFailure("candidate generation endpoint is unavailable") from exc
        finally:
            connection.close()


def _component_counts(cases: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "protocol_pass",
        "language_ok",
        "exact_ok",
        "required_content_ok",
        "forbidden_content_ok",
        "word_limit_ok",
        "completed_ok",
        "quality_pass",
    )

    def counts(items: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        return {
            field: {"passed": sum(bool(item[field]) for item in items), "total": len(items)}
            for field in fields
        }

    return {
        "overall": counts(cases),
        "by_language": {
            language: counts([item for item in cases if item["language"] == language])
            for language in ("el", "en")
        },
        "by_category": {
            category: counts([item for item in cases if item["category"] == category])
            for category in ("extraction", "instruction", "knowledge", "reasoning", "summary")
        },
        "finish_reason_counts": dict(sorted(Counter(
            "none" if item["finish_reason"] is None else item["finish_reason"] for item in cases
        ).items())),
    }


def build_candidate_report(
    base_report: dict[str, Any],
    *,
    preregistration_sha256: str,
    model: ArtifactIdentity,
    runner: ArtifactIdentity,
    harness: ArtifactIdentity,
    scorer_code_bundle: dict[str, Any],
    client: OfficialSamplingCandidateClient,
) -> dict[str, Any]:
    cases = base_report.get("cases")
    if (
        base_report.get("suite_version") != QUALITY_SUITE_VERSION
        or base_report.get("corpus_sha256") != FROZEN_CORPUS_SHA256
        or base_report.get("system_prompt_sha256") != FROZEN_SYSTEM_PROMPT_SHA256
        or base_report.get("sample_count") != 80
        or not isinstance(cases, list)
        or len(cases) != 80
    ):
        raise CandidateFailure("frozen scorer report identity mismatch")
    if client.generation_dispatches != GENERATION_DISPATCHES:
        raise CandidateFailure("candidate dispatch count differs from preregistration")
    experiment_status = (
        "official_sampling_candidate_pass"
        if base_report.get("status") == "deterministic_quality_pass"
        else "official_sampling_candidate_failed"
    )
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": experiment_status,
        "candidate_only": True,
        "authoritative_qualification": False,
        "rerun_permitted": False,
        "raw_content_persisted": False,
        "identity": {
            "preregistration_sha256": preregistration_sha256,
            "corpus_sha256": FROZEN_CORPUS_SHA256,
            "system_prompt_sha256": FROZEN_SYSTEM_PROMPT_SHA256,
            "effective_request_profile_sha256": _sha256(canonical_bytes(EFFECTIVE_REQUEST_PROFILE)),
            "model_sha256": model.sha256,
            "model_size_bytes": model.size_bytes,
            "runner_sha256": runner.sha256,
            "runner_size_bytes": runner.size_bytes,
            "harness_sha256": harness.sha256,
            "harness_size_bytes": harness.size_bytes,
            "scorer_code_bundle": scorer_code_bundle,
            "rendered_tokens": client.rendered_token_identity(),
        },
        "generation_dispatches": client.generation_dispatches,
        "sampling": TREATMENT_SAMPLING,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "component_counts": _component_counts(cases),
        "frozen_quality_report": base_report,
    }


def _safe_harness_identity() -> ArtifactIdentity:
    path = Path(__file__)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise CandidateFailure("candidate harness identity is unavailable") from exc
    return hash_artifact(path, expected_size=size, expected_sha256=_sha256(_read_small_regular(path, maximum=2 * BODY_BYTES)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-qwen35-official-sampling-candidate")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--orchestrator-config", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--runner-artifact", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--one-shot-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.one_shot_state == args.output:
            raise CandidateFailure("candidate marker and report paths must differ")
        _assert_private_destination(args.one_shot_state)
        _assert_private_destination(args.output)
        preregistration_sha256 = load_preregistration(args.preregistration)
        fixtures, corpus_sha256 = load_corpus(args.corpus)
        if (
            corpus_sha256 != FROZEN_CORPUS_SHA256
            or _sha256(FALLBACK_SYSTEM_PROMPT.encode("utf-8")) != FROZEN_SYSTEM_PROMPT_SHA256
        ):
            raise CandidateFailure("candidate frozen corpus identity mismatch")
        model_spec = EXPECTED_PREREGISTRATION["model"]
        runner_spec = EXPECTED_PREREGISTRATION["runner"]
        model = hash_artifact(
            args.model_artifact,
            expected_size=model_spec["size_bytes"],
            expected_sha256=model_spec["sha256"],
        )
        runner = hash_artifact(
            args.runner_artifact,
            expected_size=runner_spec["size_bytes"],
            expected_sha256=runner_spec["sha256"],
        )
        harness = _safe_harness_identity()
        scorer_bundle = scorer_code_bundle_identity()
        bearer = read_private_bearer(args.orchestrator_config)
        marker = OneShotMarker(args.one_shot_state, preregistration_sha256)
        client = OfficialSamplingCandidateClient(
            args.base_url,
            bearer,
            before_first_generation=marker.arm,
        )
        warmup = Fixture("warmup", "en", "Reply with only this word: ready", 16, exact="ready")
        client.prepare_rendered_tokens((warmup, *fixtures))
        base_report = run_quality_gate(client, fixtures, corpus_sha256, warmups=WARMUPS)
        report = build_candidate_report(
            base_report,
            preregistration_sha256=preregistration_sha256,
            model=model,
            runner=runner,
            harness=harness,
            scorer_code_bundle=scorer_bundle,
            client=client,
        )
        _write_private_exclusive(args.output, report)
        print(json.dumps({"status": report["status"], "report_written": True}, separators=(",", ":")))
        return 0 if report["status"] == "official_sampling_candidate_pass" else 1
    except (CandidateFailure, OSError, ValueError, TypeError):
        print(json.dumps({
            "status": "official_sampling_candidate_error",
            "report_written": False,
            "raw_content_persisted": False,
        }, separators=(",", ":")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
