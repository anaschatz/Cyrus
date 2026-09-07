"""Content-free native metrics client and Windows working-set probes.

The profile runner imports this module so its orchestration stays below the
repository module-size ratchet.  Every HTTP endpoint is authenticated and
loopback-only.  Prompt text, output text, token IDs, and secrets are never
returned from these helpers or written to disk.
"""

from __future__ import annotations

import ctypes
import http.client
import math
import os
import re
import statistics
import time
from ctypes import wintypes
from decimal import Decimal
from typing import Any, Callable
from urllib.parse import urlparse

from .errors import ApplianceError
from .strict_json import canonical_bytes, parse_body


_SHA = re.compile(r"^[0-9a-f]{64}$", re.ASCII)


class PrimaryRunnerFailure(RuntimeError):
    """A content-free runner failure safe for a diagnostic category."""


def _parse_response(raw: bytes) -> Any:
    try:
        return parse_body(raw)
    except (ApplianceError, RecursionError) as exc:
        raise PrimaryRunnerFailure("invalid gateway JSON") from exc


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise PrimaryRunnerFailure(f"invalid {name} shape")
    return value


def _number(value: Any, name: str, *, positive: bool = True) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, Decimal))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0)
    ):
        raise PrimaryRunnerFailure(f"invalid {name}")
    return float(value)


def _checked_usage(value: Any, expected_prompt_tokens: int) -> int:
    usage = _exact(
        value,
        {"prompt_tokens", "completion_tokens", "total_tokens"},
        "usage",
    )
    prompt = usage["prompt_tokens"]
    completion = usage["completion_tokens"]
    total = usage["total_tokens"]
    if (
        type(prompt) is not int
        or prompt != expected_prompt_tokens
        or type(completion) is not int
        or not 1 <= completion <= 128
        or type(total) is not int
        or total != prompt + completion
    ):
        raise PrimaryRunnerFailure("invalid token accounting")
    return completion


class PrimaryQualificationClient:
    """Strict content-discarding client for the private qualification gateway."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port is None
            or parsed.path.rstrip("/") != "/v1"
            or parsed.params
            or parsed.query
            or parsed.fragment
            or len(api_key.encode("utf-8")) < 32
        ):
            raise PrimaryRunnerFailure("invalid qualification endpoint")
        self.host = parsed.hostname
        self.port = parsed.port
        self.api_key = api_key
        self.clock = clock
        self._last_generation_serial = 0

    def _headers(self, body: bytes | None = None) -> dict[str, str]:
        result = {
            "Authorization": "Bearer " + self.api_key,
            "Host": f"127.0.0.1:{self.port}",
        }
        if body is not None:
            result.update(
                {
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                }
            )
        return result

    def _connection(self, timeout: float = 300.0) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def json_request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        body = None if payload is None else canonical_bytes(payload)
        connection = self._connection()
        try:
            connection.request(
                method,
                path,
                body=body,
                headers=self._headers(body),
            )
            response = connection.getresponse()
            raw = response.read(2_097_153)
            if len(raw) > 2_097_152:
                raise PrimaryRunnerFailure("oversized gateway response")
            value = _parse_response(raw) if raw else {}
            if not isinstance(value, dict):
                raise PrimaryRunnerFailure("non-object gateway response")
            return response.status, value
        finally:
            connection.close()

    def runtime(self) -> dict[str, Any]:
        status, value = self.json_request("GET", "/jarvis/qualification/runtime")
        if status != 200:
            raise PrimaryRunnerFailure("runtime endpoint failed")
        runtime = _exact(
            value,
            {"ready", "busy", "gateway_pid", "engine", "terminations"},
            "runtime",
        )
        if (
            type(runtime["ready"]) is not bool
            or type(runtime["busy"]) is not bool
            or type(runtime["gateway_pid"]) is not int
            or runtime["gateway_pid"] <= 0
            or not isinstance(runtime["engine"], dict)
            or not isinstance(runtime["terminations"], list)
            or len(runtime["terminations"]) > 64
        ):
            raise PrimaryRunnerFailure("runtime endpoint is invalid")
        for item in runtime["terminations"]:
            termination = _exact(
                item,
                {"pid", "kind", "phase", "elapsed_seconds", "exit_ok"},
                "termination",
            )
            if (
                type(termination["pid"]) is not int
                or termination["pid"] <= 0
                or not isinstance(termination["kind"], str)
                or not isinstance(termination["phase"], str)
                or termination["exit_ok"] is not True
            ):
                raise PrimaryRunnerFailure("runtime termination is invalid")
            _number(termination["elapsed_seconds"], "termination duration", positive=False)
        return runtime

    def wait_ready(
        self,
        *,
        timeout: float = 600.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        stop = self.clock() + timeout
        while self.clock() < stop:
            try:
                value = self.runtime()
                engine = value["engine"]
                if (
                    value["ready"] is True
                    and value["busy"] is False
                    and type(engine.get("pid")) is int
                    and engine["pid"] > 0
                ):
                    return value
            except (OSError, PrimaryRunnerFailure):
                pass
            sleep(0.05)
        raise PrimaryRunnerFailure("qualification gateway did not recover")

    def tokenize(self, request: dict[str, Any]) -> dict[str, Any]:
        status, value = self.json_request(
            "POST", "/jarvis/qualification/tokenize", request
        )
        if status != 200:
            raise PrimaryRunnerFailure("native tokenization failed")
        item = _exact(value, {"token_count", "token_ids_sha256"}, "tokenization")
        if (
            type(item["token_count"]) is not int
            or not 1 <= item["token_count"] <= 32_640
            or not isinstance(item["token_ids_sha256"], str)
            or _SHA.fullmatch(item["token_ids_sha256"]) is None
        ):
            raise PrimaryRunnerFailure("native tokenization is invalid")
        return item

    def _arm_generation(self) -> int:
        status, value = self.json_request(
            "POST", "/jarvis/qualification/generation/arm", {}
        )
        if status != 200:
            raise PrimaryRunnerFailure("native generation metric arm failed")
        item = _exact(value, {"generation_serial"}, "generation arm")
        serial = item["generation_serial"]
        if type(serial) is not int or serial != self._last_generation_serial + 1:
            raise PrimaryRunnerFailure("native generation serial is invalid")
        return serial

    def _consume_generation_metric(self, expected_serial: int) -> dict[str, Any]:
        status, value = self.json_request(
            "POST",
            "/jarvis/qualification/generation/metric",
            {"expected_serial": expected_serial},
        )
        if status != 200:
            raise PrimaryRunnerFailure("native generation metric consume failed")
        metric = _exact(
            value,
            {
                "generation_serial",
                "prompt_token_count",
                "prompt_token_ids_sha256",
                "accepted_token_count",
                "accepted_token_ids_sha256",
                "request_ttft_seconds",
                "backend_prompt_seconds",
                "backend_decode_seconds",
                "backend_decode_tokens_per_second",
                "request_to_engine_final_seconds",
            },
            "generation metric",
        )
        if (
            metric["generation_serial"] != expected_serial
            or type(metric["prompt_token_count"]) is not int
            or not 1 <= metric["prompt_token_count"] <= 32_640
            or type(metric["accepted_token_count"]) is not int
            or not 1 <= metric["accepted_token_count"] <= 128
            or any(
                not isinstance(metric[name], str)
                or _SHA.fullmatch(metric[name]) is None
                for name in (
                    "prompt_token_ids_sha256",
                    "accepted_token_ids_sha256",
                )
            )
        ):
            raise PrimaryRunnerFailure("native generation metric is invalid")
        ttft = _number(metric["request_ttft_seconds"], "request TTFT")
        prompt_seconds = _number(
            metric["backend_prompt_seconds"], "backend prompt duration", positive=False
        )
        decode_seconds = _number(
            metric["backend_decode_seconds"], "backend decode duration"
        )
        decode_tps = _number(
            metric["backend_decode_tokens_per_second"], "backend decode throughput"
        )
        elapsed = _number(
            metric["request_to_engine_final_seconds"],
            "request-to-engine-final duration",
        )
        if (
            prompt_seconds < 0
            or elapsed < ttft
            or ttft + 0.05 < prompt_seconds
            or elapsed + 0.05 < prompt_seconds + decode_seconds
            or not math.isclose(
                decode_tps,
                metric["accepted_token_count"] / decode_seconds,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise PrimaryRunnerFailure("native generation metric is inconsistent")
        self._last_generation_serial = expected_serial
        return {
            **metric,
            "request_ttft_seconds": ttft,
            "backend_prompt_seconds": prompt_seconds,
            "backend_decode_seconds": decode_seconds,
            "backend_decode_tokens_per_second": decode_tps,
            "request_to_engine_final_seconds": elapsed,
        }

    def observe(
        self, request: dict[str, Any], expected_tokenization: dict[str, Any]
    ) -> dict[str, Any]:
        tokenization = _exact(
            expected_tokenization,
            {"token_count", "token_ids_sha256"},
            "expected tokenization",
        )
        serial = self._arm_generation()
        public = (
            self._observe_stream(request, tokenization["token_count"])
            if request.get("stream") is True
            else self._observe_buffered(request, tokenization["token_count"])
        )
        metric = self._consume_generation_metric(serial)
        if (
            metric["prompt_token_count"] != tokenization["token_count"]
            or metric["prompt_token_ids_sha256"] != tokenization["token_ids_sha256"]
            or metric["accepted_token_count"] != public["completion_tokens"]
            or metric["request_to_engine_final_seconds"]
            > public["end_to_end_seconds"] + 0.05
        ):
            raise PrimaryRunnerFailure("public and native generation evidence diverged")
        return {
            "generation_serial": serial,
            "ttft_seconds": metric["request_ttft_seconds"],
            "end_to_end_seconds": public["end_to_end_seconds"],
            "accepted_decode_tokens": metric["accepted_token_count"],
            "accepted_token_ids_sha256": metric["accepted_token_ids_sha256"],
            "backend_prompt_seconds": metric["backend_prompt_seconds"],
            "backend_decode_seconds": metric["backend_decode_seconds"],
            "decode_tokens_per_second": metric[
                "backend_decode_tokens_per_second"
            ],
            "request_to_engine_final_seconds": metric[
                "request_to_engine_final_seconds"
            ],
            "finish_reason": public["finish_reason"],
        }

    def _observe_buffered(
        self, request: dict[str, Any], expected_prompt_tokens: int
    ) -> dict[str, Any]:
        body = canonical_bytes(request)
        connection = self._connection()
        started = self.clock()
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers=self._headers(body),
            )
            response = connection.getresponse()
            raw = response.read(2_097_153)
            if response.status != 200 or len(raw) > 2_097_152:
                raise PrimaryRunnerFailure("buffered completion failed")
            value = _parse_response(raw)
        finally:
            ended = self.clock()
            connection.close()
        if not isinstance(value, dict):
            raise PrimaryRunnerFailure("buffered completion is invalid")
        try:
            choices = value["choices"]
            choice = choices[0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PrimaryRunnerFailure("buffered completion shape failed") from exc
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choice, dict)
            or not isinstance(message, dict)
            or not isinstance(message.get("content"), str)
            or not message["content"]
            or choice.get("finish_reason") not in ("stop", "length")
        ):
            raise PrimaryRunnerFailure("buffered completion contract failed")
        completion = _checked_usage(value.get("usage"), expected_prompt_tokens)
        return self._public_observation(started, ended, completion, choice["finish_reason"])

    def _observe_stream(
        self, request: dict[str, Any], expected_prompt_tokens: int
    ) -> dict[str, Any]:
        body = canonical_bytes(request)
        connection = self._connection()
        started = self.clock()
        terminal: str | None = None
        usage: dict[str, Any] | None = None
        done = False
        output_seen = False
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
                raise PrimaryRunnerFailure("stream completion failed")
            while True:
                raw = response.readline(524_289)
                if not raw:
                    break
                if len(raw) > 524_288 or not raw.endswith(b"\n"):
                    raise PrimaryRunnerFailure("invalid stream frame size")
                if not raw.startswith(b"data: "):
                    if raw.strip():
                        raise PrimaryRunnerFailure("invalid stream framing")
                    continue
                payload = raw[6:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                event = _parse_response(payload)
                if not isinstance(event, dict):
                    raise PrimaryRunnerFailure("invalid stream event")
                choices = event.get("choices")
                if choices == []:
                    candidate = event.get("usage")
                    if terminal is None or usage is not None or not isinstance(candidate, dict):
                        raise PrimaryRunnerFailure("invalid stream usage ordering")
                    usage = candidate
                    continue
                if (
                    terminal is not None
                    or not isinstance(choices, list)
                    or len(choices) != 1
                    or not isinstance(choices[0], dict)
                    or not isinstance(choices[0].get("delta"), dict)
                ):
                    raise PrimaryRunnerFailure("invalid stream choice")
                choice = choices[0]
                delta = choice["delta"]
                content = delta.get("content")
                tool_calls = delta.get("tool_calls")
                if (
                    (content is not None and not isinstance(content, str))
                    or (tool_calls is not None and (
                        not isinstance(tool_calls, list)
                        or any(not isinstance(call, dict) for call in tool_calls)
                    ))
                ):
                    raise PrimaryRunnerFailure("invalid stream delta")
                if (isinstance(content, str) and bool(content)) or (
                    isinstance(tool_calls, list) and bool(tool_calls)
                ):
                    output_seen = True
                reason = choice.get("finish_reason")
                if reason is not None:
                    if reason not in ("stop", "length", "tool_calls"):
                        raise PrimaryRunnerFailure("invalid stream finish reason")
                    terminal = reason
        finally:
            ended = self.clock()
            connection.close()
        if (
            not output_seen
            or not done
            or terminal not in {"stop", "length", "tool_calls"}
        ):
            raise PrimaryRunnerFailure("incomplete stream completion")
        completion = _checked_usage(usage, expected_prompt_tokens)
        return self._public_observation(started, ended, completion, terminal)

    @staticmethod
    def _public_observation(
        started: float,
        ended: float,
        completion_tokens: int,
        finish_reason: str,
    ) -> dict[str, Any]:
        elapsed = ended - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise PrimaryRunnerFailure("invalid public request timing")
        return {
            "end_to_end_seconds": elapsed,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
        }

    def restart(self, previous_pid: int) -> tuple[dict[str, Any], float]:
        started = self.clock()
        status, value = self.json_request(
            "POST", "/jarvis/qualification/crash", {}
        )
        result = _exact(
            value,
            {"pid", "phase", "elapsed_seconds", "exit_ok"},
            "restart injection",
        )
        if (
            status != 200
            or result["pid"] != previous_pid
            or not isinstance(result["phase"], str)
            or result["exit_ok"] is not True
        ):
            raise PrimaryRunnerFailure("managed full restart injection failed")
        _number(result["elapsed_seconds"], "restart exit duration", positive=False)
        runtime = self.wait_ready()
        engine = runtime["engine"]
        if engine.get("pid") == previous_pid:
            raise PrimaryRunnerFailure("managed full restart reused its process")
        elapsed = self.clock() - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise PrimaryRunnerFailure("managed full restart timing is invalid")
        self._last_generation_serial = 0
        return runtime, elapsed


def windows_process_working_set_bytes(pid: int) -> int:
    """Read one Windows process working-set sample with an exact PID."""

    if os.name != "nt" or type(pid) is not int or pid <= 0:
        raise PrimaryRunnerFailure("native Windows process metrics are unavailable")

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    process_query_information = 0x0400
    process_vm_read = 0x0010
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(
        process_query_information | process_vm_read, False, pid
    )
    if not handle:
        raise PrimaryRunnerFailure("managed process metrics could not be opened")
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    try:
        ok = psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        value = int(counters.WorkingSetSize)
        if not ok or value <= 0:
            raise PrimaryRunnerFailure("managed process metrics are invalid")
        return value
    finally:
        kernel32.CloseHandle(handle)


def combined_working_set_bytes(gateway_pid: int, engine_pid: int) -> int:
    """Sample the fixed qualification gateway and active llama-server together."""

    if gateway_pid == engine_pid:
        raise PrimaryRunnerFailure("qualification process identities overlap")
    gateway = windows_process_working_set_bytes(gateway_pid)
    engine = windows_process_working_set_bytes(engine_pid)
    total = gateway + engine
    if total <= gateway or total <= engine:
        raise PrimaryRunnerFailure("combined process metrics overflowed")
    return total


def evaluate_memory_epochs(
    epochs: list[list[int]], *, minimum_samples: int = 12
) -> tuple[int, int, int, float, float, list[int], bool]:
    """Evaluate global and within-restart-epoch working-set growth."""

    if (
        not epochs
        or type(minimum_samples) is not int
        or minimum_samples < 2
        or any(
            len(epoch) < minimum_samples
            or any(type(value) is not int or value <= 0 for value in epoch)
            for epoch in epochs
        )
    ):
        raise PrimaryRunnerFailure("soak memory samples are incomplete")
    def robust_growth(samples: list[int]) -> tuple[int, int, float]:
        baseline = round(statistics.median(samples[:3]))
        final = round(statistics.median(samples[-3:]))
        slopes = [
            (samples[right] - samples[left]) / (right - left)
            for left in range(len(samples))
            for right in range(left + 1, len(samples))
        ]
        projected = max(
            0.0, statistics.median(slopes) * (len(samples) - 1)
        )
        end_growth = max(0, final - baseline)
        return baseline, final, max(end_growth, projected) / baseline

    flattened = [value for epoch in epochs for value in epoch]
    baseline, final, global_growth = robust_growth(flattened)
    peak = max(flattened)
    epoch_growth = [robust_growth(epoch)[2] for epoch in epochs]
    maximum_epoch_growth = max(epoch_growth)
    detected = global_growth > 0.05 or maximum_epoch_growth > 0.05
    return (
        baseline,
        final,
        peak,
        global_growth,
        maximum_epoch_growth,
        [len(epoch) for epoch in epochs],
        detected,
    )
