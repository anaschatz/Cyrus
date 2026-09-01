from __future__ import annotations

import hashlib
import math
import os
import secrets
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from .circuit import RestartCircuitBreaker
from .contracts import validate_generated_tool
from .engine import (
    EngineFailure,
    EngineProtocolFailure,
    EngineTimeout,
    GenerationEvent,
    PreparedPrompt,
)
from .errors import ApplianceError
from .gemma4 import (
    GemmaToolCallError,
    contains_gemma4_tool_marker,
    parse_gemma4_tool_call,
    render_gemma4_prompt,
    uses_gemma4_profile,
)
from .http_backend import BackendHTTP
from .limits import BACKEND_EVENT_BYTES, ENGINE_LOAD_SECONDS, IDLE_OR_EXIT_SECONDS
from .manifest import Manifest, resolve_local, verify_artifacts
from .private_temp import (
    PrivateRuntimeDirectory,
    private_temp_environment,
    trusted_windows_directory,
)
from .rendering import render_prompt
from .strict_json import canonical_bytes, canonical_text, parse_body, parse_text
from .windows_process import WindowsJobProcess

REQUIRED_FLAGS = (
    "--no-cache-prompt", "--cache-ram", "0", "--ctx-checkpoints", "0",
    "--no-cache-idle-slots", "--parallel", "1", "--no-cont-batching",
    "--no-webui", "--no-agent", "--no-mmproj", "--offline", "--log-disable",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LlamaServerEngine:
    def __init__(self, manifest: Manifest, breaker: RestartCircuitBreaker | None = None):
        self.manifest = manifest
        self.breaker = breaker or RestartCircuitBreaker()
        self.process: subprocess.Popen[bytes] | WindowsJobProcess | None = None
        self._runtime_temp: PrivateRuntimeDirectory | None = None
        self._process_tree_drained = True
        self.http: BackendHTTP | None = None
        self.secret = ""
        self.port = 0
        self._active_response = None
        self._active_lock = threading.Lock()
        self._qualification_metric_lock = threading.Lock()
        self._qualification_generation_serial = 0
        self._qualification_armed_serial: int | None = None
        self._qualification_armed_request_started: float | None = None
        self._qualification_inflight_serial: int | None = None
        self._qualification_completed_metric: dict[str, Any] | None = None
        self._qualification_clock = time.monotonic
        self.restart_recorded = False
        self.phase = "stopped"
        self.last_termination_elapsed = 0.0
        self.last_termination_exit_ok = True
        self.last_termination_kind = "none"
        self.last_termination_phase = "none"
        self.termination_serial = 0

    def start(self, deadline: float | None = None) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        if self.process is not None or self._runtime_temp is not None:
            self._terminate("stale")
        if not self.breaker.allow():
            raise EngineFailure("engine restart circuit is open")
        try:
            verify_artifacts(self.manifest)
            self.port = _free_port()
            self.secret = secrets.token_urlsafe(48)
            executable = resolve_local(self.manifest.root, self.manifest.backend["executable"]["path"], "backend.executable")
            model = resolve_local(self.manifest.root, self.manifest.model["artifacts"][0]["path"], "model.artifacts.0")
            command = [
                str(executable), "--host", "127.0.0.1", "--port", str(self.port),
                "--model", str(model), "--slots", "--no-jinja", "--cors-origins", "http://invalid.invalid",
                *REQUIRED_FLAGS,
            ]
            parameters = self.manifest.backend["parameters"]
            mapping = {
                "context_window": "--ctx-size", "gpu_layers": "--gpu-layers",
                "batch_size": "--batch-size", "ubatch_size": "--ubatch-size", "threads": "--threads",
            }
            for name in sorted(mapping):
                if name in parameters:
                    command.extend((mapping[name], str(parameters[name])))
            if "flash_attention" in parameters:
                command.extend(("--flash-attn", "on" if parameters["flash_attention"] else "off"))
            # The executable is absolute and its verified parent owns the
            # pinned native bundle. Never let an ambient user PATH or Windows
            # root satisfy a missing or substituted DLL.
            environment = {"LLAMA_API_KEY": self.secret}
            if os.name == "nt":
                windows = str(trusted_windows_directory())
                environment.update({"SYSTEMROOT": windows, "WINDIR": windows})
                runtime_temp = PrivateRuntimeDirectory(
                    self.manifest.root, prefix="cyrus-llama-runtime-"
                )
                self._runtime_temp = runtime_temp
                environment.update(private_temp_environment(runtime_temp.open()))
                self._process_tree_drained = False
                self.process = WindowsJobProcess(
                    command, environment, executable.parent
                )
            else:
                environment.update(
                    {
                        key: os.environ[key]
                        for key in ("TEMP", "TMP")
                        if key in os.environ
                    }
                )
                self.process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, env=environment,
                    cwd=str(executable.parent), close_fds=True,
                )
            self.http = BackendHTTP(self.port, self.secret)
            stop = min(deadline or time.monotonic() + ENGINE_LOAD_SECONDS, time.monotonic() + ENGINE_LOAD_SECONDS)
            while time.monotonic() < stop:
                if self.process.poll() is not None:
                    raise EngineFailure("llama-server exited during load")
                try:
                    state = self.http.request("GET", "/health", None, min(stop, time.monotonic() + 1.0))
                    if isinstance(state, dict) and state.get("status") == "ok":
                        self.restart_recorded = False
                        self.breaker.probe_succeeded()
                        self.phase = "idle"
                        return
                except EngineFailure:
                    pass
                time.sleep(0.05)
            raise EngineTimeout("llama-server load deadline exceeded")
        except Exception:
            self._terminate("startup")
            self.breaker.record_restart()
            raise

    def ready(self) -> bool:
        return self.process is not None and self.process.poll() is None and self.http is not None

    def arm_qualification_generation(self) -> int:
        """Arm exactly one content-free native timing observation.

        This method is intentionally not part of the public ``Engine``
        protocol.  Only the profile-qualification production subclass exposes
        it through an authenticated loopback endpoint.
        """

        if not self.ready():
            raise EngineFailure("llama-server is not ready")
        with self._qualification_metric_lock:
            if (
                self._qualification_armed_serial is not None
                or self._qualification_armed_request_started is not None
                or self._qualification_inflight_serial is not None
                or self._qualification_completed_metric is not None
            ):
                raise EngineProtocolFailure(
                    "a qualification generation metric is already pending"
                )
            self._qualification_generation_serial += 1
            self._qualification_armed_serial = self._qualification_generation_serial
            return self._qualification_generation_serial

    def bind_qualification_request_start(self, started: float) -> None:
        """Bind the gateway's pre-parse request start to the armed serial."""

        if (
            isinstance(started, bool)
            or not isinstance(started, (int, float))
            or not math.isfinite(float(started))
            or float(started) <= 0
        ):
            raise EngineProtocolFailure("invalid qualification request start")
        with self._qualification_metric_lock:
            if self._qualification_armed_serial is None:
                # Quality fixtures intentionally exercise the same private
                # gateway without arming performance telemetry.
                return
            if (
                self._qualification_armed_request_started is not None
                or self._qualification_inflight_serial is not None
                or self._qualification_completed_metric is not None
            ):
                raise EngineProtocolFailure(
                    "qualification request start is already bound"
                )
            self._qualification_armed_request_started = float(started)

    def consume_qualification_generation_metric(
        self, expected_serial: int
    ) -> dict[str, Any]:
        """Consume one exact generation metric; stale/reused/gapped reads fail."""

        if type(expected_serial) is not int or expected_serial <= 0:
            raise EngineProtocolFailure("invalid qualification generation serial")
        with self._qualification_metric_lock:
            metric = self._qualification_completed_metric
            if metric is not None:
                actual = metric.get("generation_serial")
                if actual != expected_serial:
                    raise EngineProtocolFailure(
                        "qualification generation serial does not match"
                    )
                self._qualification_completed_metric = None
                return dict(metric)
            if expected_serial in {
                self._qualification_armed_serial,
                self._qualification_inflight_serial,
            }:
                raise EngineProtocolFailure(
                    "qualification generation metric is not complete"
                )
            if expected_serial <= self._qualification_generation_serial:
                raise EngineProtocolFailure(
                    "qualification generation metric is stale or already consumed"
                )
            raise EngineProtocolFailure("qualification generation serial has a gap")

    def _begin_qualification_generation(self) -> tuple[int | None, float | None]:
        with self._qualification_metric_lock:
            serial = self._qualification_armed_serial
            if serial is not None:
                if (
                    self._qualification_armed_request_started is None
                    or self._qualification_inflight_serial is not None
                    or self._qualification_completed_metric is not None
                ):
                    raise EngineProtocolFailure(
                        "qualification generation metric state is inconsistent"
                    )
                request_started = self._qualification_armed_request_started
                self._qualification_armed_serial = None
                self._qualification_armed_request_started = None
                self._qualification_inflight_serial = serial
                return serial, request_started
            return None, None

    def _publish_qualification_generation_metric(
        self, serial: int, metric: dict[str, Any]
    ) -> None:
        with self._qualification_metric_lock:
            if (
                self._qualification_inflight_serial != serial
                or self._qualification_completed_metric is not None
            ):
                raise EngineProtocolFailure(
                    "qualification generation metric state is inconsistent"
                )
            self._qualification_inflight_serial = None
            self._qualification_completed_metric = dict(metric)

    @staticmethod
    def _build_qualification_generation_metric(
        serial: int,
        prompt: PreparedPrompt,
        accepted_token_ids: list[int],
        final: dict[str, Any],
        request_started: float,
        first_accepted: float | None,
        completed: float,
    ) -> dict[str, Any]:
        timings = final.get("timings")
        if not isinstance(timings, dict):
            raise EngineProtocolFailure("backend qualification timings are missing")
        prompt_n = timings.get("prompt_n")
        predicted_n = timings.get("predicted_n")
        prompt_ms = timings.get("prompt_ms")
        predicted_ms = timings.get("predicted_ms")
        if (
            type(prompt_n) is not int
            or prompt_n != len(prompt.token_ids)
            or type(predicted_n) is not int
            or predicted_n != len(accepted_token_ids)
            or predicted_n <= 0
            or isinstance(prompt_ms, bool)
            or not isinstance(prompt_ms, (int, float, Decimal))
            or not math.isfinite(float(prompt_ms))
            or float(prompt_ms) < 0
            or isinstance(predicted_ms, bool)
            or not isinstance(predicted_ms, (int, float, Decimal))
            or not math.isfinite(float(predicted_ms))
            or float(predicted_ms) <= 0
            or first_accepted is None
            or not math.isfinite(first_accepted)
            or not math.isfinite(request_started)
            or not math.isfinite(completed)
            or first_accepted <= request_started
            or completed < first_accepted
            or any(type(token) is not int or token < 0 for token in prompt.token_ids)
            or any(type(token) is not int or token < 0 for token in accepted_token_ids)
        ):
            raise EngineProtocolFailure("backend qualification timings are invalid")
        prompt_seconds = float(prompt_ms) / 1000.0
        decode_seconds = float(predicted_ms) / 1000.0
        decode_tps = predicted_n / decode_seconds
        request_ttft = first_accepted - request_started
        total_elapsed = completed - request_started
        if (
            not math.isfinite(decode_tps)
            or decode_tps <= 0
            or request_ttft + 0.05 < prompt_seconds
            or not math.isfinite(total_elapsed)
            or total_elapsed <= 0
            or total_elapsed + 0.05 < prompt_seconds + decode_seconds
        ):
            raise EngineProtocolFailure("backend qualification timings are invalid")
        return {
            "generation_serial": serial,
            "prompt_token_count": len(prompt.token_ids),
            "prompt_token_ids_sha256": hashlib.sha256(
                canonical_bytes(list(prompt.token_ids))
            ).hexdigest(),
            "accepted_token_count": predicted_n,
            "accepted_token_ids_sha256": hashlib.sha256(
                canonical_bytes(accepted_token_ids)
            ).hexdigest(),
            "request_ttft_seconds": request_ttft,
            "backend_prompt_seconds": prompt_seconds,
            "backend_decode_seconds": decode_seconds,
            "backend_decode_tokens_per_second": decode_tps,
            "request_to_engine_final_seconds": total_elapsed,
        }

    def prepare(self, request, deadline: float) -> PreparedPrompt:
        if not self.ready() or self.http is None:
            raise EngineFailure("llama-server is not ready")
        self.phase = "prepare"
        gemma4 = uses_gemma4_profile(self.manifest)
        rendered = render_gemma4_prompt(request) if gemma4 else render_prompt(request)
        response = self.http.request(
            "POST", "/tokenize", {
                "content": rendered,
                # The pinned Gemma renderer emits its authoritative <bos>
                # token itself. Generic legacy profiles retain their previous
                # backend-added-special-token behavior.
                "add_special": not gemma4,
                "parse_special": True,
                "with_pieces": False,
            }, deadline
        )
        tokens = response.get("tokens") if isinstance(response, dict) else None
        if not isinstance(tokens, list) or any(not isinstance(token, int) or isinstance(token, bool) for token in tokens):
            raise EngineProtocolFailure("tokenize returned invalid token IDs")
        self.phase = "prepared"
        return PreparedPrompt(tuple(tokens), rendered, request)

    def generate(self, prompt: PreparedPrompt, cancel: threading.Event, deadline: float) -> Iterable[GenerationEvent]:
        if not self.ready():
            raise EngineFailure("llama-server is not ready")
        qualification_serial, qualification_request_started = (
            self._begin_qualification_generation()
        )
        payload: dict[str, Any] = {
            # The immutable token sequence admitted at preflight is the exact
            # sequence consumed by generation; llama-server performs no second
            # template render or tokenization step.
            "prompt": list(prompt.token_ids),
            "n_predict": prompt.request.max_completion_tokens,
            "temperature": prompt.request.temperature,
            "top_p": prompt.request.top_p,
            "stop": list(prompt.request.stop),
            "stream": True,
            "cache_prompt": False,
            "slot_id": 0,
            "return_tokens": True,
        }
        if prompt.request.response_format.schema is not None:
            payload["json_schema"] = prompt.request.response_format.schema
        elif prompt.request.response_format.type == "json_object":
            payload["json_schema"] = {"type": "object"}
        elif prompt.request.tools and not uses_gemma4_profile(self.manifest) and (
            isinstance(prompt.request.tool_choice, tuple)
            or (prompt.request.tool_choice == "required" and len(prompt.request.tools) == 1)
        ):
            selected = prompt.request.tools[0]
            if isinstance(prompt.request.tool_choice, tuple):
                selected = next(tool for tool in prompt.request.tools if tool.name == prompt.request.tool_choice[1])
            payload["json_schema"] = {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "const": selected.name},
                    "arguments": selected.schema,
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/completion",
            data=canonical_bytes(payload),
            method="POST",
            headers={"Authorization": "Bearer " + self.secret, "Content-Type": "application/json"},
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EngineTimeout("generation deadline expired")
        self.phase = "prefill"
        try:
            response = urllib.request.urlopen(request, timeout=remaining)
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise EngineTimeout("generation request failed") from exc
        with self._active_lock:
            self._active_response = response
        buffered = prompt.request.response_format.type != "text" or bool(prompt.request.tools)
        content_parts: list[str] = []
        accepted_token_ids: list[int] = []
        final: dict[str, Any] | None = None
        qualification_first_accepted: float | None = None
        try:
            while True:
                raw_line = response.readline(BACKEND_EVENT_BYTES + 1)
                if not raw_line:
                    break
                if cancel.is_set() or time.monotonic() >= deadline:
                    raise EngineTimeout("generation cancelled or expired")
                if len(raw_line) > BACKEND_EVENT_BYTES:
                    raise EngineProtocolFailure("oversized backend event")
                line = raw_line.strip()
                if not line or line.startswith(b":"):
                    continue
                if not line.startswith(b"data: "):
                    raise EngineProtocolFailure("malformed backend SSE event")
                data = line[6:]
                try:
                    event = parse_body(data)
                except ApplianceError as exc:
                    raise EngineProtocolFailure("invalid backend SSE JSON") from exc
                if not isinstance(event, dict) or not isinstance(event.get("content", ""), str):
                    raise EngineProtocolFailure("invalid completion event")
                self.phase = "stream"
                text = event.get("content", "")
                token_ids = event.get("tokens")
                if not isinstance(token_ids, list) or any(type(token) is not int for token in token_ids):
                    raise EngineProtocolFailure("backend did not expose accepted token IDs")
                if (
                    qualification_serial is not None
                    and token_ids
                    and qualification_first_accepted is None
                ):
                    qualification_first_accepted = self._qualification_clock()
                accepted_token_ids.extend(token_ids)
                if text:
                    content_parts.append(text)
                    if not buffered:
                        yield GenerationEvent("text", text=text)
                if event.get("stop") is True:
                    final = event
                    break
        finally:
            response.close()
            with self._active_lock:
                self._active_response = None
        if final is None:
            raise EngineProtocolFailure("backend stream ended without a terminal event")
        self.phase = "final"
        count = final.get("tokens_predicted")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            timings = final.get("timings", {})
            count = timings.get("predicted_n") if isinstance(timings, dict) else None
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise EngineProtocolFailure("backend did not expose completion token accounting")
        if count != len(accepted_token_ids):
            raise EngineProtocolFailure("backend token count does not match accepted token IDs")
        stop_type = final.get("stop_type")
        reason = {"limit": "length", "eos": "stop", "word": "stop"}.get(stop_type)
        if reason is None:
            raise EngineProtocolFailure("unsupported backend finish reason")
        qualification_metric = None
        if qualification_serial is not None:
            qualification_metric = self._build_qualification_generation_metric(
                qualification_serial,
                prompt,
                accepted_token_ids,
                final,
                qualification_request_started,
                qualification_first_accepted,
                self._qualification_clock(),
            )
        combined = "".join(content_parts)
        if prompt.request.tools:
            if uses_gemma4_profile(self.manifest):
                choice = prompt.request.tool_choice
                has_marker = contains_gemma4_tool_marker(combined)
                if choice == "none" or (choice == "auto" and not has_marker):
                    if has_marker:
                        raise EngineProtocolFailure("backend emitted a forbidden Gemma tool call")
                    if qualification_metric is not None:
                        self._publish_qualification_generation_metric(
                            qualification_serial, qualification_metric
                        )
                    if combined:
                        yield GenerationEvent("text", text=combined)
                    yield GenerationEvent("final", finish_reason=reason, completion_tokens=count)
                    return
                try:
                    name, raw_arguments = parse_gemma4_tool_call(combined)
                except GemmaToolCallError as exc:
                    raise EngineProtocolFailure("backend Gemma tool output is invalid") from exc
                arguments = validate_generated_tool(name, raw_arguments, prompt.request)
                if qualification_metric is not None:
                    self._publish_qualification_generation_metric(
                        qualification_serial, qualification_metric
                    )
                yield GenerationEvent("tool", tool_name=name, tool_arguments=arguments)
                yield GenerationEvent("final", finish_reason="tool_calls", completion_tokens=count)
                return
            try:
                value = parse_text(combined, nested=True)
            except ApplianceError as exc:
                if prompt.request.tool_choice == "auto" and combined:
                    if qualification_metric is not None:
                        self._publish_qualification_generation_metric(
                            qualification_serial, qualification_metric
                        )
                    yield GenerationEvent("text", text=combined)
                    yield GenerationEvent("final", finish_reason=reason, completion_tokens=count)
                    return
                raise EngineProtocolFailure("backend tool output is invalid") from exc
            if not isinstance(value, dict) or set(value) != {"name", "arguments"} or not isinstance(value["name"], str) or not isinstance(value["arguments"], dict):
                if prompt.request.tool_choice == "auto" and combined:
                    if qualification_metric is not None:
                        self._publish_qualification_generation_metric(
                            qualification_serial, qualification_metric
                        )
                    yield GenerationEvent("text", text=combined)
                    yield GenerationEvent("final", finish_reason=reason, completion_tokens=count)
                    return
                raise EngineProtocolFailure("backend tool output is invalid")
            arguments = validate_generated_tool(value["name"], canonical_text(value["arguments"]), prompt.request)
            if qualification_metric is not None:
                self._publish_qualification_generation_metric(
                    qualification_serial, qualification_metric
                )
            yield GenerationEvent("tool", tool_name=value["name"], tool_arguments=arguments)
            yield GenerationEvent("final", finish_reason="tool_calls", completion_tokens=count)
            return
        if buffered and combined:
            yield GenerationEvent("text", text=combined)
        if qualification_metric is not None:
            self._publish_qualification_generation_metric(
                qualification_serial, qualification_metric
            )
        yield GenerationEvent("final", finish_reason=reason, completion_tokens=count)

    def cleanup(self, deadline: float) -> None:
        if self.http is None:
            raise EngineFailure("backend is unavailable during cleanup")
        self.http.request("POST", "/slots/0?action=erase", {}, deadline)
        slots = self.http.request("GET", "/slots", None, deadline)
        if not isinstance(slots, list) or len(slots) != 1 or not isinstance(slots[0], dict) or slots[0].get("is_processing") is not False:
            raise EngineProtocolFailure("slot did not become idle after erase")
        self.phase = "idle"

    def cancel(self) -> None:
        cancellation_deadline = time.monotonic() + IDLE_OR_EXIT_SECONDS
        with self._active_lock:
            response = self._active_response
        if response is not None:
            try:
                response.close()
            except OSError:
                pass
        # Closing the response is cooperative cancellation.  Fail closed if the
        # sole slot does not become idle within one second: the old process is
        # then terminated, satisfying the two-second idle-or-exited boundary.
        http = self.http
        if http is not None and self.ready():
            stop = time.monotonic() + 1.0
            while time.monotonic() < stop:
                try:
                    slots = http.request("GET", "/slots", None, min(stop, time.monotonic() + 0.1))
                    if isinstance(slots, list) and len(slots) == 1 and isinstance(slots[0], dict) and slots[0].get("is_processing") is False:
                        return
                except EngineFailure:
                    break
            self.breaker.record_restart()
            self.restart_recorded = True
            self._terminate(
                "cancel",
                timeout=max(0.0, cancellation_deadline - time.monotonic()),
            )

    def inject_cancel(self) -> dict[str, Any]:
        """Force an isolated qualification cancellation with bounded exit."""

        process = self.process
        if process is None or process.poll() is not None:
            raise EngineFailure("llama-server is unavailable")
        pid = process.pid
        phase = self.phase
        with self._active_lock:
            response = self._active_response
        if response is not None:
            try:
                response.close()
            except OSError:
                pass
        self.breaker.record_restart()
        self.restart_recorded = True
        self._terminate("cancel")
        return {
            "pid": pid,
            "phase": phase,
            "elapsed_seconds": self.last_termination_elapsed,
            "exit_ok": self.last_termination_exit_ok,
        }

    def inject_crash(self) -> dict[str, Any]:
        """Kill the managed backend for an isolated qualification case."""

        process = self.process
        if process is None or process.poll() is not None:
            raise EngineFailure("llama-server is unavailable")
        pid = process.pid
        phase = self.phase
        self._terminate("crash", kill_first=True)
        return {
            "pid": pid,
            "phase": phase,
            "elapsed_seconds": self.last_termination_elapsed,
            "exit_ok": self.last_termination_exit_ok,
        }

    def runtime_snapshot(self) -> dict[str, Any]:
        process = self.process
        return {
            "pid": process.pid if process is not None and process.poll() is None else None,
            "phase": self.phase,
            "termination_serial": self.termination_serial,
            "last_termination_kind": self.last_termination_kind,
            "last_termination_phase": self.last_termination_phase,
            "last_termination_elapsed": self.last_termination_elapsed,
            "last_termination_exit_ok": self.last_termination_exit_ok,
        }

    def close(self) -> None:
        self.cancel()
        self._terminate("close")

    def _terminate(
        self,
        kind: str,
        timeout: float = IDLE_OR_EXIT_SECONDS,
        *,
        kill_first: bool = False,
    ) -> None:
        process = self.process
        self.http = None
        if process is None:
            if self._runtime_temp is not None and not self._process_tree_drained:
                raise EngineFailure("llama-server process-tree teardown is unproven")
            self._close_runtime_temp()
            self.phase = "stopped"
            return
        if process.poll() is not None:
            self._drain_process_tree(process)
            self.process = None
            self._close_runtime_temp()
            self.phase = "stopped"
            return
        started = time.monotonic()
        terminated_phase = self.phase
        deadline = time.monotonic() + max(0.0, timeout)
        if kill_first:
            process.kill()
        else:
            process.terminate()
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    pass
        self._drain_process_tree(process)
        self.process = None
        self._close_runtime_temp()
        self.last_termination_elapsed = time.monotonic() - started
        self.last_termination_exit_ok = process.poll() is not None and (
            self.last_termination_elapsed <= IDLE_OR_EXIT_SECONDS
        )
        self.last_termination_kind = kind
        self.last_termination_phase = terminated_phase
        self.termination_serial += 1
        self.phase = "stopped"

    def _drain_process_tree(
        self, process: subprocess.Popen[bytes] | WindowsJobProcess
    ) -> None:
        close_tree = getattr(process, "close_tree", None)
        if callable(close_tree):
            close_tree(timeout=10.0)
        self._process_tree_drained = True

    def _close_runtime_temp(self) -> None:
        if not self._process_tree_drained:
            return
        owner = self._runtime_temp
        self._runtime_temp = None
        if owner is not None:
            owner.close()
