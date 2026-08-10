from __future__ import annotations

import os
import secrets
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from typing import Any

from .circuit import RestartCircuitBreaker
from .contracts import validate_generated_tool
from .engine import EngineFailure, EngineProtocolFailure, EngineTimeout, GenerationEvent, PreparedPrompt
from .errors import ApplianceError
from .http_backend import BackendHTTP
from .limits import BACKEND_EVENT_BYTES, ENGINE_LOAD_SECONDS, IDLE_OR_EXIT_SECONDS
from .manifest import Manifest, resolve_local, verify_artifacts
from .rendering import render_prompt
from .strict_json import canonical_bytes, canonical_text, parse_body, parse_text

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
        self.process: subprocess.Popen[bytes] | None = None
        self.http: BackendHTTP | None = None
        self.secret = ""
        self.port = 0
        self._active_response = None
        self._active_lock = threading.Lock()
        self.restart_recorded = False

    def start(self, deadline: float | None = None) -> None:
        if self.process is not None and self.process.poll() is None:
            return
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
            environment = {
                key: os.environ[key]
                for key in ("SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP")
                if key in os.environ
            }
            environment["LLAMA_API_KEY"] = self.secret
            self.process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env=environment, close_fds=True,
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
                        return
                except EngineFailure:
                    pass
                time.sleep(0.05)
            raise EngineTimeout("llama-server load deadline exceeded")
        except Exception:
            self._terminate()
            self.breaker.record_restart()
            raise

    def ready(self) -> bool:
        return self.process is not None and self.process.poll() is None and self.http is not None

    def prepare(self, request, deadline: float) -> PreparedPrompt:
        if not self.ready() or self.http is None:
            raise EngineFailure("llama-server is not ready")
        rendered = render_prompt(request)
        response = self.http.request(
            "POST", "/tokenize", {"content": rendered, "add_special": True, "parse_special": True, "with_pieces": False}, deadline
        )
        tokens = response.get("tokens") if isinstance(response, dict) else None
        if not isinstance(tokens, list) or any(not isinstance(token, int) or isinstance(token, bool) for token in tokens):
            raise EngineProtocolFailure("tokenize returned invalid token IDs")
        return PreparedPrompt(tuple(tokens), rendered, request)

    def generate(self, prompt: PreparedPrompt, cancel: threading.Event, deadline: float) -> Iterable[GenerationEvent]:
        if not self.ready():
            raise EngineFailure("llama-server is not ready")
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
        elif prompt.request.tools and (
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
                text = event.get("content", "")
                token_ids = event.get("tokens")
                if not isinstance(token_ids, list) or any(type(token) is not int for token in token_ids):
                    raise EngineProtocolFailure("backend did not expose accepted token IDs")
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
        combined = "".join(content_parts)
        if prompt.request.tools:
            try:
                value = parse_text(combined, nested=True)
            except ApplianceError as exc:
                if prompt.request.tool_choice == "auto" and combined:
                    yield GenerationEvent("text", text=combined)
                    yield GenerationEvent("final", finish_reason=reason, completion_tokens=count)
                    return
                raise EngineProtocolFailure("backend tool output is invalid") from exc
            if not isinstance(value, dict) or set(value) != {"name", "arguments"} or not isinstance(value["name"], str) or not isinstance(value["arguments"], dict):
                if prompt.request.tool_choice == "auto" and combined:
                    yield GenerationEvent("text", text=combined)
                    yield GenerationEvent("final", finish_reason=reason, completion_tokens=count)
                    return
                raise EngineProtocolFailure("backend tool output is invalid")
            arguments = validate_generated_tool(value["name"], canonical_text(value["arguments"]), prompt.request)
            yield GenerationEvent("tool", tool_name=value["name"], tool_arguments=arguments)
            yield GenerationEvent("final", finish_reason="tool_calls", completion_tokens=count)
            return
        if buffered and combined:
            yield GenerationEvent("text", text=combined)
        yield GenerationEvent("final", finish_reason=reason, completion_tokens=count)

    def cleanup(self, deadline: float) -> None:
        if self.http is None:
            raise EngineFailure("backend is unavailable during cleanup")
        self.http.request("POST", "/slots/0?action=erase", {}, deadline)
        slots = self.http.request("GET", "/slots", None, deadline)
        if not isinstance(slots, list) or len(slots) != 1 or not isinstance(slots[0], dict) or slots[0].get("is_processing") is not False:
            raise EngineProtocolFailure("slot did not become idle after erase")

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
            self._terminate(max(0.0, cancellation_deadline - time.monotonic()))

    def close(self) -> None:
        self.cancel()
        self._terminate()

    def _terminate(self, timeout: float = IDLE_OR_EXIT_SECONDS) -> None:
        process = self.process
        self.process = None
        self.http = None
        if process is None or process.poll() is not None:
            return
        deadline = time.monotonic() + max(0.0, timeout)
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
