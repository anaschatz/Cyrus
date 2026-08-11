from __future__ import annotations

import os
from pathlib import PurePosixPath
import select
import subprocess
import threading
import time
from collections.abc import Iterable
from typing import Any

from .contracts import validate_generated_tool
from .circuit import RestartCircuitBreaker
from .engine import EngineFailure, EngineProtocolFailure, GenerationEvent, PreparedPrompt
from .errors import ApplianceError
from .limits import BACKEND_EVENT_BYTES, ENGINE_LOAD_SECONDS, IDLE_OR_EXIT_SECONDS
from .manifest import Manifest, resolve_local, verify_artifacts
from .rendering import render_mlx_qwen3_prompt
from .strict_json import canonical_bytes, canonical_text, parse_body, parse_text


class MlxWorkerEngine:
    """Owner-only pipe supervisor for the pinned MLX-LM/Outlines worker."""

    def __init__(self, manifest: Manifest, breaker: RestartCircuitBreaker | None = None):
        self.manifest = manifest
        self.breaker = breaker or RestartCircuitBreaker()
        self.process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self.restart_recorded = False
        self.phase = "stopped"
        self.last_termination_elapsed = 0.0
        self.last_termination_exit_ok = True
        self.last_termination_kind = "none"
        self.last_termination_phase = "none"
        self.termination_serial = 0

    def start(self, deadline: float | None = None) -> None:
        if not self.breaker.allow():
            raise EngineFailure("engine restart circuit is open")
        try:
            verify_artifacts(self.manifest)
            executable = resolve_local(self.manifest.root, self.manifest.python["executable"]["path"], "python.executable")
            worker = resolve_local(self.manifest.root, self.manifest.backend["worker"]["path"], "backend.worker")
            model_paths = [item["path"] for item in self.manifest.model["artifacts"]]
            common = PurePosixPath(os.path.commonpath(model_paths))
            model_dir = resolve_local(self.manifest.root, common.as_posix(), "model.artifacts", must_exist=False)
            if not model_dir.is_dir():
                raise EngineFailure("MLX model root is invalid")
            environment = {
                key: os.environ[key]
                for key in ("PATH", "TMPDIR", "LANG", "LC_ALL")
                if key in os.environ
            }
            environment["JARVIS_MLX_MODEL_DIR"] = str(model_dir)
            environment["PYTHONNOUSERSITE"] = "1"
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            self.process = subprocess.Popen(
                [str(executable), "-I", str(worker)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=environment, cwd=str(self.manifest.root), close_fds=True,
            )
            stop = min(deadline or time.monotonic() + ENGINE_LOAD_SECONDS, time.monotonic() + ENGINE_LOAD_SECONDS)
            response = self._exchange({"op": "init"}, stop)
        except Exception:
            self.breaker.record_restart()
            self._terminate("startup")
            raise
        if response != {"event": "ready"}:
            self.breaker.record_restart()
            self._terminate("startup")
            raise EngineProtocolFailure("MLX worker failed its ready handshake")
        self.breaker.probe_succeeded()
        self.restart_recorded = False
        self.phase = "idle"

    def ready(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def prepare(self, request, deadline: float) -> PreparedPrompt:
        self.phase = "prepare"
        prompt = render_mlx_qwen3_prompt(request)
        result = self._exchange({"op": "prepare", "prompt": prompt}, deadline)
        if not isinstance(result, dict) or set(result) != {"event", "token_ids"} or result.get("event") != "prepared":
            raise EngineProtocolFailure("MLX worker returned an invalid prepare event")
        tokens = result.get("token_ids") if isinstance(result, dict) else None
        if not isinstance(tokens, list) or any(not isinstance(token, int) or isinstance(token, bool) for token in tokens):
            raise EngineProtocolFailure("MLX worker returned invalid token IDs")
        self.phase = "prepared"
        return PreparedPrompt(tuple(tokens), prompt, request)

    def generate(self, prompt: PreparedPrompt, cancel: threading.Event, deadline: float) -> Iterable[GenerationEvent]:
        schema = prompt.request.response_format.schema
        if prompt.request.response_format.type == "json_object":
            schema = {"type": "object"}
        if prompt.request.tools and (
            isinstance(prompt.request.tool_choice, tuple)
            or (prompt.request.tool_choice == "required" and len(prompt.request.tools) == 1)
        ):
            selected = prompt.request.tools[0]
            if isinstance(prompt.request.tool_choice, tuple):
                selected = next(tool for tool in prompt.request.tools if tool.name == prompt.request.tool_choice[1])
            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "const": selected.name},
                    "arguments": selected.schema,
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        command = {
            "op": "generate",
            "token_ids": list(prompt.token_ids),
            "max_tokens": prompt.request.max_completion_tokens,
            "temperature": prompt.request.temperature,
            "top_p": prompt.request.top_p,
            "stop": list(prompt.request.stop),
            "schema": schema,
            "buffered": prompt.request.response_format.type != "text" or bool(prompt.request.tools) or bool(prompt.request.stop),
        }
        self.phase = "prefill"
        self._send(command)
        parts: list[str] = []
        while not cancel.is_set() and time.monotonic() < deadline:
            event = self._read(deadline)
            kind = event.get("event")
            if kind == "text":
                self.phase = "stream"
                text = event.get("text")
                if not isinstance(text, str) or not text:
                    raise EngineProtocolFailure("MLX worker emitted an invalid text event")
                parts.append(text)
                if not command["buffered"]:
                    yield GenerationEvent("text", text=text)
            elif kind == "done":
                self.phase = "final"
                count = event.get("completion_tokens")
                reason = event.get("finish_reason")
                if type(count) is not int or count < 0 or reason not in {"stop", "length"}:
                    raise EngineProtocolFailure("MLX worker did not expose exact token accounting")
                combined = "".join(parts)
                for stop in prompt.request.stop:
                    position = combined.find(stop)
                    if position >= 0:
                        combined = combined[:position]
                        break
                if prompt.request.tools:
                    try:
                        value = parse_text(combined, nested=True)
                    except ApplianceError as exc:
                        if prompt.request.tool_choice == "auto" and combined:
                            yield GenerationEvent("text", text=combined)
                            yield GenerationEvent("final", finish_reason=reason, completion_tokens=count)
                            return
                        raise EngineProtocolFailure("MLX tool output is invalid") from exc
                    if not isinstance(value, dict) or set(value) != {"name", "arguments"} or not isinstance(value.get("arguments"), dict):
                        if prompt.request.tool_choice == "auto" and combined:
                            yield GenerationEvent("text", text=combined)
                            yield GenerationEvent("final", finish_reason=reason, completion_tokens=count)
                            return
                        raise EngineProtocolFailure("MLX tool output is invalid")
                    arguments = validate_generated_tool(value["name"], canonical_text(value["arguments"]), prompt.request)
                    yield GenerationEvent("tool", tool_name=value["name"], tool_arguments=arguments)
                    yield GenerationEvent("final", finish_reason="tool_calls", completion_tokens=count)
                else:
                    if command["buffered"] and combined:
                        yield GenerationEvent("text", text=combined)
                    yield GenerationEvent("final", finish_reason=reason, completion_tokens=count)
                return
            else:
                raise EngineProtocolFailure("MLX worker emitted an unknown event")
        raise EngineFailure("MLX generation cancelled or expired")

    def cleanup(self, deadline: float) -> None:
        response = self._exchange({"op": "cleanup"}, deadline)
        if response != {"event": "idle"}:
            raise EngineProtocolFailure("MLX worker cleanup did not become idle")
        self.phase = "idle"

    def cancel(self) -> None:
        # A stuck MLX task cannot safely share its allocator state. Termination is
        # the fail-closed cancellation primitive; weights may reload afterwards.
        if self.ready():
            self.breaker.record_restart()
            self.restart_recorded = True
            self._terminate("cancel")

    def inject_crash(self) -> dict[str, Any]:
        """Kill this worker for an isolated native qualification case."""
        process = self.process
        if process is None or process.poll() is not None:
            raise EngineFailure("MLX worker is unavailable")
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
        if self.ready():
            try:
                self._send({"op": "shutdown"})
            except EngineFailure:
                pass
        self._terminate("close")

    def _exchange(self, value: dict[str, Any], deadline: float) -> dict[str, Any]:
        with self._lock:
            self._send(value)
            return self._read(deadline)

    def _send(self, value: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.poll() is not None or process.stdin is None:
            raise EngineFailure("MLX worker is unavailable")
        payload = canonical_bytes(value)
        if len(payload) > BACKEND_EVENT_BYTES:
            raise EngineProtocolFailure("oversized MLX worker command")
        try:
            process.stdin.write(payload + b"\n")
            process.stdin.flush()
        except OSError as exc:
            raise EngineFailure("MLX worker pipe failed") from exc

    def _read(self, deadline: float) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdout is None:
            raise EngineFailure("MLX worker is unavailable")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._terminate("timeout")
            raise EngineFailure("MLX worker deadline expired")
        readable, _, _ = select.select([process.stdout], [], [], remaining)
        if not readable:
            self._terminate("timeout")
            raise EngineFailure("MLX worker deadline expired")
        raw = process.stdout.readline(BACKEND_EVENT_BYTES + 1)
        if not raw or len(raw) > BACKEND_EVENT_BYTES:
            raise EngineProtocolFailure("MLX worker emitted no valid bounded event")
        try:
            value = parse_body(raw.rstrip(b"\r\n"))
        except ApplianceError as exc:
            raise EngineProtocolFailure("MLX worker emitted invalid JSON") from exc
        if not isinstance(value, dict):
            raise EngineProtocolFailure("MLX worker event is not an object")
        return value

    def _terminate(self, kind: str, *, kill_first: bool = False) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            self.phase = "stopped"
            return
        started = time.monotonic()
        terminated_phase = self.phase
        deadline = time.monotonic() + IDLE_OR_EXIT_SECONDS
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
        self.last_termination_elapsed = time.monotonic() - started
        self.last_termination_exit_ok = process.poll() is not None and self.last_termination_elapsed <= IDLE_OR_EXIT_SECONDS
        self.last_termination_kind = kind
        self.last_termination_phase = terminated_phase
        self.termination_serial += 1
        self.phase = "stopped"
