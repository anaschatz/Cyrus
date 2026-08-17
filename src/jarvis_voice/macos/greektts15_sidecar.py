"""Digest-bound process proxy for the approved GreekTTS F1 backend."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import BinaryIO, Callable

from ..backends import SpeechUnit
from ..config import ArtifactConfig, GreekTts15Config
from ..errors import BackendError


MAXIMUM_MESSAGE_BYTES = 65_536
COMMAND_QUEUE_CAPACITY = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_worker_python(artifact: ArtifactConfig) -> Path:
    path = Path(artifact.path or "").expanduser()
    if (
        not path.is_absolute()
        or not path.is_file()
        or not os.access(path, os.X_OK)
        or not isinstance(artifact.size, int)
        or isinstance(artifact.size, bool)
        or artifact.size < 1
        or not isinstance(artifact.sha256, str)
        or len(artifact.sha256) != 64
    ):
        raise BackendError("tts.greektts15.worker_invalid", "GreekTTS worker interpreter is invalid", subsystem="tts")
    try:
        valid = path.stat().st_size == artifact.size and _sha256(path) == artifact.sha256
    except OSError as exc:
        raise BackendError("tts.greektts15.worker_invalid", "GreekTTS worker interpreter is invalid", subsystem="tts") from exc
    if not valid:
        raise BackendError("tts.greektts15.worker_changed", "GreekTTS worker interpreter changed", subsystem="tts")
    return path


def _worker_environment() -> dict[str, str]:
    environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    for name in ("PATH", "TMPDIR", "LANG", "LC_ALL"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


class GreekTtsF1SidecarBackend:
    """Run MLX/SNAC and CoreAudio outside the ASR/VAD client process."""

    def __init__(
        self,
        *,
        config: GreekTts15Config,
        on_started: Callable[[SpeechUnit], None] | None = None,
        on_completed: Callable[[SpeechUnit], None] | None = None,
        queue_capacity: int = 16,
        maximum_text_bytes: int = 16_384,
        startup_timeout_seconds: float = 120.0,
        shutdown_timeout_seconds: float = 5.0,
        process_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        worker_script: str | Path | None = None,
    ) -> None:
        if queue_capacity < 1 or maximum_text_bytes < 1 or startup_timeout_seconds <= 0 or shutdown_timeout_seconds <= 0:
            raise ValueError("GreekTTS sidecar bounds must be positive")
        python = validate_worker_python(config.worker_python)
        script = Path(worker_script) if worker_script is not None else Path(__file__).resolve().parents[3] / "scripts" / "run_greektts15_sidecar_worker.py"
        if not script.is_absolute() or script.is_symlink() or not script.is_file():
            raise BackendError("tts.greektts15.worker_script_invalid", "GreekTTS worker script is invalid", subsystem="tts")
        self.on_started = on_started or (lambda unit: None)
        self.on_completed = on_completed or (lambda unit: None)
        self.maximum_text_bytes = maximum_text_bytes
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._commands: queue.Queue[dict[str, object] | None] = queue.Queue(maxsize=COMMAND_QUEUE_CAPACITY)
        self._lock = threading.RLock()
        self._pending: dict[tuple[str, int, int], SpeechUnit] = {}
        self._started: set[tuple[str, int, int]] = set()
        self._cancelled_through_epoch = 0
        self._last_error: BackendError | None = None
        self._warm_load_ms: float | None = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._stop = threading.Event()
        self._process = process_factory(
            [str(python), str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_worker_environment(),
            bufsize=0,
            shell=False,
            start_new_session=True,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self._terminate_process()
            raise BackendError("tts.greektts15.worker_start_failed", "GreekTTS worker pipes are unavailable", subsystem="tts")
        self._stdin: BinaryIO = self._process.stdin
        self._stdout: BinaryIO = self._process.stdout
        self._writer = threading.Thread(target=self._writer_loop, name="jarvis-greektts-sidecar-writer", daemon=True)
        self._reader = threading.Thread(target=self._reader_loop, name="jarvis-greektts-sidecar-reader", daemon=True)
        self._writer.start()
        self._reader.start()
        self._publish(
            {
                "type": "init",
                "model_root": config.model_root,
                "approval_manifest_path": config.approval_manifest_path,
                "approval_manifest_sha256": config.approval_manifest_sha256,
                "repository_root": config.repository_root,
                "warm_on_start": config.warm_on_start,
                "queue_capacity": queue_capacity,
                "maximum_text_bytes": maximum_text_bytes,
            }
        )
        if not self._ready.wait(timeout=startup_timeout_seconds) or self._last_error is not None:
            error = self._last_error or BackendError(
                "tts.greektts15.worker_start_timeout",
                "GreekTTS worker did not become ready",
                subsystem="tts",
            )
            self.close()
            raise error

    @staticmethod
    def _identity(unit: SpeechUnit) -> tuple[str, int, int]:
        return unit.response_id, unit.epoch, unit.unit_index

    def _publish(self, message: dict[str, object]) -> None:
        try:
            self._commands.put_nowait(message)
        except queue.Full as exc:
            self._fail("tts.greektts15.worker_command_overflow")
            raise BackendError("tts.greektts15.worker_command_overflow", "GreekTTS worker command queue overflowed", subsystem="tts") from exc

    def enqueue(self, unit: SpeechUnit) -> None:
        size = len(unit.text.encode("utf-8"))
        if size == 0 or size > self.maximum_text_bytes or unit.locale != "el-GR":
            raise BackendError("tts.unit_invalid", "GreekTTS speech unit is invalid", subsystem="tts")
        identity = self._identity(unit)
        with self._lock:
            if self._last_error is not None:
                raise self._last_error
            if unit.epoch <= self._cancelled_through_epoch:
                return
            if identity in self._pending:
                raise BackendError("tts.greektts15.worker_duplicate_unit", "GreekTTS worker unit was duplicated", subsystem="tts")
            self._pending[identity] = unit
        try:
            self._publish(
                {
                    "type": "enqueue",
                    "response_id": unit.response_id,
                    "epoch": unit.epoch,
                    "unit_index": unit.unit_index,
                    "locale": unit.locale,
                    "text": unit.text,
                }
            )
        except Exception:
            with self._lock:
                self._pending.pop(identity, None)
            raise

    def cancel(self, new_epoch: int, reason: str) -> None:
        del reason
        with self._lock:
            self._cancelled_through_epoch = max(self._cancelled_through_epoch, new_epoch)
            self._pending = {
                identity: unit for identity, unit in self._pending.items() if unit.epoch > self._cancelled_through_epoch
            }
            self._started = {identity for identity in self._started if identity in self._pending}
        if self._last_error is None and not self._stop.is_set():
            try:
                self._publish({"type": "cancel", "epoch": new_epoch})
            except BackendError:
                return

    @property
    def is_idle(self) -> bool:
        with self._lock:
            return not self._pending

    @property
    def last_error(self) -> BackendError | None:
        return self._last_error

    @property
    def warm_load_ms(self) -> float | None:
        return self._warm_load_ms

    def wait_idle(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_idle:
                return True
            time.sleep(0.005)
        return self.is_idle

    def _writer_loop(self) -> None:
        try:
            while not self._stop.is_set():
                message = self._commands.get()
                if message is None:
                    return
                encoded = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                if len(encoded) > MAXIMUM_MESSAGE_BYTES:
                    self._fail("tts.greektts15.worker_message_too_large")
                    return
                self._stdin.write(encoded)
                self._stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            if not self._stop.is_set():
                self._fail("tts.greektts15.worker_pipe_failed")

    def _reader_loop(self) -> None:
        try:
            while not self._stop.is_set():
                raw = self._stdout.readline(MAXIMUM_MESSAGE_BYTES + 1)
                if not raw:
                    if not self._stop.is_set():
                        self._fail("tts.greektts15.worker_exited")
                    return
                if len(raw) > MAXIMUM_MESSAGE_BYTES or not raw.endswith(b"\n"):
                    self._fail("tts.greektts15.worker_protocol_invalid")
                    return
                try:
                    message = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._fail("tts.greektts15.worker_protocol_invalid")
                    return
                self._handle_message(message)
                if self._closed.is_set():
                    return
        except (OSError, ValueError):
            if not self._stop.is_set():
                self._fail("tts.greektts15.worker_pipe_failed")

    def _handle_message(self, message: object) -> None:
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            self._fail("tts.greektts15.worker_protocol_invalid")
            return
        kind = message["type"]
        if kind == "ready" and set(message) == {"type", "warm_load_ms"}:
            if self._ready.is_set():
                self._fail("tts.greektts15.worker_protocol_invalid")
                return
            value = message["warm_load_ms"]
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0):
                self._fail("tts.greektts15.worker_protocol_invalid")
                return
            self._warm_load_ms = None if value is None else float(value)
            self._ready.set()
            return
        if kind == "error" and set(message) == {"type", "code"} and isinstance(message.get("code"), str):
            self._fail(str(message["code"]))
            return
        if kind == "closed" and set(message) == {"type"}:
            self._closed.set()
            return
        if kind not in {"started", "completed"} or set(message) != {"type", "response_id", "epoch", "unit_index"}:
            self._fail("tts.greektts15.worker_protocol_invalid")
            return
        identity = message.get("response_id"), message.get("epoch"), message.get("unit_index")
        if (
            not isinstance(identity[0], str)
            or not isinstance(identity[1], int)
            or isinstance(identity[1], bool)
            or identity[1] < 1
            or not isinstance(identity[2], int)
            or isinstance(identity[2], bool)
            or identity[2] < 1
        ):
            self._fail("tts.greektts15.worker_protocol_invalid")
            return
        protocol_error = False
        with self._lock:
            unit = self._pending.get(identity)
            stale = unit is None or unit.epoch <= self._cancelled_through_epoch
            if not stale and kind == "started":
                if identity in self._started:
                    protocol_error = True
                else:
                    self._started.add(identity)
            elif not stale and identity not in self._started:
                protocol_error = True
            elif kind == "completed" and unit is not None:
                self._pending.pop(identity, None)
                self._started.discard(identity)
        if protocol_error:
            self._fail("tts.greektts15.worker_protocol_invalid")
            return
        if stale:
            return
        (self.on_started if kind == "started" else self.on_completed)(unit)

    def _fail(self, code: str) -> None:
        with self._lock:
            if self._last_error is None:
                self._last_error = BackendError(code, "GreekTTS worker failed", subsystem="tts")
            self._pending.clear()
            self._started.clear()
        self._ready.set()
        self._closed.set()
        self._terminate_process()

    def _terminate_process(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
            process.wait(timeout=1.0)

    def close(self) -> None:
        if self._stop.is_set():
            return
        try:
            if self._last_error is None and self._process.poll() is None:
                self._publish({"type": "close"})
                self._closed.wait(timeout=self._shutdown_timeout_seconds)
        except BackendError:
            pass
        finally:
            self._stop.set()
            try:
                self._commands.put_nowait(None)
            except queue.Full:
                pass
            self._terminate_process()
            for stream in (getattr(self, "_stdin", None), getattr(self, "_stdout", None)):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            for thread in (getattr(self, "_writer", None), getattr(self, "_reader", None)):
                if thread is not None and thread is not threading.current_thread():
                    thread.join(timeout=1.0)
