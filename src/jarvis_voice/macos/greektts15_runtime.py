"""Epoch-safe runtime boundary for the approved GreekTTS F1 voice."""

from __future__ import annotations

import hashlib
import json
import queue
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..backends import SpeechUnit
from ..errors import BackendError


APPROVED_CANDIDATE_SCHEMA = "jarvis.voice.greektts15-approved-candidate/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ApprovedGreekTtsCandidate:
    checkpoint_directory: Path
    checkpoint_sha256: str
    temperature: float
    top_p: float
    repetition_penalty: float


def load_approved_greektts_candidate(
    manifest_path: str | Path,
    repository_root: str | Path,
) -> ApprovedGreekTtsCandidate:
    """Load the human-approved candidate and bind it to the exact local adapter."""

    manifest_file = Path(manifest_path).expanduser()
    root = Path(repository_root).expanduser()
    if (
        not manifest_file.is_absolute()
        or not root.is_absolute()
        or manifest_file.is_symlink()
        or root.is_symlink()
        or not manifest_file.is_file()
        or not root.is_dir()
    ):
        raise BackendError("tts.greektts15.approval_manifest_invalid", "Approved voice manifest path is invalid", subsystem="tts")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.approval_manifest_invalid", "Approved voice manifest is unreadable", subsystem="tts") from exc
    if not isinstance(manifest, dict):
        raise BackendError("tts.greektts15.approval_manifest_invalid", "Approved voice manifest is invalid", subsystem="tts")
    model = manifest.get("model")
    sampling = model.get("sampling") if isinstance(model, dict) else None
    checkpoint_hash = model.get("checkpoint_sha256") if isinstance(model, dict) else None
    relative = model.get("checkpoint_directory") if isinstance(model, dict) else None
    if (
        manifest.get("schema") != APPROVED_CANDIDATE_SCHEMA
        or manifest.get("status") != "human_voice_approved"
        or manifest.get("human_voice_approved") is not True
        or manifest.get("production_promotion_allowed") is not False
        or not isinstance(relative, str)
        or Path(relative).is_absolute()
        or not isinstance(checkpoint_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", checkpoint_hash) is None
        or not isinstance(sampling, dict)
        or set(sampling) != {"temperature", "top_p", "repetition_penalty"}
        or any(
            not isinstance(sampling.get(name), (int, float)) or isinstance(sampling.get(name), bool)
            for name in ("temperature", "top_p", "repetition_penalty")
        )
    ):
        raise BackendError("tts.greektts15.approval_manifest_invalid", "Approved voice manifest is invalid", subsystem="tts")
    unresolved_checkpoint = root / relative
    if unresolved_checkpoint.is_symlink():
        raise BackendError("tts.greektts15.approved_checkpoint_invalid", "Approved F1 checkpoint changed", subsystem="tts")
    checkpoint = unresolved_checkpoint.resolve()
    try:
        checkpoint.relative_to(root.resolve())
    except ValueError as exc:
        raise BackendError("tts.greektts15.approval_manifest_invalid", "Approved checkpoint escapes its root", subsystem="tts") from exc
    adapter = checkpoint / "adapters.safetensors"
    config = checkpoint / "adapter_config.json"
    if (
        checkpoint.is_symlink()
        or not checkpoint.is_dir()
        or adapter.is_symlink()
        or config.is_symlink()
        or not adapter.is_file()
        or not config.is_file()
        or _sha256(adapter) != checkpoint_hash
    ):
        raise BackendError("tts.greektts15.approved_checkpoint_invalid", "Approved F1 checkpoint changed", subsystem="tts")
    return ApprovedGreekTtsCandidate(
        checkpoint_directory=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        temperature=float(sampling["temperature"]),
        top_p=float(sampling["top_p"]),
        repetition_penalty=float(sampling["repetition_penalty"]),
    )


@dataclass(frozen=True, slots=True)
class PcmChunk:
    pcm16: bytes
    sample_rate: int = 24_000
    channels: int = 1


class GreekTtsStreamingEngine(Protocol):
    def stream(self, unit: SpeechUnit, cancelled: threading.Event) -> Iterable[PcmChunk]: ...
    def cancel(self) -> None: ...


class PcmPlayback(Protocol):
    def write(self, chunk: PcmChunk) -> None: ...
    def finish(self) -> None: ...
    def abort(self) -> None: ...


class GreekTtsF1Backend:
    """Serialize sentence units and fence every generated chunk by output epoch."""

    def __init__(
        self,
        *,
        engine: GreekTtsStreamingEngine,
        playback_factory: Callable[[], PcmPlayback],
        on_started: Callable[[SpeechUnit], None] | None = None,
        on_completed: Callable[[SpeechUnit], None] | None = None,
        queue_capacity: int = 16,
        maximum_text_bytes: int = 16_384,
    ) -> None:
        if queue_capacity < 1 or maximum_text_bytes < 1:
            raise ValueError("GreekTTS runtime bounds must be positive")
        self.engine = engine
        self.playback_factory = playback_factory
        self.on_started = on_started or (lambda unit: None)
        self.on_completed = on_completed or (lambda unit: None)
        self.maximum_text_bytes = maximum_text_bytes
        self._queue: queue.Queue[SpeechUnit] = queue.Queue(maxsize=queue_capacity)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._cancelled_through_epoch = 0
        self._active: SpeechUnit | None = None
        self._active_cancel: threading.Event | None = None
        self._playback: PcmPlayback | None = None
        self._error: BackendError | None = None
        self._thread = threading.Thread(target=self._worker, name="jarvis-greektts-f1", daemon=True)
        self._thread.start()

    def enqueue(self, unit: SpeechUnit) -> None:
        size = len(unit.text.encode("utf-8"))
        if size == 0 or size > self.maximum_text_bytes or unit.locale != "el-GR":
            raise BackendError("tts.unit_invalid", "GreekTTS speech unit is invalid", subsystem="tts")
        with self._lock:
            if unit.epoch <= self._cancelled_through_epoch:
                return
            try:
                self._queue.put_nowait(unit)
            except queue.Full as exc:
                raise BackendError("tts.queue_overflow", "GreekTTS sentence queue is full", subsystem="tts") from exc

    def cancel(self, new_epoch: int, reason: str) -> None:
        del reason
        with self._lock:
            self._cancelled_through_epoch = max(self._cancelled_through_epoch, new_epoch)
            retained: list[SpeechUnit] = []
            while True:
                try:
                    unit = self._queue.get_nowait()
                except queue.Empty:
                    break
                if unit.epoch > self._cancelled_through_epoch:
                    retained.append(unit)
            for unit in retained:
                self._queue.put_nowait(unit)
            cancel_event = self._active_cancel
            playback = self._playback
        if cancel_event is not None:
            cancel_event.set()
        self.engine.cancel()
        if playback is not None:
            playback.abort()

    @property
    def is_idle(self) -> bool:
        with self._lock:
            return self._active is None and self._queue.empty()

    @property
    def last_error(self) -> BackendError | None:
        return self._error

    def wait_idle(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_idle:
                return True
            time.sleep(0.005)
        return self.is_idle

    def close(self) -> None:
        self._stop.set()
        self.cancel(2**63 - 1, "application_close")
        self._thread.join(timeout=2.0)

    def _is_stale(self, unit: SpeechUnit, cancelled: threading.Event) -> bool:
        with self._lock:
            return self._stop.is_set() or cancelled.is_set() or unit.epoch <= self._cancelled_through_epoch

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                unit = self._queue.get(timeout=0.02)
            except queue.Empty:
                continue
            cancelled = threading.Event()
            with self._lock:
                if unit.epoch <= self._cancelled_through_epoch:
                    continue
                self._active = unit
                self._active_cancel = cancelled
            playback: PcmPlayback | None = None
            started = False
            try:
                for chunk in self.engine.stream(unit, cancelled):
                    if self._is_stale(unit, cancelled):
                        break
                    if not chunk.pcm16 or chunk.sample_rate != 24_000 or chunk.channels != 1:
                        raise BackendError("tts.audio_chunk_invalid", "GreekTTS emitted an invalid PCM chunk", subsystem="tts")
                    if playback is None:
                        playback = self.playback_factory()
                        with self._lock:
                            self._playback = playback
                    if self._is_stale(unit, cancelled):
                        playback.abort()
                        break
                    playback.write(chunk)
                    if not started and not self._is_stale(unit, cancelled):
                        started = True
                        self.on_started(unit)
                stale = self._is_stale(unit, cancelled)
                if playback is not None and not stale:
                    playback.finish()
                stale = self._is_stale(unit, cancelled)
                if started and not stale:
                    self.on_completed(unit)
            except Exception as exc:
                self._error = BackendError("tts.greektts15.runtime_failed", "GreekTTS runtime failed", subsystem="tts")
                self._error.__cause__ = exc
                if playback is not None:
                    playback.abort()
            finally:
                with self._lock:
                    self._playback = None
                    self._active_cancel = None
                    self._active = None
