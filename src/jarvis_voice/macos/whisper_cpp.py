"""Manifest-gated owned whisper.cpp loopback sidecar."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
import wave
from array import array
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from ..backends import AsrScore, AudioChunk, LanguageMetadata, TranscriptResult
from ..config import ArtifactConfig, WhisperCppConfig
from ..errors import BackendError


class SidecarProcess(Protocol):
    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


class HttpResponse(Protocol):
    def read(self, amount: int = -1) -> bytes: ...
    def getcode(self) -> int: ...
    def __enter__(self) -> "HttpResponse": ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...


def _start_sidecar(argv: list[str]) -> SidecarProcess:
    return subprocess.Popen(  # type: ignore[return-value]
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
        close_fds=True,
    )


_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _open_loopback(request: urllib.request.Request, *, timeout: float) -> HttpResponse:
    return _opener.open(request, timeout=timeout)  # type: ignore[return-value]


def _available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _validate_artifact(artifact: ArtifactConfig, kind: str, *, executable: bool = False) -> Path:
    if not isinstance(artifact.path, str) or not artifact.path or not isinstance(artifact.size, int) or artifact.size < 1 or not isinstance(artifact.sha256, str) or len(artifact.sha256) != 64:
        raise BackendError(f"asr.whisper_cpp_{kind}_manifest_required", "Local whisper.cpp artifact manifest is required", subsystem="asr")
    path = Path(artifact.path).expanduser()
    try:
        if path.is_symlink() or not path.is_file():
            raise BackendError(f"asr.whisper_cpp_{kind}_missing", "Local whisper.cpp artifact is missing", subsystem="asr")
        if path.stat().st_size != artifact.size:
            raise BackendError(f"asr.whisper_cpp_{kind}_size_mismatch", "Local whisper.cpp artifact size differs", subsystem="asr")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != artifact.sha256.lower():
            raise BackendError(f"asr.whisper_cpp_{kind}_hash_mismatch", "Local whisper.cpp artifact hash differs", subsystem="asr")
        if executable and not os.access(path, os.X_OK):
            raise BackendError("asr.whisper_cpp_executable_not_executable", "whisper.cpp sidecar is not executable", subsystem="asr")
    except OSError as exc:
        raise BackendError(f"asr.whisper_cpp_{kind}_unreadable", "Local whisper.cpp artifact is unreadable", subsystem="asr") from exc
    return path


def audio_chunk_to_wav(audio: AudioChunk) -> bytes:
    if audio.sample_rate != 16_000:
        raise BackendError("asr.invalid_audio", "whisper.cpp requires mono 16-kHz audio", subsystem="asr")
    pcm = array("h", (round(max(-1.0, min(1.0, float(sample))) * 32767.0) for sample in audio.samples))
    if sys.byteorder != "little":
        pcm.byteswap()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def _multipart(wav: bytes, *, language: str, audio_context: int, boundary: str) -> bytes:
    marker = boundary.encode("ascii")
    chunks: list[bytes] = []

    def field(name: str, value: str) -> None:
        chunks.extend([
            b"--" + marker + b"\r\n",
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"),
            value.encode("ascii"),
            b"\r\n",
        ])

    field("response_format", "verbose_json")
    field("language", language)
    field("temperature", "0.0")
    field("audio_ctx", str(audio_context))
    chunks.extend([
        b"--" + marker + b"\r\n",
        b'Content-Disposition: form-data; name="file"; filename="turn.wav"\r\n',
        b"Content-Type: audio/wav\r\n\r\n",
        wav,
        b"\r\n--" + marker + b"--\r\n",
    ])
    return b"".join(chunks)


def parse_whisper_response(raw: bytes, *, configured_language: str) -> TranscriptResult:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendError("asr.whisper_cpp_response_invalid", "whisper.cpp returned invalid JSON", subsystem="asr") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("text"), str):
        raise BackendError("asr.whisper_cpp_response_invalid", "whisper.cpp response schema is invalid", subsystem="asr")
    text = unicodedata.normalize("NFC", payload["text"].strip())
    if not text:
        raise BackendError("asr.empty_transcript", "ASR produced no transcript", subsystem="asr")
    language = payload.get("language")
    if not isinstance(language, str) or not language:
        language = configured_language if configured_language != "auto" else "und"
    language_score = payload.get("language_probability", 0.0)
    if not isinstance(language_score, (int, float)) or isinstance(language_score, bool) or not math.isfinite(float(language_score)):
        language_score = 0.0

    weighted = 0.0
    weight = 0
    segments = payload.get("segments", [])
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            avg_logprob = segment.get("avg_logprob")
            if not isinstance(avg_logprob, (int, float)) or isinstance(avg_logprob, bool) or not math.isfinite(float(avg_logprob)):
                continue
            tokens = segment.get("tokens")
            count = max(1, len(tokens)) if isinstance(tokens, list) else 1
            weighted += float(avg_logprob) * count
            weight += count
    if weight:
        raw_score = weighted / weight
        score = max(0.0, min(1.0, math.exp(raw_score)))
        method = "exp_weighted_avg_logprob"
    else:
        raw_score = 0.0
        score = 0.0
        method = "whisper_cpp_score_unavailable"
    return TranscriptResult(
        text=text,
        language=LanguageMetadata(language, max(0.0, min(1.0, float(language_score)))),
        asr_score=AsrScore(score, raw_score, method=method, calibrated=False),
    )


class WhisperCppSidecarBackend:
    def __init__(
        self,
        config: WhisperCppConfig,
        *,
        maximum_audio_seconds: float,
        process_factory: Callable[[list[str]], SidecarProcess] = _start_sidecar,
        opener: Callable[..., HttpResponse] = _open_loopback,
        port_factory: Callable[[], int] = _available_loopback_port,
        platform_name: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.maximum_audio_samples = round(maximum_audio_seconds * 16_000)
        self.process_factory = process_factory
        self.opener = opener
        self.port_factory = port_factory
        self.platform_name = sys.platform if platform_name is None else platform_name
        self.clock = clock
        self.sleeper = sleeper
        self._process: SidecarProcess | None = None
        self._port: int | None = None
        self._request_path: str | None = None
        self._lock = threading.RLock()

    @property
    def sidecar_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def load(self) -> None:
        if self.platform_name != "darwin":
            raise BackendError("asr.whisper_cpp_macos_only", "whisper.cpp sidecar requires macOS", subsystem="asr")
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            executable = _validate_artifact(self.config.executable, "executable", executable=True)
            model = _validate_artifact(self.config.model, "model")
            port = self.port_factory()
            if not 1024 <= port <= 65535:
                raise BackendError("asr.whisper_cpp_port_invalid", "Loopback sidecar port is invalid", subsystem="asr")
            request_path = "/jarvis-" + uuid.uuid4().hex
            argv = [
                str(executable),
                "--host", "127.0.0.1",
                "--port", str(port),
                "--request-path", request_path,
                "--model", str(model),
                "--threads", str(self.config.threads),
                "--language", self.config.language,
                "--max-context", "0",
                "--flash-attn",
            ]
            try:
                self._process = self.process_factory(argv)
            except OSError as exc:
                raise BackendError("asr.whisper_cpp_start_failed", "whisper.cpp sidecar could not start", subsystem="asr") from exc
            self._port = port
            self._request_path = request_path
        deadline = self.clock() + self.config.startup_timeout_seconds
        while self.clock() < deadline:
            with self._lock:
                process = self._process
            if process is None or process.poll() is not None:
                self.close()
                raise BackendError("asr.whisper_cpp_start_failed", "whisper.cpp sidecar exited during startup", subsystem="asr")
            try:
                request = urllib.request.Request(f"http://127.0.0.1:{port}{request_path}/", method="GET")
                with self.opener(request, timeout=0.25) as response:
                    if response.getcode() == 200:
                        response.read(1)
                        return
            except (OSError, urllib.error.URLError, TimeoutError):
                pass
            self.sleeper(0.025)
        self.close()
        raise BackendError("asr.whisper_cpp_start_timeout", "whisper.cpp sidecar startup timed out", subsystem="asr")

    def transcribe(self, audio: object) -> TranscriptResult:
        if not isinstance(audio, AudioChunk) or audio.sample_rate != 16_000:
            raise BackendError("asr.invalid_audio", "whisper.cpp requires mono 16-kHz audio", subsystem="asr")
        if len(audio.samples) > self.maximum_audio_samples:
            raise BackendError("asr.audio_too_large", "Audio exceeds the configured utterance limit", subsystem="asr")
        self.load()
        wav = audio_chunk_to_wav(audio)
        boundary = "jarvis-" + uuid.uuid4().hex
        duration_seconds = len(audio.samples) / audio.sample_rate
        audio_context = 0
        if (
            self.config.qualified_audio_context
            and self.config.qualified_audio_context_minimum_seconds <= duration_seconds
            <= self.config.qualified_audio_context_maximum_seconds
        ):
            audio_context = self.config.qualified_audio_context
        body = _multipart(
            wav,
            language=self.config.language,
            audio_context=audio_context,
            boundary=boundary,
        )
        with self._lock:
            port = self._port
            request_path = self._request_path
        if port is None or request_path is None:
            raise BackendError("asr.whisper_cpp_not_running", "whisper.cpp sidecar is unavailable", subsystem="asr")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{request_path}/inference",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
            method="POST",
        )
        try:
            with self.opener(request, timeout=self.config.request_timeout_seconds) as response:
                if response.getcode() != 200:
                    raise BackendError("asr.whisper_cpp_http_failed", "whisper.cpp inference failed", subsystem="asr")
                raw = response.read(self.config.maximum_response_bytes + 1)
        except BackendError:
            raise
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise BackendError("asr.whisper_cpp_request_failed", "whisper.cpp inference request failed", subsystem="asr", retryable=True) from exc
        if len(raw) > self.config.maximum_response_bytes:
            raise BackendError("asr.whisper_cpp_response_too_large", "whisper.cpp response exceeded its limit", subsystem="asr")
        return parse_whisper_response(raw, configured_language=self.config.language)

    def cancel_current(self) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            self._port = None
            self._request_path = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=0.5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                return
            try:
                process.wait(timeout=0.5)
            except (OSError, subprocess.TimeoutExpired):
                pass
