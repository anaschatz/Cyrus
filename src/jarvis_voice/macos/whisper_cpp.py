"""Manifest-gated owned whisper.cpp loopback sidecar."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import socket
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
from typing import Protocol

from ..backends import AsrScore, AudioChunk, LanguageMetadata, TranscriptResult
from ..config import ArtifactConfig, CoreMlEncoderConfig, WhisperCppConfig
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


class _LoopbackResponseProcessor(urllib.request.HTTPErrorProcessor):
    """Keep status handling with the caller; never follow HTTP redirects."""

    def http_response(self, request, response):
        return response

    https_response = http_response


_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _LoopbackResponseProcessor())


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


def _derived_coreml_encoder_directory(model: Path) -> Path:
    stem = model.with_suffix("").name
    quantisation = stem.rsplit("-", 1)
    if (
        len(quantisation) == 2
        and len(quantisation[1]) == 4
        and quantisation[1][0] == "q"
        and quantisation[1][2] == "_"
    ):
        stem = quantisation[0]
    return model.with_name(f"{stem}-encoder.mlmodelc")


def _validate_coreml_encoder(config: CoreMlEncoderConfig, model: Path) -> Path | None:
    configured = any(value is not None for value in (config.revision, config.directory)) or bool(config.files)
    if not configured:
        return None
    if (
        not isinstance(config.revision, str)
        or len(config.revision) != 40
        or config.revision != config.revision.lower()
        or any(character not in "0123456789abcdef" for character in config.revision)
        or not isinstance(config.directory, str)
        or not config.directory
        or not config.files
    ):
        raise BackendError("asr.whisper_cpp_coreml_manifest_required", "Core ML encoder manifest is required", subsystem="asr")
    directory = Path(config.directory).expanduser()
    expected_directory = _derived_coreml_encoder_directory(model)
    if directory != expected_directory:
        raise BackendError("asr.whisper_cpp_coreml_path_mismatch", "Core ML encoder path does not match whisper.cpp model selection", subsystem="asr")

    expected_files: dict[str, ArtifactConfig] = {}
    expected_directories = {"."}
    for artifact in config.files:
        if (
            not isinstance(artifact.path, str)
            or not artifact.path
            or "\\" in artifact.path
            or artifact.path in expected_files
            or not isinstance(artifact.size, int)
            or isinstance(artifact.size, bool)
            or artifact.size < 1
            or not isinstance(artifact.sha256, str)
            or len(artifact.sha256) != 64
            or artifact.sha256 != artifact.sha256.lower()
            or any(character not in "0123456789abcdef" for character in artifact.sha256)
        ):
            raise BackendError("asr.whisper_cpp_coreml_manifest_invalid", "Core ML encoder manifest is invalid", subsystem="asr")
        relative = Path(artifact.path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise BackendError("asr.whisper_cpp_coreml_manifest_invalid", "Core ML encoder manifest is invalid", subsystem="asr")
        expected_files[relative.as_posix()] = artifact
        parent = relative.parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    try:
        if directory.is_symlink() or not directory.is_dir():
            raise BackendError("asr.whisper_cpp_coreml_missing", "Core ML encoder directory is missing", subsystem="asr")
        actual_files: set[str] = set()
        actual_directories = {"."}
        for root, directories, files in os.walk(directory, followlinks=False):
            root_path = Path(root)
            relative_root = root_path.relative_to(directory)
            if relative_root != Path("."):
                actual_directories.add(relative_root.as_posix())
            for name in directories:
                child = root_path / name
                if child.is_symlink():
                    raise BackendError("asr.whisper_cpp_coreml_symlink", "Core ML encoder contains a symbolic link", subsystem="asr")
            for name in files:
                child = root_path / name
                if child.is_symlink() or not child.is_file():
                    raise BackendError("asr.whisper_cpp_coreml_symlink", "Core ML encoder contains an invalid file", subsystem="asr")
                actual_files.add(child.relative_to(directory).as_posix())
        if actual_files != set(expected_files) or actual_directories != expected_directories:
            raise BackendError("asr.whisper_cpp_coreml_file_set_mismatch", "Core ML encoder file set differs", subsystem="asr")
        for relative_name, artifact in expected_files.items():
            path = directory / relative_name
            if path.stat().st_size != artifact.size:
                raise BackendError("asr.whisper_cpp_coreml_size_mismatch", "Core ML encoder file size differs", subsystem="asr")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != str(artifact.sha256).lower():
                raise BackendError("asr.whisper_cpp_coreml_hash_mismatch", "Core ML encoder file hash differs", subsystem="asr")
    except OSError as exc:
        raise BackendError("asr.whisper_cpp_coreml_unreadable", "Core ML encoder is unreadable", subsystem="asr") from exc
    return directory


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
    except (ValueError, RecursionError) as exc:
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
    if not isinstance(language_score, (int, float)) or isinstance(language_score, bool) or not -sys.float_info.max <= language_score <= sys.float_info.max:
        language_score = 0.0

    raw_score = 0.0
    weight = 0
    segments = payload.get("segments", [])
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            avg_logprob = segment.get("avg_logprob")
            if not isinstance(avg_logprob, (int, float)) or isinstance(avg_logprob, bool) or not -sys.float_info.max <= avg_logprob <= 0.0:
                continue
            tokens = segment.get("tokens")
            count = max(1, len(tokens)) if isinstance(tokens, list) else 1
            weight += count
            raw_score += (float(avg_logprob) - raw_score) * (count / weight)
    if weight:
        score = math.exp(raw_score)
        method = "exp_weighted_avg_logprob"
    else:
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
            _validate_coreml_encoder(self.config.coreml_encoder, model)
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
