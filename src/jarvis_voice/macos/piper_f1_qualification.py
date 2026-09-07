"""Offline qualification gate for the future STOMA F1 Piper voice.

This module does not install Piper, acquire a model, or mutate the active TTS
backend.  A real run is possible only after the two explicitly named candidate
files exist and ``piper-tts==1.6.0`` is available in the invoking environment.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import resource
import socket
import sys
import threading
import time
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol

from ..errors import BackendError
from .compact_greek_tts_pilot import (
    MAX_CANCEL_TO_SILENCE_MS,
    MAX_EXPORTED_MODEL_BYTES,
    MAX_RUNTIME_RSS_BYTES,
    MAX_WARM_FIRST_AUDIO_MS,
    PIPER_TARGET_SAMPLE_RATE,
    PIPER_VERSION,
)


MODEL_NAME = "el_GR-jarvis-f1-medium.onnx"
CONFIG_NAME = f"{MODEL_NAME}.json"
MAX_CONFIG_BYTES = 64 * 1024
WORKER_START_TIMEOUT_SECONDS = 120.0
SYNTHESIS_TIMEOUT_SECONDS = 120.0

FIXTURES = (
    ("native-phonemes", "Ο μπαμπάς πήγε στην αντάρα για να βρει τα αγκάθια."),
    ("technical", "Η τεχνητή νοημοσύνη απαντά καθαρά, γρήγορα και υπεύθυνα."),
    ("date-time", "Στις είκοσι τρεις Αυγούστου, στις οκτώ και μισή, έχουμε συνάντηση."),
    ("mixed-name", "Το JARVIS λειτουργεί τοπικά, χωρίς σύνδεση στο διαδίκτυο."),
)
WARMUP_TEXT = "Καλημέρα."
CANCELLATION_TEXT = (
    "Αυτή είναι μια μεγαλύτερη πρόταση για τη δοκιμή άμεσης διακοπής της φωνής. "
    "Το υπόλοιπο κείμενο δεν πρέπει να ακουστεί μετά την ακύρωση."
)


def _error(code: str, message: str) -> BackendError:
    return BackendError(code, message, subsystem="tts")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _piper_runtime_ready() -> bool:
    try:
        return importlib.metadata.version("piper-tts") == PIPER_VERSION
    except importlib.metadata.PackageNotFoundError:
        return False


def _candidate_root(value: str | Path) -> Path:
    root = Path(value).expanduser()
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise _error("tts.piper_f1.candidate_root_invalid", "Piper F1 candidate root is invalid")
    return root


def _valid_config(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    audio = payload.get("audio")
    espeak = payload.get("espeak")
    language = payload.get("language")
    phoneme_ids = payload.get("phoneme_id_map")
    return (
        isinstance(audio, dict)
        and audio.get("sample_rate") == PIPER_TARGET_SAMPLE_RATE
        and isinstance(espeak, dict)
        and espeak.get("voice") == "el"
        and isinstance(language, dict)
        and language.get("code") == "el"
        and payload.get("phoneme_type") == "espeak"
        and payload.get("num_speakers") == 1
        and payload.get("num_symbols") == 256
        and isinstance(phoneme_ids, dict)
        and bool(phoneme_ids)
    )


def inspect_piper_f1_candidate(candidate_root: str | Path) -> dict[str, object]:
    """Validate the exact future ONNX/config pair without loading either file."""

    root = _candidate_root(candidate_root)
    model = root / MODEL_NAME
    config = root / CONFIG_NAME
    report: dict[str, object] = {
        "ready": False,
        "code": "tts.piper_f1.model_missing",
        "model": {"name": MODEL_NAME, "present": False},
        "config": {"name": CONFIG_NAME, "present": False},
        "network_used": False,
        "installation_performed": False,
        "production_backend_changed": False,
    }
    if not model.exists() or not config.exists():
        return report
    if model.is_symlink() or config.is_symlink() or not model.is_file() or not config.is_file():
        report["code"] = "tts.piper_f1.candidate_file_invalid"
        return report
    try:
        model_size = model.stat().st_size
        config_size = config.stat().st_size
    except OSError:
        report["code"] = "tts.piper_f1.candidate_file_invalid"
        return report
    report["model"] = {
        "name": MODEL_NAME,
        "present": True,
        "size_bytes": model_size,
        "max_size_bytes": MAX_EXPORTED_MODEL_BYTES,
    }
    report["config"] = {
        "name": CONFIG_NAME,
        "present": True,
        "size_bytes": config_size,
        "max_size_bytes": MAX_CONFIG_BYTES,
    }
    if model_size <= 0 or model_size > MAX_EXPORTED_MODEL_BYTES:
        report["code"] = "tts.piper_f1.model_size_invalid"
        return report
    if config_size <= 0 or config_size > MAX_CONFIG_BYTES:
        report["code"] = "tts.piper_f1.config_size_invalid"
        return report
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        report["code"] = "tts.piper_f1.config_invalid"
        return report
    if not _valid_config(payload):
        report["code"] = "tts.piper_f1.config_incompatible"
        return report
    report["model"]["sha256"] = _sha256(model)  # type: ignore[index]
    report["config"]["sha256"] = _sha256(config)  # type: ignore[index]
    report["ready"] = True
    report["code"] = "tts.piper_f1.candidate_ready"
    return report


def piper_f1_qualification_preflight(
    candidate_root: str | Path,
    *,
    runtime_probe: Callable[[], bool] = _piper_runtime_ready,
) -> dict[str, object]:
    """Return a read-only, no-install qualification readiness report."""

    candidate = inspect_piper_f1_candidate(candidate_root)
    runtime_ready = bool(runtime_probe())
    hardware_ready = sys.platform == "darwin" and platform.machine() == "arm64"
    python_ready = sys.version_info[:2] == (3, 11)
    host_ready = hardware_ready and python_ready
    ready = bool(candidate["ready"]) and runtime_ready and host_ready
    if not hardware_ready:
        code = "tts.piper_f1.macos_arm64_required"
    elif not python_ready:
        code = "tts.piper_f1.python_311_required"
    elif not candidate["ready"]:
        code = str(candidate["code"])
    elif not runtime_ready:
        code = "tts.piper_f1.runtime_missing"
    else:
        code = "tts.piper_f1.qualification_ready"
    return {
        "ready": ready,
        "code": code,
        "candidate": candidate,
        "host": {
            "platform": sys.platform,
            "machine": platform.machine(),
            "python": platform.python_version(),
            "hardware_ready": hardware_ready,
            "python_ready": python_ready,
            "ready": host_ready,
        },
        "runtime": {"distribution": "piper-tts", "version": PIPER_VERSION, "ready": runtime_ready},
        "gates": {
            "max_model_bytes": MAX_EXPORTED_MODEL_BYTES,
            "max_peak_rss_bytes": MAX_RUNTIME_RSS_BYTES,
            "max_warm_first_audio_ms": MAX_WARM_FIRST_AUDIO_MS,
            "max_cancel_to_silence_ms": MAX_CANCEL_TO_SILENCE_MS,
            "human_voice_approval_required": True,
        },
        "network_used": False,
        "installation_performed": False,
        "production_backend_changed": False,
    }


@contextmanager
def _offline_runtime() -> Iterator[None]:
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
    previous = {name: os.environ.get(name) for name in names}
    original_socket = socket.socket
    original_create_connection = socket.create_connection

    class OfflineSocket(original_socket):
        def connect(self, address: object) -> None:
            del address
            raise _error("tts.piper_f1.network_forbidden", "Network is forbidden during qualification")

        def connect_ex(self, address: object) -> int:
            self.connect(address)
            return 1

    def deny_create_connection(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise _error("tts.piper_f1.network_forbidden", "Network is forbidden during qualification")

    os.environ.update({name: "1" for name in names})
    socket.socket = OfflineSocket
    socket.create_connection = deny_create_connection
    try:
        yield
    finally:
        socket.socket = original_socket
        socket.create_connection = original_create_connection
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


@dataclass(frozen=True, slots=True)
class PcmChunk:
    pcm16: bytes
    sample_rate: int
    channels: int = 1
    sample_width: int = 2


class SynthesisSession(Protocol):
    @property
    def peak_rss_bytes(self) -> int: ...

    def stream(self, text: str) -> Iterator[PcmChunk]: ...

    def abort(self) -> float: ...

    def close(self) -> None: ...


class PlaybackSink(Protocol):
    def open(self, chunk: PcmChunk) -> None: ...

    def write(self, pcm16: bytes) -> None: ...

    def abort(self) -> None: ...

    def close(self) -> None: ...


def _worker_main(connection: object, model_path: str, config_path: str) -> None:
    conn = connection
    try:
        with _offline_runtime():
            from piper import PiperVoice

            voice = PiperVoice.load(
                model_path,
                config_path=config_path,
                use_cuda=False,
                include_alignments=False,
            )
            conn.send({"type": "loaded", "rss_bytes": _rss_bytes()})
            while True:
                command = conn.recv()
                if command.get("type") == "close":
                    return
                if command.get("type") != "synthesize":
                    raise RuntimeError("invalid command")
                for chunk in voice.synthesize(command["text"], include_alignments=False):
                    conn.send(
                        {
                            "type": "chunk",
                            "pcm16": chunk.audio_int16_bytes,
                            "sample_rate": int(chunk.sample_rate),
                            "sample_width": int(chunk.sample_width),
                            "channels": int(chunk.sample_channels),
                            "rss_bytes": _rss_bytes(),
                        }
                    )
                conn.send({"type": "done", "rss_bytes": _rss_bytes()})
    except BaseException as exc:
        try:
            conn.send({"type": "error", "error_type": type(exc).__name__})
        except BaseException:
            pass
    finally:
        conn.close()


class ProcessPiperSession:
    """Long-lived Piper process that can be hard-stopped during ONNX inference."""

    def __init__(self, model: Path, config: Path) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self._connection = parent
        self._process = context.Process(
            target=_worker_main,
            args=(child, str(model), str(config)),
            name="jarvis-piper-f1-qualification",
            daemon=True,
        )
        self._peak_rss_bytes = 0
        self._closed = False
        self._process.start()
        child.close()
        message = self._receive(WORKER_START_TIMEOUT_SECONDS)
        if message.get("type") != "loaded":
            self.abort()
            raise _error("tts.piper_f1.runtime_load_failed", "Piper F1 worker failed to load")
        self._update_rss(message)

    @property
    def peak_rss_bytes(self) -> int:
        return self._peak_rss_bytes

    def _update_rss(self, message: dict[str, object]) -> None:
        value = message.get("rss_bytes")
        if isinstance(value, int):
            self._peak_rss_bytes = max(self._peak_rss_bytes, value)

    def _receive(self, timeout: float) -> dict[str, object]:
        if not self._connection.poll(timeout):
            raise _error("tts.piper_f1.worker_timeout", "Piper F1 worker timed out")
        try:
            message = self._connection.recv()
        except (EOFError, OSError) as exc:
            raise _error("tts.piper_f1.worker_stopped", "Piper F1 worker stopped") from exc
        if not isinstance(message, dict):
            raise _error("tts.piper_f1.worker_protocol_error", "Piper F1 worker response is invalid")
        if message.get("type") == "error":
            raise _error("tts.piper_f1.synthesis_failed", "Piper F1 synthesis failed")
        return message

    def stream(self, text: str) -> Iterator[PcmChunk]:
        if self._closed or not self._process.is_alive():
            raise _error("tts.piper_f1.worker_stopped", "Piper F1 worker is unavailable")
        self._connection.send({"type": "synthesize", "text": text})
        while True:
            message = self._receive(SYNTHESIS_TIMEOUT_SECONDS)
            self._update_rss(message)
            kind = message.get("type")
            if kind == "done":
                return
            if kind != "chunk":
                raise _error("tts.piper_f1.worker_protocol_error", "Piper F1 worker response is invalid")
            pcm = message.get("pcm16")
            if not isinstance(pcm, bytes) or not pcm:
                raise _error("tts.piper_f1.audio_invalid", "Piper F1 returned empty audio")
            yield PcmChunk(
                pcm16=pcm,
                sample_rate=int(message["sample_rate"]),
                sample_width=int(message["sample_width"]),
                channels=int(message["channels"]),
            )

    def abort(self) -> float:
        started = time.perf_counter()
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.0)
        self._closed = True
        self._connection.close()
        return (time.perf_counter() - started) * 1000.0

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.is_alive():
                self._connection.send({"type": "close"})
                self._process.join(timeout=2.0)
        except (BrokenPipeError, EOFError, OSError):
            pass
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._closed = True
        self._connection.close()


class SoundDevicePlaybackSink:
    """Optional real-output sink used only for the explicit cancellation gate."""

    def __init__(self) -> None:
        self._stream: object | None = None

    def open(self, chunk: PcmChunk) -> None:
        if chunk.channels != 1 or chunk.sample_width != 2:
            raise _error("tts.piper_f1.audio_format_invalid", "Piper F1 audio format is unsupported")
        try:
            import sounddevice
        except ImportError as exc:
            raise _error("tts.piper_f1.playback_runtime_missing", "sounddevice is unavailable") from exc
        self._stream = sounddevice.RawOutputStream(
            samplerate=chunk.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=0,
            latency="low",
        )
        self._stream.start()

    def write(self, pcm16: bytes) -> None:
        if self._stream is None:
            raise _error("tts.piper_f1.playback_not_open", "Playback stream is not open")
        self._stream.write(pcm16)

    def abort(self) -> None:
        if self._stream is not None:
            self._stream.abort()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def _write_wav(path: Path, chunks: list[PcmChunk]) -> tuple[int, float]:
    if not chunks:
        raise _error("tts.piper_f1.audio_missing", "Piper F1 produced no audio")
    first = chunks[0]
    if first.channels != 1 or first.sample_width != 2 or first.sample_rate != PIPER_TARGET_SAMPLE_RATE:
        raise _error("tts.piper_f1.audio_format_invalid", "Piper F1 audio format is unsupported")
    if any(
        item.channels != first.channels
        or item.sample_width != first.sample_width
        or item.sample_rate != first.sample_rate
        for item in chunks
    ):
        raise _error("tts.piper_f1.audio_format_changed", "Piper F1 audio format changed mid-stream")
    frames = sum(len(item.pcm16) for item in chunks) // (first.channels * first.sample_width)
    with path.open("xb") as raw:
        with wave.open(raw, "wb") as output:
            output.setnchannels(first.channels)
            output.setsampwidth(first.sample_width)
            output.setframerate(first.sample_rate)
            for item in chunks:
                output.writeframes(item.pcm16)
    os.chmod(path, 0o600)
    return frames, frames / first.sample_rate


def _prepare_output_directory(value: str | Path) -> Path:
    output = Path(value).expanduser()
    if not output.is_absolute() or output.is_symlink():
        raise _error("tts.piper_f1.output_invalid", "Qualification output directory is invalid")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise _error("tts.piper_f1.output_not_empty", "Qualification output directory must be empty")
    else:
        output.mkdir(mode=0o700, parents=True)
    os.chmod(output, 0o700)
    return output


def run_piper_f1_qualification(
    candidate_root: str | Path,
    output_directory: str | Path,
    *,
    session_factory: Callable[[Path, Path], SynthesisSession] | None = None,
    playback_sink_factory: Callable[[], PlaybackSink] | None = None,
    human_voice_approved: bool = False,
) -> dict[str, object]:
    """Run fixed offline fixtures and return an evidence-based promotion report."""

    candidate = inspect_piper_f1_candidate(candidate_root)
    if not candidate["ready"]:
        raise _error(str(candidate["code"]), "Piper F1 candidate is not ready")
    if session_factory is None and not _piper_runtime_ready():
        raise _error("tts.piper_f1.runtime_missing", "piper-tts==1.6.0 is unavailable")
    root = _candidate_root(candidate_root)
    output = _prepare_output_directory(output_directory)
    factory = ProcessPiperSession if session_factory is None else session_factory
    session = factory(root / MODEL_NAME, root / CONFIG_NAME)
    fixture_reports: list[dict[str, object]] = []
    cancellation_report: dict[str, object] = {
        "measured": False,
        "passed": False,
        "reason": "playback sink was not requested",
    }
    try:
        # Load time is deliberately excluded from the warm TTFA gate.
        list(session.stream(WARMUP_TEXT))
        for index, (label, text) in enumerate(FIXTURES, start=1):
            started = time.perf_counter()
            iterator = session.stream(text)
            try:
                first = next(iterator)
            except StopIteration as exc:
                raise _error("tts.piper_f1.audio_missing", "Piper F1 produced no audio") from exc
            first_audio_ms = (time.perf_counter() - started) * 1000.0
            chunks = [first, *iterator]
            wav_name = f"fixture-{index:02d}-{label}.wav"
            frames, duration = _write_wav(output / wav_name, chunks)
            fixture_reports.append(
                {
                    "index": index,
                    "label": label,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "wav": wav_name,
                    "frames": frames,
                    "duration_seconds": round(duration, 3),
                    "warm_first_audio_ms": round(first_audio_ms, 3),
                }
            )

        if playback_sink_factory is not None:
            iterator = session.stream(CANCELLATION_TEXT)
            try:
                first = next(iterator)
            except StopIteration as exc:
                raise _error("tts.piper_f1.audio_missing", "Piper F1 produced no cancellation audio") from exc
            sink = playback_sink_factory()
            writer_started = threading.Event()
            abort_requested = threading.Event()
            writer_failed = False

            def write_first_chunk() -> None:
                nonlocal writer_failed
                writer_started.set()
                try:
                    sink.write(first.pcm16)
                except BaseException as exc:
                    # PortAudio may release a blocked writer by raising after abort.
                    writer_failed = not abort_requested.is_set() or not isinstance(exc, Exception)

            try:
                sink.open(first)
                writer = threading.Thread(target=write_first_chunk, name="piper-f1-cancel-playback", daemon=True)
                writer.start()
                if not writer_started.wait(timeout=1.0):
                    raise _error("tts.piper_f1.playback_start_failed", "Playback did not start")
                # Leave enough audio queued for abort() to exercise an active stream.
                time.sleep(0.05)
                abort_requested.set()
                cancel_started = time.perf_counter()
                sink.abort()
                cancel_ms = (time.perf_counter() - cancel_started) * 1000.0
                writer.join(timeout=1.0)
                writer_stopped = not writer.is_alive()
            finally:
                sink.close()
            worker_stop_ms = session.abort()
            cancellation_report = {
                "measured": True,
                "measurement": "RawOutputStream.abort return latency",
                "cancel_to_silence_ms": round(cancel_ms, 3),
                "worker_stop_ms": round(worker_stop_ms, 3),
                "pre_abort_playback_ms": 50,
                "playback_write_failed": writer_failed,
                "playback_writer_stopped": writer_stopped,
                "passed": cancel_ms <= MAX_CANCEL_TO_SILENCE_MS and not writer_failed and writer_stopped,
                "physical_audio_hardware_validation_required": True,
            }
    finally:
        session.close()

    max_ttfa = max(float(item["warm_first_audio_ms"]) for item in fixture_reports)
    peak_rss = int(session.peak_rss_bytes)
    gates = {
        "model_size": int(candidate["model"]["size_bytes"]) <= MAX_EXPORTED_MODEL_BYTES,  # type: ignore[index]
        "peak_rss": peak_rss <= MAX_RUNTIME_RSS_BYTES,
        "warm_first_audio": max_ttfa <= MAX_WARM_FIRST_AUDIO_MS,
        "cancellation": bool(cancellation_report["passed"]),
        "fixtures": len(fixture_reports) == len(FIXTURES),
        "offline": True,
    }
    automated_pass = all(gates.values())
    return {
        "schema": "jarvis.voice.piper-f1-qualification/1",
        # Candidate metrics are necessary but never replace native acoustic,
        # 200-turn, recovery, and soak evidence for production promotion.
        "ready_for_promotion": False,
        "automated_pass": automated_pass,
        "code": (
            "tts.piper_f1.native_qualification_required"
            if automated_pass and human_voice_approved
            else "tts.piper_f1.human_approval_required"
            if automated_pass
            else "tts.piper_f1.qualification_failed"
        ),
        "candidate": candidate,
        "fixtures": fixture_reports,
        "metrics": {
            "peak_rss_bytes": peak_rss,
            "max_warm_first_audio_ms": round(max_ttfa, 3),
            "cancellation": cancellation_report,
        },
        "gates": gates,
        "human_voice_approved": human_voice_approved,
        "native_acceptance_required": True,
        "network_used": False,
        "installation_performed": False,
        "production_backend_changed": False,
    }


def write_qualification_report(path: str | Path, report: dict[str, object]) -> None:
    target = Path(path).expanduser()
    if not target.is_absolute() or target.is_symlink() or target.exists():
        raise _error("tts.piper_f1.report_invalid", "Qualification report path is invalid")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(target, 0o600)
