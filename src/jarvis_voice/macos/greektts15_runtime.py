"""Epoch-safe runtime boundary for the approved GreekTTS F1 voice."""

from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing
import os
import queue
import re
import threading
import time
from array import array
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..backends import SpeechUnit
from ..errors import BackendError
from .greektts15_pilot import (
    END_OF_HUMAN,
    END_OF_SPEECH,
    END_OF_TEXT,
    SAMPLE_RATE,
    SNAC_DIRECTORY,
    START_OF_HUMAN,
    START_OF_SPEECH,
    _native_adapter_config,
    extract_snac_codes,
    validate_greektts_model_root,
)


APPROVED_CANDIDATE_SCHEMA = "jarvis.voice.greektts15-approved-candidate/1"
SNAC_FRAME_TOKENS = 7
SNAC_FRAME_SAMPLES = 2_048
INCREMENTAL_START_FRAMES = 36
INCREMENTAL_STEP_FRAMES = 4
INCREMENTAL_LOOKAHEAD_FRAMES = 4
INCREMENTAL_BOUNDARY_RAMP_SAMPLES = 120
COREAUDIO_WRITE_BLOCK_SAMPLES = 480
DECODER_DRAIN_TIMEOUT_SECONDS = 15.0
DECODER_DRAIN_POLL_SECONDS = 0.1
PLAYBACK_HELPER_READY = b"R"
PLAYBACK_HELPER_DATA = b"D"
PLAYBACK_HELPER_FINISH = b"F"
PLAYBACK_HELPER_DONE = b"C"
PLAYBACK_HELPER_ERROR = b"E"


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


def load_digest_bound_greektts_candidate(
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    repository_root: str | Path,
) -> ApprovedGreekTtsCandidate:
    """Verify the approval document itself before resolving its checkpoint."""

    manifest = Path(manifest_path).expanduser()
    if (
        not manifest.is_absolute()
        or manifest.is_symlink()
        or not manifest.is_file()
        or re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None
        or _sha256(manifest) != expected_manifest_sha256
    ):
        raise BackendError("tts.greektts15.approval_manifest_changed", "Approved voice manifest changed", subsystem="tts")
    return load_approved_greektts_candidate(manifest, repository_root)


@dataclass(frozen=True, slots=True)
class PcmChunk:
    pcm16: bytes
    sample_rate: int = 24_000
    channels: int = 1


def incremental_pcm_slice(
    cumulative_pcm16: bytes,
    *,
    emitted_samples: int,
    stable_samples: int,
    previous_sample: int | None,
    ramp_samples: int = INCREMENTAL_BOUNDARY_RAMP_SAMPLES,
) -> tuple[bytes, int, int | None]:
    """Return only a stable PCM suffix and smooth cumulative-decode boundaries."""

    if len(cumulative_pcm16) % 2 or emitted_samples < 0 or stable_samples < emitted_samples or ramp_samples < 0:
        raise BackendError("tts.greektts15.incremental_pcm_invalid", "Incremental PCM bounds are invalid", subsystem="tts")
    samples = array("h")
    samples.frombytes(cumulative_pcm16)
    if stable_samples > len(samples):
        raise BackendError("tts.greektts15.incremental_pcm_invalid", "Stable PCM exceeds decoded audio", subsystem="tts")
    chunk = array("h", samples[emitted_samples:stable_samples])
    if chunk and previous_sample is not None and ramp_samples:
        count = min(ramp_samples, len(chunk))
        for index in range(count):
            alpha = (index + 1) / count
            chunk[index] = round(previous_sample * (1.0 - alpha) + chunk[index] * alpha)
    last = int(chunk[-1]) if chunk else previous_sample
    return chunk.tobytes(), stable_samples, last


def wait_for_decoded_result(
    decoded: queue.Queue[PcmChunk | Exception | None],
    decoder: threading.Thread,
    cancelled: threading.Event,
    *,
    deadline: float,
    poll_seconds: float = DECODER_DRAIN_POLL_SECONDS,
) -> PcmChunk | Exception | None:
    """Wait through transient queue gaps while keeping final decode bounded."""

    if poll_seconds <= 0:
        raise ValueError("Decoder poll interval must be positive")
    while not cancelled.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BackendError(
                "tts.greektts15.decoder_timeout",
                "GreekTTS decoder did not finish in time",
                subsystem="tts",
            )
        try:
            return decoded.get(timeout=min(poll_seconds, remaining))
        except queue.Empty:
            if not decoder.is_alive():
                raise BackendError(
                    "tts.greektts15.decoder_stopped",
                    "GreekTTS decoder stopped without a terminal result",
                    subsystem="tts",
                )
    return None


class GreekTtsStreamingEngine(Protocol):
    def warm(self) -> float: ...
    def stream(self, unit: SpeechUnit, cancelled: threading.Event) -> Iterable[PcmChunk]: ...
    def cancel(self) -> None: ...


class PcmPlayback(Protocol):
    def write(self, chunk: PcmChunk) -> None: ...
    def finish(self) -> None: ...
    def abort(self) -> None: ...


class MlxGreekTtsStreamingEngine:
    """Warm, offline MLX/SNAC engine with token-boundary cancellation checks."""

    def __init__(
        self,
        *,
        model_root: str | Path,
        candidate: ApprovedGreekTtsCandidate,
        seed: int = 10_001,
        maximum_new_tokens: int = 1_200,
    ) -> None:
        root = Path(model_root).expanduser()
        if (
            not root.is_absolute()
            or root.is_symlink()
            or not root.is_dir()
            or not validate_greektts_model_root(root).ready
            or json.loads((candidate.checkpoint_directory / "adapter_config.json").read_text(encoding="utf-8"))
            != _native_adapter_config()
        ):
            raise BackendError("tts.greektts15.runtime_profile_invalid", "GreekTTS runtime profile is invalid", subsystem="tts")
        if maximum_new_tokens < 7 or maximum_new_tokens > 1_200:
            raise BackendError("tts.greektts15.runtime_profile_invalid", "GreekTTS token bound is invalid", subsystem="tts")
        self.model_root = root
        self.candidate = candidate
        self.seed = seed
        self.maximum_new_tokens = maximum_new_tokens
        self._cancel_requested = threading.Event()
        self._decoder_quarantined = threading.Event()
        self._load_lock = threading.Lock()
        self._runtime: tuple[object, ...] | None = None
        self._warm_load_ms: float | None = None

    def _ensure_loaded(self) -> tuple[object, ...]:
        with self._load_lock:
            if self._runtime is not None:
                return self._runtime
            started = time.perf_counter()
            try:
                import mlx.core as mx
                import torch
                from mlx_lm import load
                from mlx_lm.generate import generate_step
                from mlx_lm.sample_utils import make_logits_processors, make_sampler
                from snac import SNAC
            except ImportError as exc:
                raise BackendError("tts.greektts15.dependencies_missing", "GreekTTS runtime dependencies are missing", subsystem="tts") from exc
            previous = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
            os.environ.update({name: "1" for name in previous})
            try:
                model, tokenizer = load(
                    str(self.model_root / "base"),
                    adapter_path=str(self.candidate.checkpoint_directory),
                    lazy=False,
                )
                mx.synchronize()
                snac_model = SNAC.from_pretrained(str(self.model_root / SNAC_DIRECTORY)).to("cpu").eval()
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
            self._runtime = (
                mx,
                torch,
                model,
                tokenizer,
                snac_model,
                generate_step,
                make_logits_processors,
                make_sampler,
            )
            self._warm_load_ms = (time.perf_counter() - started) * 1000.0
            return self._runtime

    def warm(self) -> float:
        """Materialize the offline model once and return its initial load time."""

        self._ensure_loaded()
        return float(self._warm_load_ms or 0.0)

    def stream(self, unit: SpeechUnit, cancelled: threading.Event) -> Iterable[PcmChunk]:
        self._cancel_requested.clear()
        if self._decoder_quarantined.is_set():
            raise BackendError(
                "tts.greektts15.decoder_quarantined",
                "GreekTTS decoder did not stop cleanly",
                subsystem="tts",
            )
        if cancelled.is_set():
            return
        mx, torch, model, tokenizer, snac_model, generate_step, make_logits_processors, make_sampler = self._ensure_loaded()
        if cancelled.is_set() or self._cancel_requested.is_set():
            return
        mx.random.seed(self.seed)
        text_ids = list(tokenizer.encode(unit.text, add_special_tokens=True))
        if not text_ids or text_ids[0] != 128_000:
            raise BackendError("tts.greektts15.tokenizer_invalid", "Tokenizer did not emit start-of-text", subsystem="tts")
        prompt_ids = [START_OF_HUMAN, *text_ids, END_OF_TEXT, END_OF_HUMAN]
        sampler = make_sampler(temp=self.candidate.temperature, top_p=self.candidate.top_p)
        processors = make_logits_processors(repetition_penalty=self.candidate.repetition_penalty)
        generated: list[int] = []
        speech_started = False
        audio_tokens = 0
        emitted_samples = 0
        previous_sample: int | None = None
        decode_jobs: queue.Queue[tuple[list[int], int] | None] = queue.Queue(maxsize=2)
        decoded: queue.Queue[PcmChunk | Exception | None] = queue.Queue(maxsize=8)
        decoder_cancelled = threading.Event()

        def publish_decoded(value: PcmChunk | Exception | None) -> bool:
            while not decoder_cancelled.is_set():
                try:
                    decoded.put(value, timeout=0.02)
                    return True
                except queue.Full:
                    continue
            return False

        def decode_increment(snapshot: list[int], stable_frames: int) -> PcmChunk | None:
            nonlocal emitted_samples, previous_sample
            code_layers = extract_snac_codes(snapshot)
            with torch.inference_mode():
                codes = [torch.tensor(layer, dtype=torch.long).unsqueeze(0) for layer in code_layers]
                waveform = snac_model.decode(codes)
            tensor = waveform.detach().squeeze().to("cpu", dtype=torch.float32)
            if tensor.ndim != 1 or tensor.numel() == 0 or not bool(torch.isfinite(tensor).all().item()):
                raise BackendError("tts.greektts15.audio_invalid", "SNAC produced invalid audio", subsystem="tts")
            pcm = tensor.clamp(-0.95, 0.95).mul(32_767.0).round().to(torch.int16).numpy().tobytes()
            stable_samples = min(stable_frames * SNAC_FRAME_SAMPLES, int(tensor.numel()))
            chunk, emitted_samples, previous_sample = incremental_pcm_slice(
                pcm,
                emitted_samples=emitted_samples,
                stable_samples=stable_samples,
                previous_sample=previous_sample,
            )
            del codes, waveform, tensor
            return PcmChunk(chunk, SAMPLE_RATE, 1) if chunk else None

        def decoder_worker() -> None:
            try:
                while not decoder_cancelled.is_set():
                    try:
                        job = decode_jobs.get(timeout=0.02)
                    except queue.Empty:
                        continue
                    if job is None:
                        break
                    snapshot, stable_frames = job
                    chunk = decode_increment(snapshot, stable_frames)
                    if chunk is not None and not publish_decoded(chunk):
                        return
            except Exception as exc:
                publish_decoded(exc)
            finally:
                publish_decoded(None)

        decoder = threading.Thread(target=decoder_worker, name="jarvis-greektts-snac", daemon=True)
        decoder.start()
        decoder_finished = False
        generation = generate_step(
            mx.array(prompt_ids),
            model,
            max_tokens=self.maximum_new_tokens,
            sampler=sampler,
            logits_processors=processors,
        )
        try:
            for token, _ in generation:
                if cancelled.is_set() or self._cancel_requested.is_set():
                    return
                token_id = int(token if isinstance(token, int) else token.item())
                generated.append(token_id)
                if token_id == START_OF_SPEECH:
                    speech_started = True
                elif token_id == END_OF_SPEECH:
                    break
                elif speech_started:
                    audio_tokens += 1
                    if audio_tokens % SNAC_FRAME_TOKENS == 0:
                        frames = audio_tokens // SNAC_FRAME_TOKENS
                        if (
                            frames >= INCREMENTAL_START_FRAMES
                            and (frames - INCREMENTAL_START_FRAMES) % INCREMENTAL_STEP_FRAMES == 0
                        ):
                            stable_frames = frames - INCREMENTAL_LOOKAHEAD_FRAMES
                            try:
                                decode_jobs.put_nowait((list(generated), stable_frames))
                            except queue.Full:
                                pass
                while True:
                    try:
                        result = decoded.get_nowait()
                    except queue.Empty:
                        break
                    if result is None:
                        decoder_finished = True
                    elif isinstance(result, Exception):
                        raise result
                    else:
                        yield result
            mx.synchronize()
            if cancelled.is_set() or self._cancel_requested.is_set():
                return
            if not generated or START_OF_SPEECH not in generated or generated[-1] != END_OF_SPEECH:
                raise BackendError("tts.greektts15.end_of_speech_missing", "GreekTTS reached its sentence token limit", subsystem="tts")
            final_frames = audio_tokens // SNAC_FRAME_TOKENS
            while not cancelled.is_set() and not self._cancel_requested.is_set():
                try:
                    decode_jobs.put((list(generated), final_frames), timeout=0.02)
                    break
                except queue.Full:
                    continue
            while decoder.is_alive():
                try:
                    decode_jobs.put(None, timeout=0.02)
                    break
                except queue.Full:
                    continue
            decoder_deadline = time.monotonic() + DECODER_DRAIN_TIMEOUT_SECONDS
            while not decoder_finished:
                result = wait_for_decoded_result(
                    decoded,
                    decoder,
                    decoder_cancelled,
                    deadline=decoder_deadline,
                )
                if result is None:
                    decoder_finished = True
                elif isinstance(result, Exception):
                    raise result
                else:
                    yield result
        finally:
            close_generation = getattr(generation, "close", None)
            if callable(close_generation):
                close_generation()
            decoder_cancelled.set()
            while True:
                try:
                    decoded.get_nowait()
                except queue.Empty:
                    break
            try:
                decode_jobs.put_nowait(None)
            except queue.Full:
                pass
            decoder.join(timeout=15.0)
            if decoder.is_alive():
                self._decoder_quarantined.set()
            mx.synchronize()
            gc.collect()
            mx.clear_cache()

    def cancel(self) -> None:
        self._cancel_requested.set()


class SoundDevicePcmPlayback:
    """One low-latency CoreAudio PCM stream with serialized native calls."""

    def __init__(self, sounddevice_module: object | None = None) -> None:
        if sounddevice_module is None:
            try:
                import sounddevice as sounddevice_module
            except ImportError as exc:
                raise BackendError("tts.playback_dependency_missing", "sounddevice is unavailable", subsystem="tts") from exc
        self._lock = threading.Lock()
        self._abort_requested = threading.Event()
        self._closed = False
        settings = sounddevice_module.CoreAudioSettings(
            change_device_parameters=False,
            fail_if_conversion_required=False,
            conversion_quality="max",
        )
        self._stream = sounddevice_module.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=0,
            latency="low",
            extra_settings=settings,
        )
        self._stream.start()

    def write(self, chunk: PcmChunk) -> None:
        if chunk.sample_rate != SAMPLE_RATE or chunk.channels != 1 or not chunk.pcm16 or len(chunk.pcm16) % 2:
            raise BackendError("tts.audio_chunk_invalid", "Playback chunk format is invalid", subsystem="tts")
        block_bytes = COREAUDIO_WRITE_BLOCK_SAMPLES * 2
        for offset in range(0, len(chunk.pcm16), block_bytes):
            if self._abort_requested.is_set():
                return
            with self._lock:
                if self._closed or self._abort_requested.is_set():
                    return
                underflowed = self._stream.write(chunk.pcm16[offset : offset + block_bytes])
            if underflowed:
                raise BackendError("tts.playback_underflow", "CoreAudio playback underflowed", subsystem="tts")

    def finish(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stream.stop()
            self._stream.close()

    def abort(self) -> None:
        self._abort_requested.set()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stream.abort()
            self._stream.close()


def _coreaudio_playback_worker(connection: object) -> None:
    """Own PortAudio entirely inside a disposable spawned process."""

    playback: SoundDevicePcmPlayback | None = None
    try:
        playback = SoundDevicePcmPlayback()
        connection.send_bytes(PLAYBACK_HELPER_READY)
        while True:
            message = connection.recv_bytes()
            kind, payload = message[:1], message[1:]
            if kind == PLAYBACK_HELPER_DATA and payload:
                playback.write(PcmChunk(payload))
            elif kind == PLAYBACK_HELPER_FINISH and not payload:
                playback.finish()
                playback = None
                connection.send_bytes(PLAYBACK_HELPER_DONE)
                return
            else:
                raise BackendError("tts.playback_protocol_invalid", "Playback helper message is invalid", subsystem="tts")
    except EOFError:
        pass
    except Exception:
        try:
            connection.send_bytes(PLAYBACK_HELPER_ERROR)
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if playback is not None:
            playback.abort()
        connection.close()


class IsolatedSoundDevicePcmPlayback:
    """Proxy PCM to a disposable process so hard abort cannot crash the client."""

    def __init__(
        self,
        *,
        process_context: object | None = None,
        startup_timeout_seconds: float = 5.0,
        shutdown_timeout_seconds: float = 10.0,
    ) -> None:
        if startup_timeout_seconds <= 0 or shutdown_timeout_seconds <= 0:
            raise ValueError("Playback helper timeouts must be positive")
        context = process_context or multiprocessing.get_context("spawn")
        self._connection, child_connection = context.Pipe(duplex=True)
        self._process = context.Process(
            target=_coreaudio_playback_worker,
            args=(child_connection,),
            name="jarvis-coreaudio-playback",
            daemon=True,
        )
        self._aborted = threading.Event()
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._process.start()
        child_connection.close()
        if not self._connection.poll(startup_timeout_seconds):
            self.abort()
            raise BackendError("tts.playback_helper_start_timeout", "CoreAudio helper did not start", subsystem="tts")
        try:
            message = self._connection.recv_bytes()
        except (EOFError, OSError) as exc:
            self.abort()
            raise BackendError("tts.playback_helper_start_failed", "CoreAudio helper failed to start", subsystem="tts") from exc
        if message != PLAYBACK_HELPER_READY:
            self.abort()
            raise BackendError("tts.playback_helper_start_failed", "CoreAudio helper failed to start", subsystem="tts")

    def write(self, chunk: PcmChunk) -> None:
        if chunk.sample_rate != SAMPLE_RATE or chunk.channels != 1 or not chunk.pcm16 or len(chunk.pcm16) % 2:
            raise BackendError("tts.audio_chunk_invalid", "Playback chunk format is invalid", subsystem="tts")
        if self._aborted.is_set():
            return
        try:
            self._connection.send_bytes(PLAYBACK_HELPER_DATA + chunk.pcm16)
        except (BrokenPipeError, EOFError, OSError) as exc:
            if self._aborted.is_set():
                return
            raise BackendError("tts.playback_helper_failed", "CoreAudio helper stopped", subsystem="tts") from exc

    def finish(self) -> None:
        if self._aborted.is_set():
            return
        try:
            self._connection.send_bytes(PLAYBACK_HELPER_FINISH)
            if not self._connection.poll(self._shutdown_timeout_seconds):
                raise BackendError("tts.playback_helper_finish_timeout", "CoreAudio helper did not finish", subsystem="tts")
            message = self._connection.recv_bytes()
            if message != PLAYBACK_HELPER_DONE:
                raise BackendError("tts.playback_helper_failed", "CoreAudio helper stopped", subsystem="tts")
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                raise BackendError("tts.playback_helper_exit_timeout", "CoreAudio helper did not exit", subsystem="tts")
            self._connection.close()
        except BackendError:
            self.abort()
            raise
        except (BrokenPipeError, EOFError, OSError) as exc:
            self.abort()
            raise BackendError("tts.playback_helper_failed", "CoreAudio helper stopped", subsystem="tts") from exc

    def abort(self) -> None:
        if self._aborted.is_set():
            return
        self._aborted.set()
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=1.0)
        self._connection.close()

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
        warm_on_start: bool = False,
        warm_timeout_seconds: float = 60.0,
    ) -> None:
        if queue_capacity < 1 or maximum_text_bytes < 1 or warm_timeout_seconds <= 0:
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
        self._warm_on_start = warm_on_start
        self._warm_ready = threading.Event()
        self._warm_load_ms: float | None = None
        self._thread = threading.Thread(target=self._worker, name="jarvis-greektts-f1", daemon=True)
        self._thread.start()
        if warm_on_start:
            if not self._warm_ready.wait(warm_timeout_seconds):
                self.close()
                raise BackendError("tts.greektts15.warm_timeout", "GreekTTS warm load timed out", subsystem="tts")
            if self._error is not None:
                error = self._error
                self.close()
                raise error

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

    def close(self) -> None:
        self._stop.set()
        self.cancel(2**63 - 1, "application_close")
        self._thread.join(timeout=2.0)

    def _is_stale(self, unit: SpeechUnit, cancelled: threading.Event) -> bool:
        with self._lock:
            return self._stop.is_set() or cancelled.is_set() or unit.epoch <= self._cancelled_through_epoch

    def _play_chunks(
        self,
        unit: SpeechUnit,
        cancelled: threading.Event,
        playback: PcmPlayback,
        chunks: queue.Queue[PcmChunk | None],
        started: threading.Event,
        errors: list[Exception],
    ) -> None:
        try:
            while not self._is_stale(unit, cancelled):
                chunk = chunks.get()
                if chunk is None:
                    playback.finish()
                    return
                if not started.is_set():
                    self.on_started(unit)
                    started.set()
                playback.write(chunk)
        except Exception as exc:
            if not self._is_stale(unit, cancelled):
                errors.append(exc)
            cancelled.set()
            playback.abort()

    @staticmethod
    def _queue_playback_item(
        chunks: queue.Queue[PcmChunk | None],
        item: PcmChunk | None,
        writer: threading.Thread,
        cancelled: threading.Event,
        *,
        allow_cancelled: bool = False,
    ) -> bool:
        while writer.is_alive() and (allow_cancelled or not cancelled.is_set()):
            try:
                chunks.put(item, timeout=0.02)
                return True
            except queue.Full:
                continue
        return False

    def _worker(self) -> None:
        if self._warm_on_start:
            try:
                self._warm_load_ms = self.engine.warm()
            except Exception as exc:
                self._error = BackendError("tts.greektts15.warm_failed", "GreekTTS warm load failed", subsystem="tts")
                self._error.__cause__ = exc
                self._warm_ready.set()
                return
            self._warm_ready.set()
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
            playback_chunks: queue.Queue[PcmChunk | None] | None = None
            playback_thread: threading.Thread | None = None
            playback_started = threading.Event()
            playback_errors: list[Exception] = []
            try:
                for chunk in self.engine.stream(unit, cancelled):
                    if self._is_stale(unit, cancelled):
                        break
                    if not chunk.pcm16 or chunk.sample_rate != 24_000 or chunk.channels != 1:
                        raise BackendError("tts.audio_chunk_invalid", "GreekTTS emitted an invalid PCM chunk", subsystem="tts")
                    if playback is None:
                        playback = self.playback_factory()
                        playback_chunks = queue.Queue(maxsize=8)
                        playback_thread = threading.Thread(
                            target=self._play_chunks,
                            args=(unit, cancelled, playback, playback_chunks, playback_started, playback_errors),
                            name="jarvis-greektts-playback",
                            daemon=True,
                        )
                        with self._lock:
                            self._playback = playback
                        playback_thread.start()
                    if self._is_stale(unit, cancelled):
                        playback.abort()
                        break
                    assert playback_chunks is not None and playback_thread is not None
                    if not self._queue_playback_item(playback_chunks, chunk, playback_thread, cancelled):
                        break
                stale = self._is_stale(unit, cancelled)
                if playback_thread is not None and playback_chunks is not None:
                    self._queue_playback_item(
                        playback_chunks,
                        None,
                        playback_thread,
                        cancelled,
                        allow_cancelled=True,
                    )
                    playback_thread.join(timeout=10.0)
                    if playback_thread.is_alive():
                        playback.abort()
                        raise BackendError("tts.playback_timeout", "GreekTTS playback did not drain", subsystem="tts")
                if playback_errors:
                    raise playback_errors[0]
                stale = self._is_stale(unit, cancelled)
                if playback_started.is_set() and not stale:
                    self.on_completed(unit)
            except Exception as exc:
                playback_unsafe = isinstance(exc, BackendError) and exc.code == "tts.playback_timeout"
                if playback_unsafe or not self._is_stale(unit, cancelled):
                    self._error = BackendError("tts.greektts15.runtime_failed", "GreekTTS runtime failed", subsystem="tts")
                    self._error.__cause__ = exc
                if playback is not None:
                    playback.abort()
            finally:
                with self._lock:
                    self._playback = None
                    self._active_cancel = None
                    self._active = None
