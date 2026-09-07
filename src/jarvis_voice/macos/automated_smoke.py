"""Deterministic native macOS ASR/VAD/TTS qualification without user content."""

from __future__ import annotations

import queue
import subprocess
import tempfile
import time
import wave
from array import array
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from contextlib import ExitStack
from pathlib import Path

from ..backends import AudioChunk, SpeechUnit
from ..config import VoiceConfig
from ..core import VoiceCore
from ..errors import BackendError
from ..fakes import FakeOrchestrator
from ..qualification import write_report
from ..vad import SileroOnnxVad
from .asr_selection import create_macos_asr_backend
from .system_tts import MacSystemTtsBackend


AUTOMATED_SMOKE_SCHEMA_VERSION = 1
CASE_ORDER = ("greek_fixture", "english_fixture", "silence_fixture", "programmatic_barge_in")
FIXTURE_TEXT: Mapping[str, str] = {
    "el-GR": "Καλημέρα. Αυτή είναι μια δοκιμή του συστήματος φωνής.",
    "en-US": "Good morning. This is a test of the voice system.",
}
RESPONSE_TEXT: Mapping[str, tuple[str, str]] = {
    "el-GR": ("Η ελληνική δοκιμή ολοκληρώθηκε.", "Το σύστημα είναι έτοιμο."),
    "en-US": ("The English test is complete.", "The system is ready."),
}
BARGE_TEXT = (
    "Αυτή είναι μια εσκεμμένα μεγάλη πρόταση που πρέπει να διακοπεί αμέσως από το νέο push to talk epoch πριν ολοκληρωθεί η αναπαραγωγή.",
    "Αυτή η δεύτερη μονάδα δεν πρέπει να αναπαραχθεί.",
)


def _milliseconds(later: float, earlier: float) -> float:
    return round(max(0.0, (later - earlier) * 1000.0), 3)


def read_pcm16_wav(path: str | Path, *, maximum_seconds: float = 15.0) -> AudioChunk:
    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file() or source.stat().st_size > 32 * 1024 * 1024:
            raise BackendError("smoke.fixture_file_invalid", "Fixture WAV is invalid", subsystem="smoke")
        with wave.open(str(source), "rb") as stream:
            channels = stream.getnchannels()
            width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            frames = stream.getnframes()
            if channels != 1 or width != 2 or sample_rate != 16_000 or not 1 <= frames <= round(sample_rate * maximum_seconds):
                raise BackendError("smoke.fixture_format_invalid", "Fixture must be bounded mono PCM16 at 16 kHz", subsystem="smoke")
            raw = stream.readframes(frames)
            if len(raw) != frames * width:
                raise BackendError("smoke.fixture_truncated", "Fixture WAV is truncated", subsystem="smoke")
    except BackendError:
        raise
    except (OSError, EOFError, wave.Error) as exc:
        raise BackendError("smoke.fixture_read_failed", "Fixture WAV could not be read", subsystem="smoke") from exc
    integers = array("h")
    integers.frombytes(raw)
    samples = array("f", (value / 32768.0 for value in integers))
    return AudioChunk(samples, sample_rate)


class MacSpeechFixtureSynthesizer:
    """Generate fixed test audio through content-private stdin and bounded files."""

    def __init__(
        self,
        *,
        process_runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
        say_executable: str = "/usr/bin/say",
        afconvert_executable: str = "/usr/bin/afconvert",
    ) -> None:
        self.process_runner = process_runner
        self.say_executable = say_executable
        self.afconvert_executable = afconvert_executable

    def synthesize(self, *, locale: str, voice: str) -> AudioChunk:
        text = FIXTURE_TEXT.get(locale)
        if text is None or not voice:
            raise BackendError("smoke.fixture_locale_invalid", "Fixture locale or voice is invalid", subsystem="smoke")
        with tempfile.TemporaryDirectory(prefix="jarvis-m5e-a-") as temporary:
            root = Path(temporary)
            aiff = root / "fixture.aiff"
            wav = root / "fixture.wav"
            try:
                spoken = self.process_runner(
                    [self.say_executable, "-v", voice, "-o", str(aiff), "-f", "-"],
                    input=text.encode("utf-8"),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30.0,
                    check=False,
                    shell=False,
                )
                if spoken.returncode != 0:
                    raise BackendError("smoke.fixture_synthesis_failed", "System fixture synthesis failed", subsystem="smoke")
                converted = self.process_runner(
                    [self.afconvert_executable, str(aiff), str(wav), "-f", "WAVE", "-d", "LEI16@16000", "-c", "1"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30.0,
                    check=False,
                    shell=False,
                )
                if converted.returncode != 0:
                    raise BackendError("smoke.fixture_conversion_failed", "System fixture conversion failed", subsystem="smoke")
                return read_pcm16_wav(wav)
            except BackendError:
                raise
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise BackendError("smoke.fixture_process_failed", "System fixture process failed", subsystem="smoke") from exc


@dataclass(frozen=True, slots=True)
class TtsCallback:
    kind: str
    unit: SpeechUnit
    timestamp: float


def _vad_from_config(config: VoiceConfig) -> SileroOnnxVad:
    artifact = config.vad.artifact
    if not artifact.path or artifact.size is None or not artifact.sha256:
        raise BackendError("vad.artifact_manifest_required", "Silero artifact manifest is required", subsystem="vad")
    return SileroOnnxVad(
        artifact_path=artifact.path,
        size=artifact.size,
        sha256=artifact.sha256,
        threshold=config.vad.threshold,
        minimum_speech_ms=config.vad.minimum_speech_ms,
        padding_ms=config.vad.padding_ms,
    )


def _await_completed_playback(
    core: VoiceCore,
    tts: MacSystemTtsBackend,
    callbacks: queue.Queue[TtsCallback],
    *,
    last_unit_index: int,
    deadline_seconds: float = 45.0,
) -> tuple[float, float]:
    deadline = time.monotonic() + deadline_seconds
    first_started: float | None = None
    completed_at: float | None = None
    while time.monotonic() < deadline:
        try:
            callback = callbacks.get(timeout=0.05)
        except queue.Empty:
            if tts.last_error is not None:
                raise tts.last_error
            continue
        if callback.kind == "started":
            first_started = callback.timestamp if first_started is None else first_started
            outbound = core.on_tts_unit_started(callback.unit)
        else:
            if callback.unit.unit_index == last_unit_index and not tts.wait_idle(timeout=2.0):
                raise BackendError("smoke.tts_not_idle", "System TTS did not become idle", subsystem="smoke")
            outbound = core.on_tts_unit_completed(callback.unit)
        if any(event.type == "playback.completed" for event in outbound):
            completed_at = time.perf_counter()
            break
    if first_started is None or completed_at is None or not tts.is_idle:
        raise BackendError("smoke.playback_incomplete", "Fixture playback did not complete", subsystem="smoke")
    return first_started, completed_at


def _base_report(cases: Mapping[str, Mapping[str, object]], *, ready: bool, code: str, tts_exercised: bool) -> dict[str, object]:
    return {
        "schema_version": AUTOMATED_SMOKE_SCHEMA_VERSION,
        "milestone": "5E-A",
        "ready": ready,
        "code": code,
        "cases": {name: dict(cases[name]) for name in CASE_ORDER},
        "native": {
            "fixture_injected": True,
            "microphone_opened": False,
            "physical_ptt_exercised": False,
            "system_tts_exercised": tts_exercised,
            "satisfies_interactive_m5e": False,
        },
        "privacy": {
            "temporary_fixture_files_retained": False,
            "raw_audio_persisted": False,
            "transcript_text_persisted": False,
            "tts_text_persisted_in_report": False,
            "protocol_ids_persisted": False,
            "content_captured_in_report": False,
        },
        "network_used": False,
    }


def run_automated_macos_smoke(
    config: VoiceConfig,
    *,
    output: str,
    synthesizer: MacSpeechFixtureSynthesizer | None = None,
) -> dict[str, object]:
    cases: dict[str, dict[str, object]] = {name: {"status": "pending", "metrics": {}} for name in CASE_ORDER}
    current_case = CASE_ORDER[0]
    tts_exercised = False
    callbacks: queue.Queue[TtsCallback] = queue.Queue()
    try:
        with ExitStack() as cleanup:
            tts = MacSystemTtsBackend(
                voice_mapping=config.macos.system_voices,
                on_started=lambda unit: callbacks.put_nowait(TtsCallback("started", unit, time.perf_counter())),
                on_completed=lambda unit: callbacks.put_nowait(TtsCallback("completed", unit, time.perf_counter())),
            )
            cleanup.callback(tts.close)
            asr = create_macos_asr_backend(config)
            close_asr = getattr(asr, "close", None)
            if close_asr is not None:
                cleanup.callback(close_asr)
            core = VoiceCore(asr=asr, tts=tts)
            vad = _vad_from_config(config)
            orchestrator = FakeOrchestrator(session_id=core.session_id)
            fixtures = synthesizer or MacSpeechFixtureSynthesizer()
            for current_case, locale, expected_language in (
                ("greek_fixture", "el-GR", "el"),
                ("english_fixture", "en-US", "en"),
            ):
                voice = config.macos.system_voices.get(locale, "")
                audio = fixtures.synthesize(locale=locale, voice=voice)
                core.ptt_down()
                started = time.perf_counter()
                transcript = core.ptt_up_and_process(audio, vad)
                transcript_at = time.perf_counter()
                if transcript.type != "input.transcript.final":
                    cases[current_case] = {
                        "status": "blocked",
                        "code": "smoke.fixture_vad_no_speech",
                        "vad_speech_detected": False,
                        "metrics": {"fixture_vad_ms": _milliseconds(transcript_at, started)},
                    }
                    raise BackendError("smoke.fixture_vad_no_speech", "Speech fixture was rejected by VAD", subsystem="smoke")
                language = transcript.payload.get("language", {})
                language_code = language.get("code") if isinstance(language, Mapping) else None
                if language_code != expected_language:
                    cases[current_case] = {
                        "status": "blocked",
                        "code": "smoke.fixture_language_mismatch",
                        "detected_language": str(language_code) if language_code else "unknown",
                        "expected_language": expected_language,
                        "vad_speech_detected": True,
                        "metrics": {"fixture_to_transcript_final_ms": _milliseconds(transcript_at, started)},
                    }
                    raise BackendError("smoke.fixture_language_mismatch", "Fixture ASR language did not match", subsystem="smoke")
                for reply in orchestrator.accept(transcript):
                    core.handle_server_event(reply)
                commit_at: float | None = None
                speech_done_at: float | None = None
                response = orchestrator.script_response(
                    RESPONSE_TEXT[locale],
                    locale=locale,
                    turn_id=transcript.turn_id,
                    interleave_speech=True,
                )
                for server_event in response.events:
                    if server_event.type == "response.speech.commit" and commit_at is None:
                        commit_at = time.perf_counter()
                    elif server_event.type == "response.speech.done":
                        speech_done_at = time.perf_counter()
                    core.handle_server_event(server_event)
                if commit_at is None or speech_done_at is None:
                    raise BackendError("smoke.fixture_protocol_incomplete", "Fixture speech protocol was incomplete", subsystem="smoke")
                tts_started_at, playback_completed_at = _await_completed_playback(
                    core,
                    tts,
                    callbacks,
                    last_unit_index=len(RESPONSE_TEXT[locale]),
                )
                tts_exercised = True
                cases[current_case] = {
                    "status": "ready",
                    "language": expected_language,
                    "vad_speech_detected": True,
                    "speech_units": len(RESPONSE_TEXT[locale]),
                    "metrics": {
                        "fixture_to_transcript_final_ms": _milliseconds(transcript_at, started),
                        "speech_commit_to_tts_start_ms": _milliseconds(tts_started_at, commit_at),
                        "speech_done_to_playback_completed_ms": _milliseconds(playback_completed_at, speech_done_at),
                    },
                }

            current_case = "silence_fixture"
            core.ptt_down()
            silence_started = time.perf_counter()
            no_speech = core.ptt_up_and_process(AudioChunk(array("f", [0.0] * 16_000), 16_000), vad)
            silence_completed = time.perf_counter()
            if no_speech.type != "input.no_speech":
                raise BackendError("smoke.silence_detected_as_speech", "Synthetic silence was classified as speech", subsystem="smoke")
            cases[current_case] = {
                "status": "ready",
                "asr_invoked": False,
                "metrics": {"silence_vad_ms": _milliseconds(silence_completed, silence_started)},
            }

            current_case = "programmatic_barge_in"
            scripted = orchestrator.script_response(BARGE_TEXT, locale="el-GR", turn_id=no_speech.turn_id, interleave_speech=True)
            for server_event in scripted.events:
                core.handle_server_event(server_event)
            try:
                first = callbacks.get(timeout=5.0)
            except queue.Empty as exc:
                raise BackendError("smoke.barge_tts_not_started", "Barge-in TTS did not start", subsystem="smoke") from exc
            if first.kind != "started":
                raise BackendError("smoke.barge_callback_order_invalid", "Barge-in callback order was invalid", subsystem="smoke")
            core.on_tts_unit_started(first.unit)
            down_at = time.perf_counter()
            outbound = core.ptt_down()
            cancel_dispatched_at = time.perf_counter()
            outbound_types = {event.type for event in outbound}
            if not {"response.cancel", "playback.interrupted"}.issubset(outbound_types):
                raise BackendError("smoke.barge_protocol_incomplete", "Barge-in protocol events were incomplete", subsystem="smoke")
            if not tts.wait_idle(timeout=2.0):
                raise BackendError("smoke.barge_tts_not_idle", "Cancelled system TTS did not become idle", subsystem="smoke")
            if tts.last_error is not None:
                raise tts.last_error
            idle_at = time.perf_counter()
            while True:
                try:
                    callback = callbacks.get_nowait()
                except queue.Empty:
                    break
                stale = core.on_tts_unit_started(callback.unit) if callback.kind == "started" else core.on_tts_unit_completed(callback.unit)
                if any(event.type == "playback.completed" for event in stale):
                    raise BackendError("smoke.tombstoned_response_completed", "Tombstoned response completed playback", subsystem="smoke")
            cases[current_case] = {
                "status": "ready",
                "playback_interrupted": True,
                "backend_idle": True,
                "stale_completion_rejected": True,
                "metrics": {
                    "ptt_down_to_cancel_dispatch_ms": _milliseconds(cancel_dispatched_at, down_at),
                    "ptt_down_to_backend_idle_ms": _milliseconds(idle_at, down_at),
                },
            }
            report = _base_report(cases, ready=True, code="smoke.macos_automated_fixture_ready", tts_exercised=tts_exercised)
    except BackendError as exc:
        if cases[current_case].get("status") != "blocked":
            cases[current_case] = {"status": "blocked", "code": exc.code, "metrics": {}}
        report = _base_report(cases, ready=False, code=exc.code, tts_exercised=tts_exercised)
    write_report(output, report)
    return report
