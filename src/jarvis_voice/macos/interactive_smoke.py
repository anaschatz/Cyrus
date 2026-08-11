"""Guided, content-free native macOS PTT smoke qualification."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from ..config import VoiceConfig
from ..errors import BackendError
from ..fakes import FakeOrchestrator
from ..protocol import Envelope
from ..qualification import write_report
from .runtime import MacVoiceRuntime


SMOKE_SCHEMA_VERSION = 1
CASE_ORDER = ("greek_turn", "english_turn", "no_speech", "barge_in")
TRACEABLE_STAGE_EVENTS = frozenset(
    {
        "audio_stream_started",
        "ptt_source_started",
        "ptt_down_received",
        "ptt_up_received",
        "input_no_speech_created",
        "client_error_created",
        "transcript_final_created",
        "speech_commit_received",
        "tts_unit_started",
        "speech_done_received",
        "playback_completed_created",
        "tts_cancel_dispatched",
        "response_cancel_created",
        "playback_interrupted_created",
    }
)


def _milliseconds(later: float, earlier: float) -> float:
    return round(max(0.0, (later - earlier) * 1000.0), 3)


@dataclass(frozen=True, slots=True)
class SmokePrompt:
    code: str
    message: str


class GuidedSmokeController:
    """Consumes content-free runtime telemetry and advances strict native cases."""

    def __init__(self, *, ptt_key: str) -> None:
        self.ptt_key = ptt_key
        self.stage = "greek_turn"
        self.finished = False
        self.failure_code: str | None = None
        self.cases: dict[str, dict[str, object]] = {
            name: {"status": "pending", "metrics": {}} for name in CASE_ORDER
        }
        self._markers: dict[str, float] = {}
        self._barge_tts_started = False
        self._barge_backend_idle = False
        self._prompts: list[SmokePrompt] = [self._prompt_for("greek_turn")]
        self._microphone_opened = False
        self._ptt_started = False
        self._tts_exercised = False

    def _prompt_for(self, stage: str) -> SmokePrompt:
        if stage == "greek_turn":
            return SmokePrompt("smoke.prompt.greek", f"Κράτησε {self.ptt_key}, πες μία σύντομη ελληνική πρόταση και άφησέ το.")
        if stage == "english_turn":
            return SmokePrompt("smoke.prompt.english", f"Hold {self.ptt_key}, say one short English sentence, then release it.")
        if stage == "no_speech":
            return SmokePrompt("smoke.prompt.silence", f"Πάτησε και άφησε σύντομα {self.ptt_key} χωρίς να μιλήσεις.")
        return SmokePrompt("smoke.prompt.barge_trigger", f"Κράτησε {self.ptt_key}, πες μία σύντομη πρόταση και άφησέ το. Περίμενε μετά την ένδειξη BARGE-IN NOW.")

    def take_prompts(self) -> tuple[SmokePrompt, ...]:
        prompts = tuple(self._prompts)
        self._prompts.clear()
        return prompts

    def _fail(self, code: str) -> None:
        if self.finished:
            return
        self.failure_code = code
        self.cases[self.stage] = {"status": "blocked", "code": code, "metrics": {}}
        self.finished = True

    def _advance(self, metrics: Mapping[str, float], **details: object) -> None:
        result: dict[str, object] = {"status": "ready", "metrics": dict(metrics)}
        result.update(details)
        self.cases[self.stage] = result
        index = CASE_ORDER.index(self.stage)
        self._markers.clear()
        if index == len(CASE_ORDER) - 1:
            self.finished = True
            return
        self.stage = CASE_ORDER[index + 1]
        self._prompts.append(self._prompt_for(self.stage))

    def _voice_event(self, name: str, timestamp: float, entry: Mapping[str, object]) -> None:
        if name == "ptt_up_received":
            self._markers["ptt_up"] = timestamp
        elif name == "input_no_speech_created":
            self._fail("smoke.expected_speech")
        elif name == "client_error_created":
            runtime_code = entry.get("code")
            self._fail("smoke.client_error")
            self.cases[self.stage]["runtime_code"] = (
                runtime_code if isinstance(runtime_code, str) else "client.error"
            )
        elif name == "transcript_final_created":
            expected = "el" if self.stage == "greek_turn" else "en"
            if entry.get("language") != expected:
                self._fail("smoke.language_mismatch")
                self.cases[self.stage]["expected_language"] = expected
                self.cases[self.stage]["detected_language"] = str(
                    entry.get("language", "unknown")
                )
                score = entry.get("language_score")
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    self.cases[self.stage]["language_score"] = round(float(score), 6)
                return
            self._markers["transcript_final"] = timestamp
            self._markers["language_verified"] = timestamp
        elif name == "speech_commit_received" and "speech_commit" not in self._markers:
            self._markers["speech_commit"] = timestamp
        elif name == "tts_unit_started" and "tts_start" not in self._markers:
            self._markers["tts_start"] = timestamp
        elif name == "speech_done_received":
            self._markers["speech_done"] = timestamp
        elif name == "playback_completed_created":
            required = {"ptt_up", "transcript_final", "speech_commit", "tts_start", "speech_done"}
            if not required.issubset(self._markers):
                self._fail("smoke.voice_sequence_incomplete")
                return
            metrics = {
                "ptt_up_to_transcript_final_ms": _milliseconds(self._markers["transcript_final"], self._markers["ptt_up"]),
                "speech_commit_to_tts_start_ms": _milliseconds(self._markers["tts_start"], self._markers["speech_commit"]),
                "speech_done_to_playback_completed_ms": _milliseconds(timestamp, self._markers["speech_done"]),
            }
            self._advance(metrics, language="el" if self.stage == "greek_turn" else "en")

    def _silence_event(self, name: str, timestamp: float) -> None:
        if name == "ptt_up_received":
            self._markers["ptt_up"] = timestamp
        elif name == "transcript_final_created":
            self._fail("smoke.expected_no_speech")
        elif name == "input_no_speech_created":
            if "ptt_up" not in self._markers:
                self._fail("smoke.silence_sequence_incomplete")
                return
            self._advance({"ptt_up_to_no_speech_ms": _milliseconds(timestamp, self._markers["ptt_up"])})

    def _barge_event(self, name: str, timestamp: float, entry: Mapping[str, object]) -> None:
        if not self._barge_tts_started:
            if name == "input_no_speech_created":
                self._fail("smoke.barge_trigger_no_speech")
            elif name == "transcript_final_created":
                self._markers["barge_trigger_transcript"] = timestamp
            elif name == "tts_unit_started" and "barge_trigger_transcript" in self._markers:
                self._barge_tts_started = True
                self._tts_exercised = True
                self._markers["barge_tts_start"] = timestamp
                self._prompts.append(
                    SmokePrompt(
                        "smoke.prompt.barge_now",
                        f"BARGE-IN NOW: ενώ ακούγεται το TTS, πάτησε και άφησε {self.ptt_key} χωρίς να μιλήσεις.",
                    )
                )
            return
        if name == "ptt_down_received" and entry.get("output_active") is True and "barge_down" not in self._markers:
            self._markers["barge_down"] = timestamp
        elif name == "tts_cancel_dispatched" and "barge_down" in self._markers:
            self._markers["cancel_dispatch"] = timestamp
        elif name == "response_cancel_created" and "barge_down" in self._markers:
            self._markers["response_cancel"] = timestamp
        elif name == "playback_interrupted_created" and "barge_down" in self._markers:
            self._markers["playback_interrupted"] = timestamp
        elif name == "ptt_up_received" and "barge_down" in self._markers:
            self._markers["barge_up"] = timestamp
        elif name == "playback_completed_created" and "barge_down" in self._markers:
            self._fail("smoke.tombstoned_response_completed")

    def feed(self, entry: Mapping[str, object]) -> None:
        if self.finished:
            return
        name, timestamp = entry.get("event"), entry.get("timestamp")
        if not isinstance(name, str) or not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
            self._fail("smoke.telemetry_invalid")
            return
        value = float(timestamp)
        if name == "audio_stream_started":
            self._microphone_opened = True
        elif name == "ptt_source_started":
            self._ptt_started = True
        elif name == "tts_unit_started":
            self._tts_exercised = True
        if self.stage in {"greek_turn", "english_turn"}:
            self._voice_event(name, value, entry)
        elif self.stage == "no_speech":
            self._silence_event(name, value)
        else:
            self._barge_event(name, value, entry)

    def tick(self, *, backend_idle: bool, now: float) -> None:
        if self.finished or self.stage != "barge_in" or "barge_down" not in self._markers:
            return
        if backend_idle and not self._barge_backend_idle:
            self._barge_backend_idle = True
            self._markers["backend_idle"] = now
        required = {"barge_down", "cancel_dispatch", "response_cancel", "playback_interrupted", "barge_up", "backend_idle"}
        if required.issubset(self._markers):
            metrics = {
                "ptt_down_to_cancel_dispatch_ms": _milliseconds(self._markers["cancel_dispatch"], self._markers["barge_down"]),
                "ptt_down_to_backend_idle_ms": _milliseconds(self._markers["backend_idle"], self._markers["barge_down"]),
            }
            self._advance(metrics, playback_interrupted=True, backend_idle=True)

    def timeout(self) -> None:
        self._fail("smoke.interaction_timeout")

    def report(self, *, runtime_error: str | None = None) -> dict[str, object]:
        if runtime_error is not None and self.failure_code is None:
            self._fail(runtime_error)
        ready = self.finished and self.failure_code is None and all(
            item.get("status") == "ready" for item in self.cases.values()
        )
        return {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "milestone": "5E",
            "ready": ready,
            "code": "smoke.macos_interactive_ready" if ready else (self.failure_code or "smoke.incomplete"),
            "cases": self.cases,
            "native": {
                "microphone_opened": self._microphone_opened,
                "passive_ptt_started": self._ptt_started,
                "system_tts_exercised": self._tts_exercised,
                "acoustic_silence_measured": False,
                "satisfies_native_ready_gate": False,
            },
            "privacy": {
                "raw_audio_persisted": False,
                "transcript_text_persisted": False,
                "protocol_ids_persisted": False,
                "content_captured_in_report": False,
            },
            "orchestrator_loopback_only": True,
            "network_used": False,
        }


class MacSmokeLoopbackServer:
    """Temporary loopback server with a deliberately long barge-in response."""

    def __init__(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "ws" or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
            raise BackendError("smoke.non_loopback_server", "Smoke server must use loopback WebSocket", subsystem="smoke")
        self.host = "127.0.0.1" if parsed.hostname == "localhost" else str(parsed.hostname)
        self.port = parsed.port or 8765
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._acknowledged: set[str] = set()
        self._transcript_count = 0

    async def _handle(self, websocket: Any) -> None:
        orchestrator: FakeOrchestrator | None = None
        async for raw in websocket:
            event = Envelope.from_json(raw)
            if orchestrator is None:
                orchestrator = FakeOrchestrator(session_id=event.session_id)
                orchestrator.acknowledged_event_ids = self._acknowledged
            duplicate = event.event_id in self._acknowledged
            for reply in orchestrator.accept(event):
                await websocket.send(reply.to_json())
            if event.type == "input.transcript.final" and not duplicate:
                self._transcript_count += 1
                language = event.payload.get("language", {})
                code = language.get("code", "en") if isinstance(language, Mapping) else "en"
                greek = code == "el"
                locale = "el-GR" if greek else "en-US"
                if self._transcript_count >= 3:
                    sentences = (
                        (
                            "Αυτή είναι μια εσκεμμένα μεγάλη πρόταση για τη δοκιμή άμεσης διακοπής της αναπαραγωγής όταν πατηθεί ξανά το πλήκτρο ομιλίας.",
                            "Η δεύτερη μονάδα δεν πρέπει να ακουστεί μετά τη διακοπή.",
                        )
                        if greek
                        else (
                            "This is an intentionally long sentence used to verify immediate playback interruption when push to talk is pressed again.",
                            "The second unit must not be heard after cancellation.",
                        )
                    )
                else:
                    sentences = ("Βεβαίως.", "Η δοκιμή ολοκληρώθηκε.") if greek else ("Certainly.", "The test is complete.")
                scripted = orchestrator.script_response(sentences, locale=locale, turn_id=event.turn_id, interleave_speech=True)
                for reply in scripted.events:
                    await websocket.send(reply.to_json())
            elif event.type == "response.cancel" and event.response_id is not None:
                cancelled = orchestrator.factory.create(
                    "response.cancelled",
                    {"reason": "client_cancel"},
                    turn_id=event.turn_id,
                    response_id=event.response_id,
                )
                await websocket.send(cancelled.to_json())

    async def _serve(self) -> None:
        try:
            from websockets.asyncio.server import serve  # type: ignore[import-not-found]

            async with serve(self._handle, self.host, self.port):
                self._ready.set()
                while not self._stop.is_set():
                    await asyncio.sleep(0.02)
        except BaseException as exc:
            self._error = exc
            self._ready.set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=lambda: asyncio.run(self._serve()), name="jarvis-macos-smoke-server", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            self.stop()
            raise BackendError("smoke.server_start_timeout", "Smoke server startup timed out", subsystem="smoke")
        if self._error is not None:
            self.stop()
            raise BackendError("smoke.server_start_failed", "Smoke server could not bind to loopback", subsystem="smoke") from self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None


def _runtime_error_code(exc: BaseException) -> str:
    return exc.code if isinstance(exc, BackendError) else "smoke.runtime_failed"


def run_interactive_macos_smoke(
    config: VoiceConfig,
    *,
    output: str,
    timeout_seconds: float = 300.0,
    runtime_factory: Callable[[VoiceConfig], MacVoiceRuntime] = MacVoiceRuntime,
    server_factory: Callable[[str], MacSmokeLoopbackServer] = MacSmokeLoopbackServer,
    prompt_sink: Callable[[SmokePrompt], None] | None = None,
    stage_sink: Callable[[str], None] | None = None,
) -> dict[str, object]:
    if timeout_seconds <= 0 or timeout_seconds > 3600:
        raise BackendError("smoke.timeout_invalid", "Smoke timeout is invalid", subsystem="smoke")
    sink = prompt_sink or (lambda prompt: print(f"[{prompt.code}] {prompt.message}", flush=True))
    controller = GuidedSmokeController(ptt_key=config.ptt_key)
    server = server_factory(config.orchestrator_url)
    runtime = runtime_factory(config)
    runtime_error: list[str] = []

    def run_runtime() -> None:
        try:
            runtime.run()
        except BaseException as exc:
            runtime_error.append(_runtime_error_code(exc))

    thread = threading.Thread(target=run_runtime, name="jarvis-macos-smoke-runtime", daemon=True)
    cursor = 0
    deadline = time.monotonic() + timeout_seconds
    try:
        server.start()
        thread.start()
        for prompt in controller.take_prompts():
            sink(prompt)
        while not controller.finished and not runtime_error:
            snapshot: Sequence[Mapping[str, object]] = runtime.timing_snapshot()
            for entry in snapshot[cursor:]:
                event_name = entry.get("event")
                if stage_sink is not None and isinstance(event_name, str) and event_name in TRACEABLE_STAGE_EVENTS:
                    runtime_code = entry.get("code")
                    if event_name == "client_error_created" and isinstance(runtime_code, str):
                        stage_sink(f"{event_name}:{runtime_code}")
                    elif event_name == "transcript_final_created":
                        stage_sink(
                            f"{event_name}:{entry.get('language', 'unknown')}:"
                            f"{entry.get('language_score', 0.0)}"
                        )
                    else:
                        stage_sink(event_name)
                controller.feed(entry)
            cursor = len(snapshot)
            controller.tick(backend_idle=runtime.tts.is_idle, now=time.perf_counter())
            for prompt in controller.take_prompts():
                sink(prompt)
            if time.monotonic() >= deadline:
                controller.timeout()
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        runtime_error.append("smoke.user_aborted")
    except BackendError as exc:
        runtime_error.append(exc.code)
    finally:
        runtime.request_stop()
        if thread.is_alive():
            thread.join(timeout=config.application.shutdown_timeout_seconds)
        server.stop()
    report = controller.report(runtime_error=runtime_error[0] if runtime_error else None)
    write_report(output, report)
    return report
