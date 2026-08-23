from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.backends import AudioChunk, SpeechUnit
from jarvis_voice.config import ArtifactConfig, WhisperCppConfig, voice_config_from_mapping
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.audio_capture import CoreAudioCapture
from jarvis_voice.macos.doctor import macos_doctor_report
from jarvis_voice.voice_auth import create_voice_auth_key_file
from jarvis_voice.macos.ptt import (
    CG_EVENT_KEY_DOWN,
    CG_EVENT_KEY_UP,
    CG_EVENT_TAP_DISABLED_BY_TIMEOUT,
    MAC_CONTROL_OPTION_KEYCODE,
    MAC_RIGHT_OPTION_KEYCODE,
    CoreGraphicsEventTapAdapter,
    MacEventTapPttSource,
    mac_function_keycode,
    mac_ptt_keycode,
    mac_ptt_vkey,
)
from jarvis_voice.macos.recovery import MacRecoverySupervisor
from jarvis_voice.macos.runtime import TIMING_TRACE_PREFIX, content_free_timing_trace
from jarvis_voice.macos.system_tts import MacSystemTtsBackend, _start_say
from jarvis_voice.macos.whisper_cpp import WhisperCppSidecarBackend, parse_whisper_response
from jarvis_voice.ptt import PttEventPublisher, PttTransition, function_key_vkey


def artifact(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class MacTimingTraceTests(unittest.TestCase):
    def test_trace_is_content_free_and_machine_readable(self) -> None:
        trace = content_free_timing_trace("asr_result_ready", 12.5)

        self.assertTrue(trace.startswith(TIMING_TRACE_PREFIX))
        self.assertEqual(
            json.loads(trace.removeprefix(TIMING_TRACE_PREFIX)),
            {"event": "asr_result_ready", "timestamp": 12.5},
        )
        for forbidden in ("text", "transcript", "turn_id", "response_id", "/Users/"):
            self.assertNotIn(forbidden, trace)


class FakeRawInputStream:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    @property
    def active(self) -> bool:
        return self.started and not self.stopped and not self.closed


class MacAudioTests(unittest.TestCase):
    def test_coreaudio_uses_native_rate_and_minimal_callback(self) -> None:
        streams: list[FakeRawInputStream] = []
        settings: list[dict[str, object]] = []
        observed: list[tuple[bytes, int, float]] = []

        def stream(**kwargs: object) -> FakeRawInputStream:
            created = FakeRawInputStream(**kwargs)
            streams.append(created)
            return created

        fake_sd = SimpleNamespace(
            query_devices=lambda device, kind: {"default_samplerate": 48_000},
            CoreAudioSettings=lambda **kwargs: settings.append(kwargs) or ("coreaudio", kwargs),
            RawInputStream=stream,
        )
        with mock.patch.dict(sys.modules, {"sounddevice": fake_sd}), mock.patch("jarvis_voice.macos.audio_capture.sys.platform", "darwin"):
            capture = CoreAudioCapture(
                queue_capacity=4,
                maximum_utterance_seconds=1.0,
                frame_observer=lambda pcm, rate, timestamp: observed.append((pcm, rate, timestamp)),
            )
            capture.start()
            capture.begin_turn()
            streams[0].kwargs["callback"](array("f", [0.25, -0.5]).tobytes(), 2, None, None)  # type: ignore[index,operator]
            result = capture.end_turn()
            self.assertTrue(capture.is_healthy)
            capture.stop()
        self.assertEqual(result.sample_rate, 48_000)
        self.assertEqual(list(result.samples), [0.25, -0.5])
        self.assertEqual(settings, [{"change_device_parameters": False, "fail_if_conversion_required": False, "conversion_quality": "max"}])
        self.assertEqual(streams[0].kwargs["blocksize"], 0)
        self.assertEqual(streams[0].kwargs["latency"], "low")
        self.assertTrue(streams[0].stopped and streams[0].closed)
        self.assertEqual(observed[0][1], 48_000)
        self.assertEqual(observed[0][0], array("f", [0.25, -0.5]).tobytes())

    def test_coreaudio_restart_reopens_after_dead_stream(self) -> None:
        streams: list[FakeRawInputStream] = []
        fake_sd = SimpleNamespace(
            query_devices=lambda device, kind: {"default_samplerate": 48_000},
            CoreAudioSettings=lambda **kwargs: kwargs,
            RawInputStream=lambda **kwargs: streams.append(FakeRawInputStream(**kwargs)) or streams[-1],
        )
        with mock.patch.dict(sys.modules, {"sounddevice": fake_sd}), mock.patch("jarvis_voice.macos.audio_capture.sys.platform", "darwin"):
            capture = CoreAudioCapture(queue_capacity=4, maximum_utterance_seconds=1.0)
            capture.start()
            streams[0].stopped = True
            self.assertFalse(capture.is_healthy)
            capture.restart()
            self.assertTrue(capture.is_healthy)
            capture.stop()
        self.assertEqual(len(streams), 2)


class MacRecoveryTests(unittest.TestCase):
    def test_detects_dead_audio_and_retries_same_recovery_until_success(self) -> None:
        supervisor = MacRecoverySupervisor(initial_retry_seconds=0.25)
        request = supervisor.poll(audio_healthy=False, monotonic_now=1.0, wall_now=10.0)
        self.assertEqual(request.kind if request else None, "microphone_disconnect")
        supervisor.failed("microphone_disconnect", monotonic_now=1.0)
        self.assertIsNone(supervisor.poll(audio_healthy=True, monotonic_now=1.1, wall_now=10.1))
        retry = supervisor.poll(audio_healthy=True, monotonic_now=1.3, wall_now=10.3)
        self.assertEqual(retry.kind if retry else None, "microphone_disconnect")
        supervisor.succeeded()
        self.assertIsNone(supervisor.poll(audio_healthy=True, monotonic_now=1.4, wall_now=10.4))

    def test_detects_wake_sized_heartbeat_gap(self) -> None:
        supervisor = MacRecoverySupervisor(sleep_gap_seconds=5.0)
        self.assertIsNone(supervisor.poll(audio_healthy=True, monotonic_now=1.0, wall_now=10.0))
        request = supervisor.poll(audio_healthy=True, monotonic_now=1.1, wall_now=20.0)
        self.assertEqual(request.kind if request else None, "sleep_resume")


class FakeTapAdapter:
    def __init__(self, permitted: bool) -> None:
        self.permitted = permitted
        self.stopped = False

    def preflight(self) -> bool:
        return self.permitted

    def run(self, callback, on_ready) -> None:  # type: ignore[no-untyped-def]
        on_ready()

    def stop(self) -> None:
        self.stopped = True


class MacPttTests(unittest.TestCase):
    def publisher(self) -> PttEventPublisher:
        return PttEventPublisher(capacity=8, maximum_hold_seconds=5.0, ptt_vkey=function_key_vkey("F9"))

    def test_mapping_autorepeat_and_passive_transitions(self) -> None:
        publisher = self.publisher()
        source = MacEventTapPttSource(publisher, key="F9", adapter=FakeTapAdapter(True))
        source.handle_event(CG_EVENT_KEY_DOWN, mac_function_keycode("F9"), True, 1.0)
        self.assertTrue(publisher.events.empty())
        source.handle_event(CG_EVENT_KEY_DOWN, mac_function_keycode("F8"), False, 1.1)
        self.assertTrue(publisher.events.empty())
        source.handle_event(CG_EVENT_KEY_DOWN, mac_function_keycode("F9"), False, 1.2)
        source.handle_event(CG_EVENT_KEY_UP, mac_function_keycode("F9"), False, 1.3)
        self.assertEqual(publisher.events.get_nowait().transition, PttTransition.DOWN)
        self.assertEqual(publisher.events.get_nowait().transition, PttTransition.UP)
        self.assertTrue(source.passive)
        self.assertIsNone(source.suppression_mode)

    def test_disabled_tap_forces_release_and_permission_is_explicit(self) -> None:
        publisher = self.publisher()
        source = MacEventTapPttSource(publisher, key="F9", adapter=FakeTapAdapter(True))
        source.handle_event(CG_EVENT_KEY_DOWN, mac_function_keycode("F9"), False, 1.0)
        source.handle_event(CG_EVENT_TAP_DISABLED_BY_TIMEOUT, -1, False, 1.1)
        self.assertEqual(publisher.events.get_nowait().transition, PttTransition.DOWN)
        forced = publisher.events.get_nowait()
        self.assertEqual((forced.transition, forced.reason), (PttTransition.FORCED_UP, "event_tap_disabled"))
        denied = MacEventTapPttSource(self.publisher(), key="F9", adapter=FakeTapAdapter(False))
        with mock.patch("jarvis_voice.macos.ptt.sys.platform", "darwin"), self.assertRaisesRegex(BackendError, "input.macos_listen_permission_required"):
            denied.start()

    def test_disabled_tap_reenable_uses_owned_mach_port_not_callback_proxy(self) -> None:
        calls: list[tuple[object, bool]] = []
        cg = SimpleNamespace(
            CGEventTapEnable=lambda tap, enabled: calls.append((tap, enabled))
        )
        adapter = CoreGraphicsEventTapAdapter()
        adapter._event_tap = 0xCAFE

        adapter._reenable_event_tap(cg)  # type: ignore[arg-type]

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0].value, 0xCAFE)  # type: ignore[union-attr]
        self.assertTrue(calls[0][1])

    def test_control_option_chord_publishes_passive_transitions(self) -> None:
        publisher = PttEventPublisher(
            capacity=8,
            maximum_hold_seconds=5.0,
            ptt_vkey=mac_ptt_vkey("CONTROL_OPTION"),
        )
        source = MacEventTapPttSource(
            publisher,
            key="CONTROL_OPTION",
            adapter=FakeTapAdapter(True),
        )
        source.handle_event(CG_EVENT_KEY_DOWN, MAC_CONTROL_OPTION_KEYCODE, False, 1.0)
        source.handle_event(CG_EVENT_KEY_UP, MAC_CONTROL_OPTION_KEYCODE, False, 1.1)
        self.assertEqual(publisher.events.get_nowait().transition, PttTransition.DOWN)
        self.assertEqual(publisher.events.get_nowait().transition, PttTransition.UP)
        self.assertEqual(mac_ptt_keycode("CONTROL_OPTION"), MAC_CONTROL_OPTION_KEYCODE)

    def test_right_option_publishes_single_modifier_transitions(self) -> None:
        publisher = PttEventPublisher(
            capacity=8,
            maximum_hold_seconds=5.0,
            ptt_vkey=mac_ptt_vkey("RIGHT_OPTION"),
        )
        source = MacEventTapPttSource(
            publisher,
            key="RIGHT_OPTION",
            adapter=FakeTapAdapter(True),
        )
        source.handle_event(CG_EVENT_KEY_DOWN, MAC_RIGHT_OPTION_KEYCODE, False, 1.0)
        source.handle_event(CG_EVENT_KEY_UP, MAC_RIGHT_OPTION_KEYCODE, False, 1.1)
        self.assertEqual(publisher.events.get_nowait().transition, PttTransition.DOWN)
        self.assertEqual(publisher.events.get_nowait().transition, PttTransition.UP)
        self.assertEqual(mac_ptt_keycode("RIGHT_OPTION"), MAC_RIGHT_OPTION_KEYCODE)

    def test_only_f1_through_f20_are_supported(self) -> None:
        self.assertEqual(mac_function_keycode("F20"), 90)
        with self.assertRaisesRegex(BackendError, "config.unsupported_macos_ptt_key"):
            mac_function_keycode("F21")


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = io.BytesIO(body)
        self.status = status

    def read(self, amount: int = -1) -> bytes:
        return self.body.read(amount)

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        return None


class FakeSidecarProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode or 0


class WhisperCppTests(unittest.TestCase):
    def make_config(self, root: Path, *, response_cap: int = 4096) -> WhisperCppConfig:
        executable = root / "whisper-server"
        model = root / "model.bin"
        executable.write_bytes(b"native-sidecar")
        executable.chmod(0o700)
        model.write_bytes(b"local-model")
        exe = artifact(executable)
        mod = artifact(model)
        return WhisperCppConfig(
            executable=ArtifactConfig(**exe),  # type: ignore[arg-type]
            model=ArtifactConfig(**mod),  # type: ignore[arg-type]
            language="auto",
            threads=4,
            startup_timeout_seconds=1.0,
            request_timeout_seconds=2.0,
            maximum_response_bytes=response_cap,
        )

    def test_owned_loopback_sidecar_and_in_memory_multipart(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.make_config(root)
            argv_seen: list[list[str]] = []
            requests: list[object] = []
            process = FakeSidecarProcess()

            def opener(request, *, timeout):  # type: ignore[no-untyped-def]
                requests.append((request, timeout))
                if request.get_method() == "GET":
                    return FakeResponse(b"ok")
                return FakeResponse(json.dumps({
                    "text": "  Καλημέρα  ",
                    "language": "el",
                    "language_probability": 0.93,
                    "segments": [{"avg_logprob": -0.2, "tokens": [1, 2]}],
                }).encode())

            backend = WhisperCppSidecarBackend(
                config,
                maximum_audio_seconds=2.0,
                process_factory=lambda argv: argv_seen.append(argv) or process,
                opener=opener,
                port_factory=lambda: 43_210,
                platform_name="darwin",
            )
            result = backend.transcribe(AudioChunk(array("f", [0.0, 0.25, -0.25]), 16_000))
            backend.close()
        argv = argv_seen[0]
        self.assertIn("127.0.0.1", argv)
        context_index = argv.index("--max-context")
        self.assertEqual(argv[context_index + 1], "0")
        self.assertNotIn("--no-context", argv)
        self.assertIn("--flash-attn", argv)
        self.assertNotIn("--convert", argv)
        self.assertNotIn("--prompt", argv)
        post = requests[1][0]
        self.assertRegex(post.full_url, r"^http://127\.0\.0\.1:43210/jarvis-[0-9a-f]{32}/inference$")
        self.assertIn("--request-path", argv)
        self.assertIn(b'filename="turn.wav"', post.data)
        self.assertIn(b"RIFF", post.data)
        self.assertNotIn(result.text.encode(), b"\x00".join(item.encode() for item in argv))
        self.assertEqual((result.text, result.language.code), ("Καλημέρα", "el"))
        self.assertAlmostEqual(result.asr_score.raw_avg_logprob, -0.2)
        self.assertTrue(process.terminated)

    def test_manifest_precedes_process_and_response_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.make_config(root, response_cap=8)
            config.executable.path and Path(config.executable.path).write_bytes(b"changed")
            called = False

            def factory(argv):  # type: ignore[no-untyped-def]
                nonlocal called
                called = True
                return FakeSidecarProcess()

            backend = WhisperCppSidecarBackend(config, maximum_audio_seconds=1.0, process_factory=factory, platform_name="darwin")
            with self.assertRaisesRegex(BackendError, "asr.whisper_cpp_executable_size_mismatch"):
                backend.load()
            self.assertFalse(called)

            valid = self.make_config(root, response_cap=8)
            process = FakeSidecarProcess()
            calls = 0

            def opener(request, *, timeout):  # type: ignore[no-untyped-def]
                nonlocal calls
                calls += 1
                return FakeResponse(b"ok" if calls == 1 else b"123456789")

            bounded = WhisperCppSidecarBackend(valid, maximum_audio_seconds=1.0, process_factory=lambda argv: process, opener=opener, port_factory=lambda: 43211, platform_name="darwin")
            with self.assertRaisesRegex(BackendError, "asr.whisper_cpp_response_too_large"):
                bounded.transcribe(AudioChunk(array("f", [0.0]), 16_000))
            bounded.close()

    def test_parser_never_labels_missing_logprob_as_confidence(self) -> None:
        result = parse_whisper_response(b'{"text":"hello","language":"en"}', configured_language="auto")
        self.assertEqual(result.asr_score.method, "whisper_cpp_score_unavailable")
        self.assertFalse(result.asr_score.calibrated)
        self.assertEqual(result.asr_score.value, 0.0)


class Sink:
    def __init__(self, process: "FakeSayProcess") -> None:
        self.process = process

    def write(self, data: bytes) -> int:
        self.process.input_bytes += data
        return len(data)

    def close(self) -> None:
        if self.process.complete_on_close:
            self.process.returncode = 0


class FakeSayProcess:
    def __init__(self, complete_on_close: bool) -> None:
        self.complete_on_close = complete_on_close
        self.returncode: int | None = None
        self.input_bytes = b""
        self.stdin = Sink(self)
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode or 0


class StubbornSayProcess(FakeSayProcess):
    def __init__(self) -> None:
        super().__init__(False)

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("say", timeout)
        return self.returncode


class MacTtsTests(unittest.TestCase):
    def test_say_inherits_voice_group_for_supervisor_crash_containment(self) -> None:
        with mock.patch(
            "jarvis_voice.macos.system_tts.subprocess.Popen"
        ) as popen:
            _start_say(["/usr/bin/say", "-f", "-"])

        kwargs = popen.call_args.kwargs
        self.assertNotIn("start_new_session", kwargs)
        self.assertFalse(kwargs["shell"])

    def test_text_is_stdin_only_and_units_remain_ordered(self) -> None:
        processes: list[FakeSayProcess] = []
        argv: list[list[str]] = []
        completed: list[int] = []

        def factory(args: list[str]) -> FakeSayProcess:
            argv.append(args)
            process = FakeSayProcess(True)
            processes.append(process)
            return process

        backend = MacSystemTtsBackend(voice_mapping={"el-GR": "Melina"}, process_factory=factory, on_completed=lambda unit: completed.append(unit.unit_index))
        backend.enqueue(SpeechUnit("r", 1, 1, "el-GR", "Πρώτη πρόταση."))
        backend.enqueue(SpeechUnit("r", 1, 2, "el-GR", "Δεύτερη πρόταση."))
        self.assertTrue(backend.wait_idle())
        backend.close()
        self.assertEqual(completed, [1, 2])
        self.assertEqual(argv, [["/usr/bin/say", "-v", "Melina", "-f", "-"]] * 2)
        self.assertEqual([item.input_bytes.decode() for item in processes], ["Πρώτη πρόταση.", "Δεύτερη πρόταση."])
        self.assertNotIn("Πρώτη πρόταση.", " ".join(argv[0]))

    def test_completion_callback_observes_queued_then_idle_state(self) -> None:
        release_worker = threading.Event()
        completed = threading.Event()
        observed: list[tuple[int, bool]] = []
        backend_ref: dict[str, MacSystemTtsBackend] = {}

        def on_completed(unit: SpeechUnit) -> None:
            observed.append((unit.unit_index, backend_ref["backend"].is_idle))
            if unit.unit_index == 2:
                completed.set()

        def factory(_argv: list[str]) -> FakeSayProcess:
            self.assertTrue(release_worker.wait(1.0))
            return FakeSayProcess(True)

        backend = MacSystemTtsBackend(
            voice_mapping={},
            process_factory=factory,
            on_completed=on_completed,
        )
        backend_ref["backend"] = backend
        self.addCleanup(backend.close)

        backend.enqueue(SpeechUnit("r", 1, 1, "en-US", "final"))
        backend.enqueue(SpeechUnit("r", 1, 2, "en-US", "really final"))
        release_worker.set()

        self.assertTrue(completed.wait(1.0))
        self.assertEqual(observed, [(1, False), (2, True)])

    def test_cancel_terminates_active_and_rejects_late_completion(self) -> None:
        process = FakeSayProcess(False)
        started = threading.Event()
        completed: list[int] = []
        backend = MacSystemTtsBackend(
            voice_mapping={},
            process_factory=lambda argv: process,
            on_started=lambda unit: started.set(),
            on_completed=lambda unit: completed.append(unit.unit_index),
        )
        backend.enqueue(SpeechUnit("r", 1, 1, "en-US", "active"))
        self.assertTrue(started.wait(1.0))
        backend.enqueue(SpeechUnit("r", 1, 2, "en-US", "pending"))
        backend.cancel(1, "ptt_down")
        self.assertTrue(backend.wait_idle())
        backend.close()
        self.assertTrue(process.terminated)
        self.assertEqual(completed, [])

    def test_close_escalates_exact_stubborn_say_process_and_reaps(self) -> None:
        process = StubbornSayProcess()
        started = threading.Event()
        completed: list[int] = []
        backend = MacSystemTtsBackend(
            voice_mapping={},
            process_factory=lambda _argv: process,
            on_started=lambda _unit: started.set(),
            on_completed=lambda unit: completed.append(unit.unit_index),
        )
        backend.enqueue(SpeechUnit("r", 1, 1, "en-US", "active"))
        self.assertTrue(started.wait(1.0))

        backend.close()

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(completed, [])
        self.assertTrue(backend.is_idle)


class MacDoctorConfigTests(unittest.TestCase):
    def test_greektts_runtime_profile_requires_absolute_digest_bound_paths(self) -> None:
        digest = "a" * 64
        config = voice_config_from_mapping({
            "macos": {
                "tts_backend": "greektts15_f1",
                "greektts15": {
                    "model_root": "/private/models/greektts15",
                    "approval_manifest_path": "/private/approved.json",
                    "approval_manifest_sha256": digest,
                    "repository_root": "/private/repository",
                    "warm_on_start": True,
                },
            },
        })
        self.assertEqual(config.macos.tts_backend, "greektts15_f1")
        self.assertEqual(config.macos.greektts15.approval_manifest_sha256, digest)
        with self.assertRaisesRegex(BackendError, "config.greektts15_profile_required"):
            voice_config_from_mapping({
                "macos": {
                    "tts_backend": "greektts15_f1",
                    "greektts15": {
                        "model_root": "relative",
                        "approval_manifest_path": "/private/approved.json",
                        "approval_manifest_sha256": digest,
                        "repository_root": "/private/repository",
                    },
                },
            })

    def test_strict_config_and_ready_report_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "whisper-server"
            model = root / "model.bin"
            vad = root / "silero.onnx"
            auth_key = root / "voice-auth.json"
            executable.write_bytes(b"exe")
            executable.chmod(0o700)
            model.write_bytes(b"model")
            vad.write_bytes(b"vad")
            create_voice_auth_key_file(auth_key, key=b"k" * 32)
            config = voice_config_from_mapping({
                "ptt_key": "F9",
                "orchestrator_url": "ws://127.0.0.1:8765",
                "voice_auth_key_file": str(auth_key),
                "vad": {"artifact": artifact(vad)},
                "macos": {
                    "whisper_cpp": {
                        "executable": artifact(executable),
                        "model": artifact(model),
                        "language": "auto",
                    },
                    "system_voices": {"el-GR": "Melina", "en-US": "Samantha"},
                },
            })
            ready_dependencies = {name: {"status": "ready", "code": "dependency.available"} for name in ("sounddevice", "soxr", "onnxruntime", "websockets")}
            report = macos_doctor_report(
                config,
                platform_name="darwin",
                machine="arm64",
                macos_version="14.6",
                python_version=(3, 11),
                pointer_bits=64,
                permission_probe=lambda: True,
                audio_probe=lambda device: True,
                voice_probe=lambda: {"Melina", "Samantha"},
                dependencies=ready_dependencies,
            )
            encoded = json.dumps(report)
            self.assertTrue(report["ready"])
            self.assertFalse(report["network_used"])
            self.assertFalse(report["content_captured"])
            self.assertNotIn(str(root), encoded)

        with self.assertRaisesRegex(BackendError, "config.invalid_field"):
            voice_config_from_mapping({"macos": {"unexpected": True}})

    def test_cli_exposes_macos_commands(self) -> None:
        from jarvis_voice.cli import build_parser

        self.assertEqual(build_parser().parse_args(["doctor-macos", "--config", "voice.json"]).command, "doctor-macos")
        self.assertEqual(build_parser().parse_args(["run-macos", "--config", "voice.json"]).command, "run-macos")


@unittest.skipUnless(sys.platform == "darwin", "native macOS framework smoke required")
class NativeMacSmokeTests(unittest.TestCase):
    def test_coregraphics_permission_preflight_is_non_prompting(self) -> None:
        from jarvis_voice.macos.ptt import CoreGraphicsEventTapAdapter

        self.assertIsInstance(CoreGraphicsEventTapAdapter().preflight(), bool)

    @unittest.skipUnless(os.path.isfile("/usr/bin/say"), "macOS say executable required")
    def test_system_say_executable_is_present(self) -> None:
        self.assertTrue(os.access("/usr/bin/say", os.X_OK))


if __name__ == "__main__":
    unittest.main()
