from __future__ import annotations

import hashlib
import inspect
import sys
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.backends import SpeechUnit
from jarvis_voice.config import ArtifactConfig, voice_config_from_mapping
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.doctor import macos_doctor_report
from jarvis_voice.macos.kokoro_onnx import KokoroOnnxTtsBackend
from jarvis_voice.macos.native_collector import mac_runtime_profile_fingerprint
from jarvis_voice.pocket_tts import (
    PocketAudioChunk,
    PocketTtsBackend,
    POCKET_TTS_WARM_TIMEOUT_SECONDS,
    SoundDeviceStreamingPlayback,
    validate_pocket_tts_artifact,
)


class FakeEngine:
    def __init__(self, *, fail_warm: bool = False, block: bool = False) -> None:
        self.fail_warm = fail_warm
        self.block = block
        self.cancelled = threading.Event()
        self.first_chunk = threading.Event()
        self.texts: list[str] = []

    def warm(self) -> float:
        if self.fail_warm:
            raise RuntimeError("unavailable")
        return 18.5

    def stream(
        self, text: str, cancelled: threading.Event
    ) -> object:
        self.texts.append(text)
        for payload in (
            b"\x01\x00" * 480,
            b"\x02\x00" * 480,
            b"\x03\x00" * 480,
        ):
            if cancelled.is_set():
                return
            self.first_chunk.set()
            yield PocketAudioChunk(payload)
            if self.block and payload.startswith(b"\x03\x00"):
                self.cancelled.wait(1.0)

    def cancel(self) -> None:
        self.cancelled.set()


class FakeFallback:
    def __init__(self) -> None:
        self.units: list[SpeechUnit] = []
        self.cancelled: list[tuple[int, str]] = []
        self.closed = False

    def enqueue(self, unit: SpeechUnit) -> None:
        self.units.append(unit)

    def cancel(self, new_epoch: int, reason: str) -> None:
        self.cancelled.append((new_epoch, reason))

    @property
    def is_idle(self) -> bool:
        return True

    @property
    def last_error(self) -> BackendError | None:
        return None

    def close(self) -> None:
        self.closed = True


class FakePlayback:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.aborted = threading.Event()
        self.finished = False
        self.chunks: list[bytes] = []

    def start(self, sample_rate: int) -> None:
        if sample_rate != 24_000:
            raise AssertionError("unexpected sample rate")
        self.started.set()

    def write(self, pcm16: bytes) -> None:
        self.chunks.append(pcm16)

    def finish(self) -> None:
        self.finished = True

    def abort(self) -> None:
        self.aborted.set()


def speech_unit(
    *, epoch: int = 1, locale: str = "en-US", index: int = 1
) -> SpeechUnit:
    return SpeechUnit("response", epoch, index, locale, "All systems are ready.")


def artifact(path: str = "/models/artifact") -> dict[str, object]:
    return {"path": path, "size": 1, "sha256": "a" * 64}


class PocketTtsConfigTests(unittest.TestCase):
    def test_mac_and_windows_candidate_use_one_locked_artifact_profile(self) -> None:
        config = voice_config_from_mapping(
            {
                "windows_tts_backend": "pocket_tts",
                "pocket_tts": {
                    "model": artifact("/models/pocket/model.safetensors"),
                    "tokenizer": artifact("/models/pocket/tokenizer.model"),
                    "voice_state": artifact("/models/pocket/george.safetensors"),
                    "temperature": 0.7,
                    "max_tokens": 50,
                    "warm_on_start": True,
                },
                "macos": {"tts_backend": "pocket_tts"},
            }
        )

        self.assertEqual(config.windows_tts_backend, "pocket_tts")
        self.assertEqual(config.macos.tts_backend, "pocket_tts")
        self.assertEqual(config.pocket_tts.max_tokens, 50)

    def test_candidate_rejects_unbound_or_mutated_settings(self) -> None:
        with self.assertRaisesRegex(BackendError, "config.pocket_tts_profile_required"):
            voice_config_from_mapping(
                {
                    "macos": {"tts_backend": "pocket_tts"},
                    "pocket_tts": {
                        "model": artifact("relative"),
                        "tokenizer": artifact("/tokenizer"),
                        "voice_state": artifact("/voice"),
                    },
                }
            )
        with self.assertRaisesRegex(BackendError, "config.pocket_tts_candidate_locked"):
            voice_config_from_mapping(
                {
                    "windows_tts_backend": "pocket_tts",
                    "pocket_tts": {
                        "model": artifact("/model"),
                        "tokenizer": artifact("/tokenizer"),
                        "voice_state": artifact("/voice"),
                        "temperature": 0.8,
                    },
                }
            )

    def test_artifact_integrity_rejects_changed_bytes_and_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "model.safetensors"
            path.write_bytes(b"model")
            manifest = ArtifactConfig(
                str(path), 5, hashlib.sha256(b"model").hexdigest()
            )
            self.assertEqual(validate_pocket_tts_artifact(manifest, "model"), path)
            path.write_bytes(b"other")
            with self.assertRaisesRegex(BackendError, "tts.pocket.artifact_changed"):
                validate_pocket_tts_artifact(manifest, "model")
            path.write_bytes(b"model")
            linked = root / "linked.safetensors"
            linked.symlink_to(path)
            linked_manifest = ArtifactConfig(
                str(linked), 5, hashlib.sha256(b"model").hexdigest()
            )
            with self.assertRaisesRegex(BackendError, "tts.pocket.artifact_changed"):
                validate_pocket_tts_artifact(linked_manifest, "model")

    def test_macos_doctor_reports_offline_streaming_profile_without_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests: dict[str, dict[str, object]] = {}
            for name in ("model", "tokenizer", "voice_state"):
                path = root / name
                content = name.encode("ascii")
                path.write_bytes(content)
                manifests[name] = {
                    "path": str(path),
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            config = voice_config_from_mapping(
                {
                    "macos": {
                        "tts_backend": "pocket_tts",
                        "system_voices": {"el-GR": "Melina", "en-US": "Samantha"},
                    },
                    "pocket_tts": manifests,
                }
            )
            dependencies = {
                name: {
                    "status": "ready",
                    "code": "dependency.available",
                    "package": name,
                    "version": {
                        "pocket-tts": "2.1.0",
                        "torch": "2.13.0",
                        "PyYAML": "6.0.3",
                    }.get(name, "test"),
                }
                for name in (
                    "sounddevice",
                    "soxr",
                    "onnxruntime",
                    "websockets",
                    "pocket-tts",
                    "torch",
                    "PyYAML",
                )
            }

            report = macos_doctor_report(
                config,
                dependencies=dependencies,
                voice_probe=lambda: {"Melina", "Samantha"},
                restricted_native_probe=True,
            )

            tts = report["checks"]["tts"]
            self.assertEqual(tts["status"], "ready")
            self.assertEqual(tts["backend"], "pocket_tts")
            self.assertTrue(tts["streaming"])
            self.assertFalse(tts["network_fallback"])
            self.assertNotIn(str(root), str(tts))

    def test_native_profile_binds_pocket_artifacts_and_runtime_settings(self) -> None:
        base = {
            "macos": {
                "tts_backend": "pocket_tts",
                "system_voices": {"el-GR": "Melina", "en-US": "Samantha"},
            },
            "pocket_tts": {
                "model": artifact("/models/pocket/model.safetensors"),
                "tokenizer": artifact("/models/pocket/tokenizer.model"),
                "voice_state": artifact("/models/pocket/george.safetensors"),
                "warm_on_start": True,
            },
        }
        original = mac_runtime_profile_fingerprint(voice_config_from_mapping(base))
        changed = {
            **base,
            "pocket_tts": {
                **base["pocket_tts"],
                "voice_state": {
                    **base["pocket_tts"]["voice_state"],
                    "sha256": "b" * 64,
                },
            },
        }
        changed_fingerprint = mac_runtime_profile_fingerprint(
            voice_config_from_mapping(changed)
        )

        self.assertNotEqual(original, changed_fingerprint)
        self.assertEqual(len(original), 64)
        self.assertNotIn("/models", original)


class PocketTtsBackendTests(unittest.TestCase):
    def test_cancel_attempts_every_backend_after_failure(self) -> None:
        for backend_type in (PocketTtsBackend, KokoroOnnxTtsBackend):
            for failing in ("engine", "playback", "fallback"):
                for error in (OSError("cancel failed"), KeyboardInterrupt()):
                    with self.subTest(backend=backend_type.__name__, failing=failing, error=type(error).__name__):
                        engine, playback, fallback = FakeEngine(), FakePlayback(), FakeFallback()
                        backend = backend_type(engine=engine, fallback_backend=fallback, warm_on_start=False)
                        cancelled = threading.Event()
                        with backend._lock:
                            backend._playback = playback
                            backend._active_cancel = cancelled
                        calls = []
                        try:
                            with ExitStack() as patches:
                                for name, target, method in (("engine", engine, "cancel"), ("playback", playback, "abort"), ("fallback", fallback, "cancel")):
                                    def call(*args, operation=name):
                                        calls.append(operation)
                                        if operation == failing:
                                            raise error
                                    patches.enter_context(mock.patch.object(target, method, side_effect=call))
                                with self.assertRaises(type(error)):
                                    backend.cancel(1, "test")
                                self.assertTrue(cancelled.is_set())
                                self.assertEqual(calls, ["engine", "playback", "fallback"])
                        finally:
                            backend.close()

    def test_close_attempts_join_and_fallback_close_after_failure(self) -> None:
        for failing in ("cancel", "join", "fallback"):
            for error in (OSError("close failed"), KeyboardInterrupt()):
                with self.subTest(failing=failing, error=type(error).__name__):
                    backend = PocketTtsBackend(engine=FakeEngine(), fallback_backend=FakeFallback(), warm_on_start=False)
                    calls = []
                    try:
                        with ExitStack() as patches:
                            for name, target, method in (("cancel", backend, "cancel"), ("join", backend._thread, "join"), ("fallback", backend.fallback_backend, "close")):
                                def call(*args, operation=name, **kwargs):
                                    calls.append(operation)
                                    if operation == failing:
                                        raise error
                                patches.enter_context(mock.patch.object(target, method, side_effect=call))
                            with self.assertRaises(type(error)):
                                backend.close()
                            self.assertTrue(backend._stop.is_set())
                            self.assertEqual(calls, ["cancel", "join", "fallback"])
                    finally:
                        backend._stop.set()
                        backend._thread.join(1)
                        backend.fallback_backend.close()
                    self.assertFalse(backend._thread.is_alive())

    def test_default_warm_timeout_covers_qualified_cold_load_budget(self) -> None:
        parameter = inspect.signature(PocketTtsBackend).parameters[
            "warm_timeout_seconds"
        ]

        self.assertEqual(POCKET_TTS_WARM_TIMEOUT_SECONDS, 60.0)
        self.assertEqual(parameter.default, POCKET_TTS_WARM_TIMEOUT_SECONDS)

    def test_english_stream_starts_once_and_completion_observes_idle(self) -> None:
        engine = FakeEngine()
        fallback = FakeFallback()
        playback = FakePlayback()
        started: list[int] = []
        completed: list[tuple[int, bool]] = []
        backend_ref: dict[str, PocketTtsBackend] = {}
        backend = PocketTtsBackend(
            engine=engine,
            fallback_backend=fallback,
            playback_factory=lambda: playback,
            on_started=lambda unit: started.append(unit.unit_index),
            on_completed=lambda unit: completed.append(
                (unit.unit_index, backend_ref["backend"].is_idle)
            ),
            warm_on_start=False,
        )
        backend_ref["backend"] = backend
        self.addCleanup(backend.close)

        backend.enqueue(speech_unit())

        self.assertTrue(backend.wait_idle())
        self.assertEqual(engine.texts, ["All systems are ready."])
        self.assertEqual(started, [1])
        self.assertEqual(completed, [(1, True)])
        self.assertEqual(len(playback.chunks), 3)
        self.assertTrue(playback.finished)
        self.assertEqual(fallback.units, [])

    def test_ptt_cancel_aborts_playback_and_suppresses_completion(self) -> None:
        engine = FakeEngine(block=True)
        fallback = FakeFallback()
        playback = FakePlayback()
        completed: list[int] = []
        backend = PocketTtsBackend(
            engine=engine,
            fallback_backend=fallback,
            playback_factory=lambda: playback,
            on_completed=lambda unit: completed.append(unit.unit_index),
            warm_on_start=False,
        )
        self.addCleanup(backend.close)
        backend.enqueue(speech_unit())
        self.assertTrue(playback.started.wait(1.0))

        started = time.perf_counter()
        backend.cancel(1, "ptt_down")

        self.assertTrue(playback.aborted.wait(0.2))
        self.assertLess(time.perf_counter() - started, 0.2)
        self.assertTrue(backend.wait_idle())
        self.assertTrue(engine.cancelled.is_set())
        self.assertEqual(completed, [])

    def test_non_english_units_route_to_system_backend_without_model_use(self) -> None:
        engine = FakeEngine()
        fallback = FakeFallback()
        backend = PocketTtsBackend(
            engine=engine,
            fallback_backend=fallback,
            playback_factory=FakePlayback,
            warm_on_start=False,
        )
        self.addCleanup(backend.close)

        selected = speech_unit(locale="el-GR")
        backend.enqueue(selected)

        self.assertTrue(backend.wait_idle())
        self.assertEqual(fallback.units, [selected])
        self.assertEqual(engine.texts, [])

    def test_cancel_during_finish_suppresses_protocol_completion(self) -> None:
        finish_started = threading.Event()
        release_finish = threading.Event()
        completed: list[int] = []

        class FinishBlockingPlayback(FakePlayback):
            def finish(self) -> None:
                finish_started.set()
                release_finish.wait(1.0)
                self.finished = True

            def abort(self) -> None:
                super().abort()
                release_finish.set()

        playback = FinishBlockingPlayback()
        backend = PocketTtsBackend(
            engine=FakeEngine(),
            fallback_backend=FakeFallback(),
            playback_factory=lambda: playback,
            on_completed=lambda unit: completed.append(unit.unit_index),
            warm_on_start=False,
        )
        self.addCleanup(backend.close)
        backend.enqueue(speech_unit())
        self.assertTrue(finish_started.wait(0.5))

        backend.cancel(1, "ptt_down")

        self.assertTrue(backend.wait_idle())
        self.assertTrue(playback.aborted.is_set())
        self.assertEqual(completed, [])

    def test_warm_failure_permanently_degrades_to_fallback(self) -> None:
        fallback = FakeFallback()
        backend = PocketTtsBackend(
            engine=FakeEngine(fail_warm=True),
            fallback_backend=fallback,
            playback_factory=FakePlayback,
            warm_on_start=True,
        )
        self.addCleanup(backend.close)
        selected = speech_unit()

        backend.enqueue(selected)

        self.assertTrue(backend.wait_idle())
        self.assertTrue(backend.degraded_to_fallback)
        self.assertEqual(fallback.units, [selected])


class SoundDeviceStreamingPlaybackTests(unittest.TestCase):
    def test_stream_starts_only_after_initial_audio_is_buffered(self) -> None:
        started = threading.Event()
        calls: list[str] = []

        class Stream:
            def start(self):
                calls.append("start")
                started.set()
            def write(self, _value): calls.append("write")
            def stop(self): pass
            def abort(self): pass
            def close(self): pass

        module = SimpleNamespace(RawOutputStream=lambda **_kwargs: Stream())
        with mock.patch.dict(sys.modules, {"sounddevice": module}):
            playback = SoundDeviceStreamingPlayback(queue_capacity=4)
            playback.start(24_000)
            self.assertFalse(started.is_set())
            playback.write(b"one")
            playback.write(b"two")
            self.assertFalse(started.is_set())
            playback.write(b"three")
            self.assertTrue(started.wait(0.2))
            playback.finish()

        self.assertEqual(calls[:2], ["start", "write"])

    def test_writer_drains_chunks_in_order_before_finish(self) -> None:
        writes: list[bytes] = []

        class Stream:
            def start(self): pass
            def write(self, value): writes.append(value)
            def stop(self): pass
            def abort(self): pass
            def close(self): pass

        module = SimpleNamespace(RawOutputStream=lambda **_kwargs: Stream())
        with mock.patch.dict(sys.modules, {"sounddevice": module}):
            playback = SoundDeviceStreamingPlayback(queue_capacity=4)
            playback.start(24_000)
            playback.write(b"one")
            playback.write(b"two")
            playback.write(b"three")
            playback.finish()

        self.assertEqual(writes, [b"one", b"two", b"three"])

    def test_abort_unblocks_writer_and_discards_pending_audio(self) -> None:
        write_started = threading.Event()
        aborted = threading.Event()
        writes: list[bytes] = []

        class Stream:
            def start(self): pass
            def write(self, value):
                writes.append(value)
                write_started.set()
                aborted.wait(1.0)
            def stop(self): pass
            def abort(self): aborted.set()
            def close(self): pass

        module = SimpleNamespace(RawOutputStream=lambda **_kwargs: Stream())
        with mock.patch.dict(sys.modules, {"sounddevice": module}):
            playback = SoundDeviceStreamingPlayback(queue_capacity=4)
            playback.start(24_000)
            playback.write(b"active")
            playback.write(b"stale")
            playback.write(b"stale-two")
            self.assertTrue(write_started.wait(0.2))
            started = time.perf_counter()
            playback.abort()

        self.assertLess(time.perf_counter() - started, 0.2)
        self.assertTrue(aborted.is_set())
        self.assertEqual(writes, [b"active"])

    def test_abort_wins_race_with_finish_without_playback_error(self) -> None:
        write_started = threading.Event()
        aborted = threading.Event()
        finish_errors: list[BaseException] = []

        class Stream:
            def start(self): pass
            def write(self, _value):
                write_started.set()
                aborted.wait(1.0)
            def stop(self):
                if aborted.is_set():
                    raise RuntimeError("stopped after abort")
            def abort(self): aborted.set()
            def close(self): pass

        module = SimpleNamespace(RawOutputStream=lambda **_kwargs: Stream())
        with mock.patch.dict(sys.modules, {"sounddevice": module}):
            playback = SoundDeviceStreamingPlayback(queue_capacity=4)
            playback.start(24_000)
            playback.write(b"active")
            playback.write(b"stale")
            playback.write(b"stale-two")
            self.assertTrue(write_started.wait(0.2))

            def finish() -> None:
                try:
                    playback.finish()
                except BaseException as exc:
                    finish_errors.append(exc)

            thread = threading.Thread(target=finish)
            thread.start()
            playback.abort()
            thread.join(0.5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(finish_errors, [])


if __name__ == "__main__":
    unittest.main()
