from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace

import _bootstrap  # noqa: F401

from jarvis_voice.backends import SpeechUnit
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.greektts15_runtime import (
    GreekTtsF1Backend,
    MlxGreekTtsStreamingEngine,
    PcmChunk,
    SoundDevicePcmPlayback,
    load_approved_greektts_candidate,
    load_digest_bound_greektts_candidate,
    incremental_pcm_slice,
)


class FakeEngine:
    def __init__(self) -> None:
        self.units: list[SpeechUnit] = []
        self.cancelled = False
        self.first_chunk = threading.Event()
        self.release = threading.Event()

    def stream(self, unit: SpeechUnit, cancelled: threading.Event):
        self.units.append(unit)
        yield PcmChunk(b"\x01\x00" * 20)
        self.first_chunk.set()
        self.release.wait(1.0)
        if not cancelled.is_set():
            yield PcmChunk(b"\x02\x00" * 20)

    def cancel(self) -> None:
        self.cancelled = True
        self.release.set()


class FakeWarmEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.warm_thread: int | None = None

    def warm(self) -> float:
        self.warm_thread = threading.get_ident()
        return 12.5


class FakePlayback:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.finished = False
        self.aborted = False

    def write(self, chunk: PcmChunk) -> None:
        self.chunks.append(chunk.pcm16)

    def finish(self) -> None:
        self.finished = True

    def abort(self) -> None:
        self.aborted = True


class GreekTts15RuntimeTests(unittest.TestCase):
    def test_cancelled_generation_is_closed_and_metal_is_synchronized(self) -> None:
        cancelled = threading.Event()
        state = {"closed": False, "synchronized": 0, "cleared": 0}

        class Random:
            def seed(self, value): del value

        class Mx:
            random = Random()
            def array(self, value): return value
            def synchronize(self): state["synchronized"] += 1
            def clear_cache(self): state["cleared"] += 1

        class Tokenizer:
            def encode(self, text, add_special_tokens): del text, add_special_tokens; return [128_000]

        def generate_step(*args, **kwargs):
            del args, kwargs
            try:
                cancelled.set()
                yield 1, None
            finally:
                state["closed"] = True

        engine = object.__new__(MlxGreekTtsStreamingEngine)
        engine.seed = 1
        engine.maximum_new_tokens = 10
        engine.candidate = SimpleNamespace(temperature=0.4, top_p=0.95, repetition_penalty=1.1)
        engine._cancel_requested = threading.Event()
        engine._decoder_quarantined = threading.Event()
        engine._load_lock = threading.Lock()
        mx = Mx()
        engine._runtime = (
            mx,
            object(),
            object(),
            Tokenizer(),
            object(),
            generate_step,
            lambda **kwargs: kwargs,
            lambda **kwargs: kwargs,
        )
        self.assertEqual(list(engine.stream(SpeechUnit("r", 1, 1, "el-GR", "Δοκιμή."), cancelled)), [])
        self.assertTrue(state["closed"])
        self.assertEqual(state["synchronized"], 1)
        self.assertEqual(state["cleared"], 1)

    def test_quarantined_decoder_refuses_reuse_before_loading_the_model(self) -> None:
        engine = object.__new__(MlxGreekTtsStreamingEngine)
        engine._cancel_requested = threading.Event()
        engine._decoder_quarantined = threading.Event()
        engine._decoder_quarantined.set()
        unit = SpeechUnit("r", 1, 1, "el-GR", "Δοκιμή.")
        with self.assertRaisesRegex(BackendError, "tts.greektts15.decoder_quarantined"):
            list(engine.stream(unit, threading.Event()))

    def test_incremental_pcm_emits_only_new_stable_samples_and_smooths_boundary(self) -> None:
        pcm = array("h", [100, 200, 300, 400, 500, 600]).tobytes()
        chunk, emitted, last = incremental_pcm_slice(
            pcm,
            emitted_samples=2,
            stable_samples=6,
            previous_sample=-100,
            ramp_samples=2,
        )
        values = array("h")
        values.frombytes(chunk)
        self.assertEqual(values.tolist(), [100, 400, 500, 600])
        self.assertEqual(emitted, 6)
        self.assertEqual(last, 600)
        with self.assertRaisesRegex(BackendError, "tts.greektts15.incremental_pcm_invalid"):
            incremental_pcm_slice(pcm, emitted_samples=4, stable_samples=7, previous_sample=None)

    def test_warm_load_runs_on_the_persistent_inference_worker(self) -> None:
        engine = FakeWarmEngine()
        caller_thread = threading.get_ident()
        backend = GreekTtsF1Backend(
            engine=engine,
            playback_factory=FakePlayback,
            warm_on_start=True,
        )
        try:
            self.assertEqual(backend.warm_load_ms, 12.5)
            self.assertIsNotNone(engine.warm_thread)
            self.assertNotEqual(engine.warm_thread, caller_thread)
        finally:
            backend.close()

    def test_coreaudio_player_uses_low_latency_pcm_and_hard_abort(self) -> None:
        class Stream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.started = False
                self.aborted = False
                self.closed = False
                self.writes: list[bytes] = []

            def start(self): self.started = True
            def write(self, value): self.writes.append(value); return False
            def stop(self): pass
            def abort(self): self.aborted = True
            def close(self): self.closed = True

        class SoundDevice:
            def __init__(self): self.stream: Stream | None = None; self.settings = None
            def CoreAudioSettings(self, **kwargs): self.settings = kwargs; return kwargs
            def RawOutputStream(self, **kwargs): self.stream = Stream(**kwargs); return self.stream

        module = SoundDevice()
        playback = SoundDevicePcmPlayback(module)
        playback.write(PcmChunk(b"\x01\x00" * 20))
        playback.abort()
        self.assertIsNotNone(module.stream)
        assert module.stream is not None
        self.assertTrue(module.stream.started)
        self.assertEqual(module.stream.kwargs["latency"], "low")
        self.assertEqual(module.stream.kwargs["blocksize"], 0)
        self.assertEqual(module.stream.writes, [b"\x01\x00" * 20])
        self.assertTrue(module.stream.aborted)
        self.assertTrue(module.stream.closed)

    def test_coreaudio_abort_waits_only_for_the_current_bounded_pcm_write(self) -> None:
        writing = threading.Event()

        class Stream:
            def start(self): pass
            def write(self, value):
                del value
                writing.set()
                time.sleep(0.02)
                return False
            def stop(self): pass
            def abort(self): pass
            def close(self): pass

        class SoundDevice:
            def CoreAudioSettings(self, **kwargs): return kwargs
            def RawOutputStream(self, **kwargs):
                del kwargs
                return Stream()

        playback = SoundDevicePcmPlayback(SoundDevice())
        writer = threading.Thread(target=lambda: playback.write(PcmChunk(b"\x01\x00" * 20)))
        writer.start()
        self.assertTrue(writing.wait(1.0))
        started = time.perf_counter()
        playback.abort()
        writer.join(1.0)
        self.assertLess((time.perf_counter() - started) * 1000.0, 200.0)
        self.assertFalse(writer.is_alive())

    def test_coreaudio_splits_generated_audio_into_twenty_millisecond_writes(self) -> None:
        class Stream:
            def __init__(self): self.writes: list[bytes] = []
            def start(self): pass
            def write(self, value): self.writes.append(value); return False
            def stop(self): pass
            def abort(self): pass
            def close(self): pass

        class SoundDevice:
            def __init__(self): self.stream = Stream()
            def CoreAudioSettings(self, **kwargs): return kwargs
            def RawOutputStream(self, **kwargs):
                del kwargs
                return self.stream

        module = SoundDevice()
        playback = SoundDevicePcmPlayback(module)
        playback.write(PcmChunk(b"\x01\x00" * 1_000))
        playback.finish()
        self.assertEqual([len(value) for value in module.stream.writes], [960, 960, 80])

    def test_approved_manifest_is_bound_to_exact_adapter_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoints" / "approved"
            checkpoint.mkdir(parents=True)
            adapter = checkpoint / "adapters.safetensors"
            adapter.write_bytes(b"approved-adapter")
            (checkpoint / "adapter_config.json").write_text("{}")
            digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
            manifest = root / "approved.json"
            manifest.write_text(json.dumps({
                "schema": "jarvis.voice.greektts15-approved-candidate/1",
                "status": "human_voice_approved",
                "model": {
                    "checkpoint_directory": "checkpoints/approved",
                    "checkpoint_sha256": digest,
                    "sampling": {"temperature": 0.4, "top_p": 0.95, "repetition_penalty": 1.1},
                },
                "human_voice_approved": True,
                "production_promotion_allowed": False,
            }))
            candidate = load_approved_greektts_candidate(manifest.resolve(), root.resolve())
            adapter.write_bytes(b"changed")
            with self.assertRaises(BackendError) as raised:
                load_approved_greektts_candidate(manifest.resolve(), root.resolve())
        self.assertEqual(candidate.checkpoint_sha256, digest)
        self.assertEqual(raised.exception.code, "tts.greektts15.approved_checkpoint_invalid")

    def test_fixture_only_manifest_is_rejected_by_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "fixture-only.json"
            manifest.write_text(json.dumps({
                "schema": "jarvis.voice.greektts15-approved-candidate/1",
                "status": "fixture_only_approval",
                "model": {},
                "human_voice_approved": False,
                "production_promotion_allowed": False,
            }))
            with self.assertRaises(BackendError) as raised:
                load_approved_greektts_candidate(manifest.resolve(), root.resolve())
        self.assertEqual(raised.exception.code, "tts.greektts15.approval_manifest_invalid")

    def test_runtime_config_binds_the_approval_manifest_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            adapter = checkpoint / "adapters.safetensors"
            adapter.write_bytes(b"approved")
            (checkpoint / "adapter_config.json").write_text("{}")
            manifest = root / "approved.json"
            manifest.write_text(json.dumps({
                "schema": "jarvis.voice.greektts15-approved-candidate/1",
                "status": "human_voice_approved",
                "model": {
                    "checkpoint_directory": "checkpoint",
                    "checkpoint_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
                    "sampling": {"temperature": 0.4, "top_p": 0.95, "repetition_penalty": 1.1},
                },
                "human_voice_approved": True,
                "production_promotion_allowed": False,
            }))
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            candidate = load_digest_bound_greektts_candidate(manifest.resolve(), digest, root.resolve())
            manifest.write_text(manifest.read_text() + "\n")
            with self.assertRaises(BackendError) as raised:
                load_digest_bound_greektts_candidate(manifest.resolve(), digest, root.resolve())
        self.assertEqual(candidate.checkpoint_sha256, hashlib.sha256(b"approved").hexdigest())
        self.assertEqual(raised.exception.code, "tts.greektts15.approval_manifest_changed")

    def test_sentence_units_are_ordered_and_complete_only_after_playback(self) -> None:
        engine = FakeEngine()
        engine.release.set()
        playbacks: list[FakePlayback] = []
        completed: list[int] = []
        backend = GreekTtsF1Backend(
            engine=engine,
            playback_factory=lambda: playbacks.append(FakePlayback()) or playbacks[-1],
            on_completed=lambda unit: completed.append(unit.unit_index),
        )
        backend.enqueue(SpeechUnit("r", 1, 1, "el-GR", "Πρώτη πρόταση."))
        backend.enqueue(SpeechUnit("r", 1, 2, "el-GR", "Δεύτερη πρόταση."))
        self.assertTrue(backend.wait_idle())
        backend.close()
        self.assertEqual([unit.unit_index for unit in engine.units], [1, 2])
        self.assertEqual(completed, [1, 2])
        self.assertTrue(all(playback.finished for playback in playbacks))

    def test_started_callback_precedes_the_first_blocking_pcm_write(self) -> None:
        engine = FakeEngine()
        engine.release.set()
        order: list[str] = []

        class OrderedPlayback(FakePlayback):
            def write(self, chunk: PcmChunk) -> None:
                order.append("write")
                super().write(chunk)

            def finish(self) -> None:
                order.append("finish")
                super().finish()

        backend = GreekTtsF1Backend(
            engine=engine,
            playback_factory=OrderedPlayback,
            on_started=lambda unit: order.append("started"),
            on_completed=lambda unit: order.append("completed"),
        )
        backend.enqueue(SpeechUnit("r", 1, 1, "el-GR", "Δοκιμή."))
        self.assertTrue(backend.wait_idle())
        backend.close()
        self.assertEqual(order, ["started", "write", "write", "finish", "completed"])

    def test_generation_continues_while_pcm_writer_is_blocked(self) -> None:
        generated = threading.Event()
        write_started = threading.Event()
        release = threading.Event()

        class StreamingEngine(FakeEngine):
            def stream(self, unit: SpeechUnit, cancelled: threading.Event):
                del unit, cancelled
                yield PcmChunk(b"\x01\x00" * 20)
                yield PcmChunk(b"\x02\x00" * 20)
                generated.set()

        class SlowPlayback(FakePlayback):
            def write(self, chunk: PcmChunk) -> None:
                write_started.set()
                release.wait(1.0)
                super().write(chunk)

        backend = GreekTtsF1Backend(engine=StreamingEngine(), playback_factory=SlowPlayback)
        backend.enqueue(SpeechUnit("r", 1, 1, "el-GR", "Δοκιμή."))
        self.assertTrue(write_started.wait(1.0))
        self.assertTrue(generated.wait(1.0))
        release.set()
        self.assertTrue(backend.wait_idle())
        backend.close()

    def test_cancel_aborts_playback_and_rejects_every_late_chunk_and_completion(self) -> None:
        engine = FakeEngine()
        playback = FakePlayback()
        completed: list[int] = []
        backend = GreekTtsF1Backend(
            engine=engine,
            playback_factory=lambda: playback,
            on_completed=lambda unit: completed.append(unit.unit_index),
        )
        backend.enqueue(SpeechUnit("r", 1, 1, "el-GR", "Ενεργή πρόταση."))
        self.assertTrue(engine.first_chunk.wait(1.0))
        backend.enqueue(SpeechUnit("r", 1, 2, "el-GR", "Επόμενη πρόταση."))
        backend.cancel(1, "ptt_down")
        self.assertTrue(backend.wait_idle())
        backend.close()
        self.assertTrue(engine.cancelled)
        self.assertTrue(playback.aborted)
        self.assertEqual(len(playback.chunks), 1)
        self.assertEqual(completed, [])

    def test_expected_late_backend_exception_after_cancel_is_not_a_runtime_error(self) -> None:
        class LateFailureEngine(FakeEngine):
            def stream(self, unit: SpeechUnit, cancelled: threading.Event):
                del unit, cancelled
                yield PcmChunk(b"\x01\x00" * 20)
                self.first_chunk.set()
                self.release.wait(1.0)
                raise RuntimeError("abort released the backend")

        engine = LateFailureEngine()
        backend = GreekTtsF1Backend(engine=engine, playback_factory=FakePlayback)
        backend.enqueue(SpeechUnit("r", 1, 1, "el-GR", "Ενεργή πρόταση."))
        self.assertTrue(engine.first_chunk.wait(1.0))
        backend.cancel(1, "ptt_down")
        self.assertTrue(backend.wait_idle())
        self.assertIsNone(backend.last_error)
        backend.close()

    def test_queue_overflow_is_fail_closed(self) -> None:
        engine = FakeEngine()
        backend = GreekTtsF1Backend(engine=engine, playback_factory=FakePlayback, queue_capacity=1)
        backend.enqueue(SpeechUnit("r", 1, 1, "el-GR", "Πρώτη."))
        self.assertTrue(engine.first_chunk.wait(1.0))
        backend.enqueue(SpeechUnit("r", 1, 2, "el-GR", "Δεύτερη."))
        with self.assertRaises(BackendError) as raised:
            backend.enqueue(SpeechUnit("r", 1, 3, "el-GR", "Τρίτη."))
        backend.cancel(1, "test")
        backend.close()
        self.assertEqual(raised.exception.code, "tts.queue_overflow")


if __name__ == "__main__":
    unittest.main()
