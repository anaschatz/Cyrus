from __future__ import annotations

import hashlib
import json
import queue
import tempfile
import threading
import time
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.backends import SpeechUnit
from jarvis_voice.errors import BackendError
from jarvis_voice.macos import greektts15_runtime as runtime
from jarvis_voice.macos.greektts15_runtime import (
    GreekTtsF1Backend,
    IsolatedSoundDevicePcmPlayback,
    MlxGreekTtsStreamingEngine,
    PcmChunk,
    SoundDevicePcmPlayback,
    load_approved_greektts_candidate,
    load_digest_bound_greektts_candidate,
    incremental_pcm_slice,
    wait_for_decoded_result,
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
    @staticmethod
    def _fake_playback_process_context():
        class Connection:
            def __init__(self, *, child=False):
                self.child = child
                self.incoming = [] if child else [b"R"]
                self.sent: list[bytes] = []
                self.closed = False

            def poll(self, timeout):
                del timeout
                return bool(self.incoming)

            def recv_bytes(self):
                return self.incoming.pop(0)

            def send_bytes(self, value):
                self.sent.append(value)
                if value == b"F":
                    self.incoming.append(b"C")

            def close(self):
                self.closed = True

        class Process:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.alive = False
                self.terminated = False
                self.killed = False

            def start(self): self.alive = True
            def is_alive(self): return self.alive
            def terminate(self): self.terminated = True; self.alive = False
            def kill(self): self.killed = True; self.alive = False
            def join(self, timeout=None): del timeout; self.alive = False

        class Context:
            def __init__(self):
                self.parent = Connection()
                self.child = Connection(child=True)
                self.process = None

            def Pipe(self, duplex):
                self.duplex = duplex
                return self.parent, self.child

            def Process(self, **kwargs):
                self.process = Process(**kwargs)
                return self.process

        return Context()

    def test_isolated_coreaudio_proxies_pcm_and_finishes_helper(self) -> None:
        context = self._fake_playback_process_context()
        playback = IsolatedSoundDevicePcmPlayback(process_context=context)
        playback.write(PcmChunk(b"\x01\x00" * 20))
        playback.finish()
        self.assertEqual(context.parent.sent, [b"D" + b"\x01\x00" * 20, b"F"])
        self.assertTrue(context.child.closed)
        self.assertTrue(context.parent.closed)
        self.assertFalse(context.process.is_alive())
        self.assertEqual(context.process.kwargs["target"].__name__, "_coreaudio_playback_worker")

    def test_isolated_coreaudio_abort_terminates_helper_without_native_abort(self) -> None:
        context = self._fake_playback_process_context()
        playback = IsolatedSoundDevicePcmPlayback(process_context=context)
        playback.abort()
        self.assertTrue(context.process.terminated)
        self.assertFalse(context.process.killed)
        self.assertTrue(context.parent.closed)

    def test_isolated_playback_startup_failures_release_owned_resources(self) -> None:
        for phase in ("factory", "start", "poll", "recv"):
            with self.subTest(phase=phase):
                context = self._fake_playback_process_context()
                context.parent, context.child = runtime.multiprocessing.Pipe(duplex=True)
                context.child.send_bytes(b"R")
                create_process = context.Process

                def process_factory(**kwargs):
                    if phase == "factory":
                        raise OSError("synthetic process construction failure")
                    process = create_process(**kwargs)
                    if phase == "start":
                        process.start = mock.Mock(side_effect=OSError("synthetic process start failure"))
                    return process

                context.Process = process_factory
                if phase in ("poll", "recv"):
                    operation = "poll" if phase == "poll" else "recv_bytes"
                    setattr(context.parent, operation, mock.Mock(side_effect=OSError("synthetic IPC failure")))
                try:
                    with self.assertRaises((OSError, BackendError)):
                        IsolatedSoundDevicePcmPlayback(process_context=context)
                    self.assertTrue(context.child.closed)
                    self.assertTrue(context.parent.closed)
                    if context.process is not None:
                        self.assertFalse(context.process.is_alive())
                finally:
                    context.child.close()
                    context.parent.close()

    def test_isolated_playback_requires_finite_timeouts_before_creating_pipes(self) -> None:
        for parameter in ("startup_timeout_seconds", "shutdown_timeout_seconds"):
            for value in (float("nan"), float("inf")):
                with self.subTest(parameter=parameter, value=value):
                    context = self._fake_playback_process_context()
                    with self.assertRaises(ValueError):
                        playback = IsolatedSoundDevicePcmPlayback(process_context=context, **{parameter: value})
                        playback.abort()
                    self.assertIsNone(context.process)

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

    def test_final_decoder_wait_survives_a_transient_empty_queue(self) -> None:
        decoded: queue.Queue[PcmChunk | Exception | None] = queue.Queue()
        expected = PcmChunk(b"\x01\x00")

        def publish_later() -> None:
            time.sleep(0.03)
            decoded.put(expected)

        decoder = threading.Thread(target=publish_later)
        decoder.start()
        result = wait_for_decoded_result(
            decoded,
            decoder,
            threading.Event(),
            deadline=time.monotonic() + 0.2,
            poll_seconds=0.01,
        )
        decoder.join()
        self.assertEqual(result, expected)

    def test_final_decoder_wait_is_bounded(self) -> None:
        decoded: queue.Queue[PcmChunk | Exception | None] = queue.Queue()
        release = threading.Event()
        decoder = threading.Thread(target=lambda: release.wait(0.2))
        decoder.start()
        try:
            with self.assertRaisesRegex(BackendError, "tts.greektts15.decoder_timeout"):
                wait_for_decoded_result(
                    decoded,
                    decoder,
                    threading.Event(),
                    deadline=time.monotonic() + 0.03,
                    poll_seconds=0.01,
                )
        finally:
            release.set()
            decoder.join()

    def test_final_decoder_wait_rejects_silent_worker_exit(self) -> None:
        decoded: queue.Queue[PcmChunk | Exception | None] = queue.Queue()
        decoder = threading.Thread(target=lambda: None)
        decoder.start()
        decoder.join()

        with self.assertRaisesRegex(BackendError, "tts.greektts15.decoder_stopped"):
            wait_for_decoded_result(
                decoded,
                decoder,
                threading.Event(),
                deadline=time.monotonic() + 0.2,
                poll_seconds=0.01,
            )

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
        self.assertIsNotNone(module.stream)
        assert module.stream is not None
        self.assertTrue(module.stream.started)
        self.assertEqual(module.stream.kwargs["latency"], "low")
        self.assertEqual(module.stream.kwargs["blocksize"], 0)
        playback.abort()
        self.assertEqual(module.stream.writes, [b"\x01\x00" * 20])
        self.assertTrue(module.stream.aborted)
        self.assertTrue(module.stream.closed)

    def test_coreaudio_abort_prevents_writer_lock_starvation(self) -> None:
        first_write = threading.Event()

        class Stream:
            def start(self): pass
            def write(self, value):
                del value
                first_write.set()
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
        writer = threading.Thread(target=lambda: playback.write(PcmChunk(b"\x01\x00" * 4_800)))
        writer.start()
        self.assertTrue(first_write.wait(1.0))
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

    def test_native_stream_errors_still_close_the_stream(self) -> None:
        for operation in ("start", "stop", "abort"):
            with self.subTest(operation=operation):
                stream = SimpleNamespace(start=mock.Mock(), stop=mock.Mock(), abort=mock.Mock(), close=mock.Mock())
                getattr(stream, operation).side_effect = OSError("synthetic stream failure")
                module = SimpleNamespace(CoreAudioSettings=lambda **kwargs: kwargs, RawOutputStream=lambda **kwargs: stream)
                with self.assertRaises(OSError):
                    playback = SoundDevicePcmPlayback(module)
                    if operation == "stop":
                        playback.finish()
                    elif operation == "abort":
                        playback.abort()
                stream.close.assert_called_once()

    def test_helper_closes_connection_even_when_playback_abort_fails(self) -> None:
        playback = SimpleNamespace(abort=mock.Mock(side_effect=OSError("synthetic abort failure")))
        connection = SimpleNamespace(send_bytes=mock.Mock(), recv_bytes=mock.Mock(side_effect=EOFError), close=mock.Mock())
        with mock.patch.object(runtime, "SoundDevicePcmPlayback", return_value=playback):
            with self.assertRaises(OSError):
                runtime._coreaudio_playback_worker(connection)
        connection.close.assert_called_once()

    def test_backend_close_joins_worker_when_playback_abort_fails(self) -> None:
        engine, playback = FakeEngine(), FakePlayback()
        backend = GreekTtsF1Backend(engine=engine, playback_factory=lambda: playback)
        try:
            backend.enqueue(SpeechUnit("r", 1, 1, "el-GR", "fixture"))
            self.assertTrue(engine.first_chunk.wait(1))
            with mock.patch.object(playback, "abort", side_effect=OSError("synthetic abort failure")), mock.patch.object(backend._thread, "join", wraps=backend._thread.join) as join:
                with self.assertRaises(OSError):
                    backend.close()
                join.assert_called_once_with(timeout=2.0)
            self.assertFalse(backend._thread.is_alive())
        finally:
            backend.close()

    def test_fatal_generation_or_playback_setup_stops_queue_admission(self) -> None:
        for phase in ("generation", "playback"):
            with self.subTest(phase=phase):
                release, units = threading.Event(), []
                def stream(unit, cancelled):
                    units.append(unit)
                    if not release.wait(1):
                        raise OSError("test did not release engine")
                    if phase == "generation":
                        raise OSError("synthetic generation failure")
                    yield PcmChunk(b"\0\0")
                engine = SimpleNamespace(stream=stream, cancel=release.set)
                backend = GreekTtsF1Backend(engine=engine, playback_factory=mock.Mock(side_effect=OSError("synthetic playback setup failure")))
                try:
                    backend.enqueue(SpeechUnit("r", 1, 1, "el-GR", "first"))
                    backend.enqueue(SpeechUnit("r", 1, 2, "el-GR", "second"))
                    release.set()
                    backend._thread.join(timeout=0.2)
                    self.assertEqual(len(units), 1)
                    self.assertEqual(backend.last_error.code, "tts.greektts15.runtime_failed")
                    with self.assertRaises(BackendError) as raised:
                        backend.enqueue(SpeechUnit("r", 1, 3, "el-GR", "third"))
                    self.assertIs(raised.exception, backend.last_error)
                finally:
                    release.set()
                    backend.close()

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
            approved = manifest.read_text()
            for name, value in (("temperature", float("nan")), ("top_p", float("inf")), ("repetition_penalty", float("-inf")), ("temperature", 10**400)):
                with self.subTest(sampling=name, value=value):
                    document = json.loads(approved)
                    document["model"]["sampling"][name] = value
                    manifest.write_text(json.dumps(document))
                    with self.assertRaises(BackendError) as invalid:
                        load_approved_greektts_candidate(manifest.resolve(), root.resolve())
                    self.assertEqual(invalid.exception.code, "tts.greektts15.approval_manifest_invalid")
            manifest.write_text(approved)
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
            changed = json.loads(manifest.read_text())
            changed["model"]["sampling"]["temperature"] = 0.8
            open_file, reads = Path.open, []

            def replace_on_second_read(path, mode="r", *args, **kwargs):
                if path == manifest.resolve() and "r" in mode:
                    reads.append(path)
                    if len(reads) == 2:
                        with open_file(path, "w") as output:
                            json.dump(changed, output)
                return open_file(path, mode, *args, **kwargs)

            with mock.patch.object(Path, "open", replace_on_second_read):
                bound = load_digest_bound_greektts_candidate(manifest.resolve(), digest, root.resolve())
            self.assertEqual(bound.temperature, candidate.temperature)
            manifest.write_text(manifest.read_text() + "\n")
            with self.assertRaises(BackendError) as raised:
                load_digest_bound_greektts_candidate(manifest.resolve(), digest, root.resolve())
        self.assertEqual(candidate.checkpoint_sha256, hashlib.sha256(b"approved").hexdigest())
        self.assertEqual(raised.exception.code, "tts.greektts15.approval_manifest_changed")

    def test_invalid_manifest_encoding_and_nesting_are_typed_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            manifest = root / "approved.json"
            for content in (b"\xff", b"[" * 1100 + b"0" + b"]" * 1100, b"{\"value\":" + b"1" * 5000 + b"}"):
                manifest.write_bytes(content)
                digest = hashlib.sha256(content).hexdigest()
                for bound in (False, True):
                    with self.subTest(content=content[:8], digest_bound=bound):
                        with self.assertRaises(BackendError) as raised:
                            if bound:
                                load_digest_bound_greektts_candidate(manifest, digest, root)
                            else:
                                load_approved_greektts_candidate(manifest, root)
                        self.assertEqual(raised.exception.code, "tts.greektts15.approval_manifest_invalid")

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
