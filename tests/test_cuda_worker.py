from __future__ import annotations

import hashlib
import tempfile
import threading
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace

import _bootstrap  # noqa: F401

from jarvis_voice.backends import AsrScore, AudioChunk, LanguageMetadata, TranscriptResult
from jarvis_voice.cuda_worker import CudaFasterWhisperBackend, CudaWorkerConfig, ProcessCudaWorkerSession, _safe_worker_error, run_cuda_worker
from jarvis_voice.errors import BackendError
from jarvis_voice.gpu import GpuDevice
from jarvis_voice.manifest import ModelFileSpec, ModelManifest


RESULT = TranscriptResult("Γεια σου.", LanguageMetadata("el", 0.9), AsrScore(0.8, -0.2))


class StaticInventory:
    def devices(self):
        return (GpuDevice(0, "GPU", 10_000, 8_000, "driver"),)


class MemoryConnection:
    def __init__(self, incoming: list[dict[str, object]]) -> None:
        self.incoming = list(incoming)
        self.sent: list[dict[str, object]] = []
        self.closed = False

    def send(self, frame): self.sent.append(frame)
    def recv(self): return self.incoming.pop(0)
    def close(self): self.closed = True


class FakeModel:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.calls: list[dict[str, object]] = []

    def transcribe(self, samples, **kwargs):
        del samples
        self.calls.append(dict(kwargs))
        if self.failure:
            raise self.failure
        segment = SimpleNamespace(text="Γεια σου.", avg_logprob=-0.2, tokens=[1, 2])
        return iter([segment]), SimpleNamespace(language="el", language_probability=0.9)


class FakeSession:
    def __init__(self, order: list[str], *, error: BackendError | None = None) -> None:
        self.order = order
        self.error = error
        self.metrics: list[dict[str, object]] = []
        self.started = False
        self.cancelled = False
        self.closed = False
        self.calls = 0

    @property
    def alive(self): return self.started and not self.closed and not self.cancelled
    def start(self): self.started = True; self.order.append("start")
    def transcribe(self, audio):
        del audio
        self.calls += 1; self.order.append("transcribe")
        if self.error: raise self.error
        return RESULT
    def cancel(self): self.cancelled = True; self.order.append("cancel")
    def close(self): self.closed = True; self.order.append("close")


class FakeProcess:
    def __init__(self, *, alive: bool = True) -> None:
        self._alive = alive
        self.terminated = False
    def start(self): pass
    def is_alive(self): return self._alive
    def terminate(self): self.terminated = True; self._alive = False
    def join(self, timeout=None): del timeout
    def kill(self): self._alive = False
    def close(self):
        if self._alive: raise ValueError("process still alive")


class PollConnection:
    def __init__(self, frames: list[dict[str, object]], process: FakeProcess, *, crash_on_transcribe: bool = False) -> None:
        self.frames = list(frames); self.process = process; self.crash_on_transcribe = crash_on_transcribe; self.sent = []; self.closed = False
    def poll(self, timeout): del timeout; return bool(self.frames)
    def recv(self): return self.frames.pop(0)
    def send(self, frame):
        self.sent.append(frame)
        if frame.get("type") == "transcribe" and not self.frames and self.crash_on_transcribe:
            self.process._alive = False
    def close(self): self.closed = True


class FakeContext:
    def __init__(self, parent: PollConnection, process: FakeProcess) -> None:
        self.parent = parent; self.process = process
    def Pipe(self, duplex=True): del duplex; return self.parent, MemoryConnection([])
    def Process(self, **kwargs): del kwargs; return self.process


class CudaWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        content = b"model"; (self.directory / "model.bin").write_bytes(content)
        self.manifest = ModelManifest("test", "revision", "fake", (ModelFileSpec("model.bin", len(content), hashlib.sha256(content).hexdigest()),))

    def config(self, maximum_audio_bytes: int = 1024) -> CudaWorkerConfig:
        return CudaWorkerConfig(str(self.directory), 0, "int8_float16", 1, 1, maximum_audio_bytes, self.manifest)

    def test_worker_revalidates_manifest_and_uses_explicit_local_cuda_arguments(self) -> None:
        audio = array("f", [0.1]).tobytes()
        connection = MemoryConnection([{"type": "transcribe", "request_id": "r", "sample_rate": 16_000, "audio": audio}, {"type": "shutdown"}])
        created = []
        model = FakeModel()
        run_cuda_worker(connection, self.config(), model_factory=lambda *args, **kwargs: created.append((args, kwargs)) or model)
        args, kwargs = created[0]
        self.assertEqual(args, (str(self.directory),))
        self.assertEqual(kwargs, {"device": "cuda", "device_index": 0, "compute_type": "int8_float16", "local_files_only": True})
        self.assertEqual(
            model.calls,
            [
                {
                    "vad_filter": False,
                    "task": "transcribe",
                    "beam_size": 5,
                    "condition_on_previous_text": False,
                }
            ],
        )
        self.assertEqual([frame["type"] for frame in connection.sent], ["ready", "result", "shutdown.complete"])

    def test_worker_uses_same_low_confidence_bilingual_tie_break_as_cpu(self) -> None:
        class AmbiguousModel(FakeModel):
            def transcribe(self, samples, **kwargs):
                del samples
                self.calls.append(dict(kwargs))
                language = kwargs.get("language")
                score = -0.2 if language == "el" else -0.8
                text = "καθαρός ουρανός" if language == "el" else "uncertain"
                segment = SimpleNamespace(text=text, avg_logprob=score, tokens=[1, 2])
                return iter([segment]), SimpleNamespace(
                    language="el",
                    language_probability=0.6,
                )

        audio = array("f", [0.1]).tobytes()
        connection = MemoryConnection(
            [
                {
                    "type": "transcribe",
                    "request_id": "r",
                    "sample_rate": 16_000,
                    "audio": audio,
                },
                {"type": "shutdown"},
            ]
        )
        model = AmbiguousModel()
        run_cuda_worker(
            connection,
            self.config(),
            model_factory=lambda *args, **kwargs: model,
        )
        result = connection.sent[1]
        self.assertEqual(result["text"], "καθαρός ουρανός")
        self.assertEqual(result["language_code"], "el")
        self.assertEqual(
            [call.get("language") for call in model.calls],
            [None, "el", "en"],
        )

    def test_manifest_failure_happens_before_worker_model_factory(self) -> None:
        self.config()
        (self.directory / "model.bin").write_bytes(b"changed-size")
        connection = MemoryConnection([]); calls = []
        run_cuda_worker(connection, self.config(), model_factory=lambda *a, **k: calls.append(a))
        self.assertEqual(calls, [])
        self.assertEqual(connection.sent[0]["code"], "model.size_mismatch")

    def test_parent_manifest_failure_prevents_session_construction(self) -> None:
        (self.directory / "model.bin").write_bytes(b"changed-size")
        calls = []
        with self.assertRaisesRegex(BackendError, "model.size_mismatch"):
            CudaFasterWhisperBackend(self.config(), inventory=StaticInventory(), residency="turn_scoped", minimum_free_vram_mb=0, reserve_for_other_workloads_mb=0, warm_restart_limit=0, session_factory=lambda config: calls.append(config))
        self.assertEqual(calls, [])

    def test_worker_bounds_ipc_and_classifies_oom(self) -> None:
        oversized = MemoryConnection([{"type": "transcribe", "request_id": "r", "sample_rate": 16_000, "audio": b"12345678"}, {"type": "shutdown"}])
        run_cuda_worker(oversized, self.config(maximum_audio_bytes=4), model_factory=lambda *a, **k: FakeModel())
        self.assertEqual(oversized.sent[1]["code"], "cuda.worker_invalid_audio")
        oom = MemoryConnection([{"type": "transcribe", "request_id": "r", "sample_rate": 16_000, "audio": array("f", [0]).tobytes()}, {"type": "shutdown"}])
        run_cuda_worker(oom, self.config(), model_factory=lambda *a, **k: FakeModel(RuntimeError("CUDA out of memory")))
        self.assertEqual(oom.sent[1]["code"], "cuda.oom")
        load_oom = MemoryConnection([])
        run_cuda_worker(load_oom, self.config(), model_factory=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")))
        self.assertEqual(load_oom.sent[0]["code"], "cuda.oom")

    def test_worker_error_frame_redacts_text_paths_and_device_details(self) -> None:
        sensitive = "secret transcript C:/private/model GPU-UUID process-command"
        frame = str(_safe_worker_error(RuntimeError(sensitive), "transcribe"))
        self.assertNotIn(sensitive, frame)
        self.assertEqual(frame, "{'type': 'error', 'code': 'cuda.transcription_failed', 'retryable': True}")

    def test_turn_scoped_worker_exits_before_result_returns(self) -> None:
        order: list[str] = []; session = FakeSession(order)
        backend = CudaFasterWhisperBackend(self.config(), inventory=StaticInventory(), residency="turn_scoped", minimum_free_vram_mb=1, reserve_for_other_workloads_mb=1, warm_restart_limit=0, session_factory=lambda config: session)
        self.assertEqual(backend.transcribe(AudioChunk(array("f", [1]), 16_000)), RESULT)
        self.assertEqual(order[:3], ["start", "transcribe", "close"])
        self.assertTrue(session.closed)

    def test_warm_worker_reuse_and_bounded_restart(self) -> None:
        order: list[str] = []; sessions: list[FakeSession] = []
        def factory(config):
            del config
            session = FakeSession(order); sessions.append(session); return session
        backend = CudaFasterWhisperBackend(self.config(), inventory=StaticInventory(), residency="warm", minimum_free_vram_mb=1, reserve_for_other_workloads_mb=0, warm_restart_limit=1, session_factory=factory)
        audio = AudioChunk(array("f", [1]), 16_000)
        backend.transcribe(audio); backend.transcribe(audio)
        self.assertEqual(len(sessions), 1); self.assertEqual(sessions[0].calls, 2)
        sessions[0].error = BackendError("cuda.worker_crashed", "failed", subsystem="cuda")
        with self.assertRaises(BackendError): backend.transcribe(audio)
        second = FakeSession(order, error=BackendError("cuda.worker_crashed", "failed", subsystem="cuda"))
        sessions.append(second); backend.session_factory = lambda config: second
        with self.assertRaises(BackendError): backend.transcribe(audio)
        with self.assertRaisesRegex(BackendError, "cuda.warm_restart_limit"): backend.transcribe(audio)

    def test_parent_detects_worker_crash_during_transcription(self) -> None:
        process = FakeProcess()
        parent = PollConnection([{"type": "ready", "load_duration_seconds": 0.1}], process, crash_on_transcribe=True)
        session = ProcessCudaWorkerSession(self.config(), context=FakeContext(parent, process))
        session.start()
        with self.assertRaisesRegex(BackendError, "cuda.worker_crashed"):
            session.transcribe(AudioChunk(array("f", [1]), 16_000))

    def test_parent_enforces_start_and_transcription_timeouts(self) -> None:
        start_process = FakeProcess()
        start_session = ProcessCudaWorkerSession(self.config(), context=FakeContext(PollConnection([], start_process), start_process))
        with self.assertRaisesRegex(BackendError, "cuda.worker_start_timeout"):
            start_session.start()
        transcribe_process = FakeProcess()
        transcribe_parent = PollConnection([{"type": "ready", "load_duration_seconds": 0.1}], transcribe_process)
        transcribe_session = ProcessCudaWorkerSession(self.config(), context=FakeContext(transcribe_parent, transcribe_process))
        transcribe_session.start()
        with self.assertRaisesRegex(BackendError, "cuda.transcription_timeout"):
            transcribe_session.transcribe(AudioChunk(array("f", [1]), 16_000))

    def test_cancel_current_terminates_active_cuda_session(self) -> None:
        order: list[str] = []; started = threading.Event(); released = threading.Event()
        class BlockingSession(FakeSession):
            def transcribe(self, audio):
                del audio; started.set(); released.wait(1)
                raise BackendError("cuda.worker_crashed", "cancelled", subsystem="cuda")
            def cancel(self):
                super().cancel(); released.set()
        session = BlockingSession(order)
        backend = CudaFasterWhisperBackend(self.config(), inventory=StaticInventory(), residency="turn_scoped", minimum_free_vram_mb=0, reserve_for_other_workloads_mb=0, warm_restart_limit=0, session_factory=lambda config: session)
        errors = []
        thread = threading.Thread(target=lambda: self._capture_error(errors, backend, AudioChunk(array("f", [1]), 16_000)))
        thread.start(); self.assertTrue(started.wait(1)); backend.cancel_current(); thread.join(2)
        self.assertTrue(session.cancelled); self.assertFalse(thread.is_alive()); self.assertEqual(errors[0].code, "cuda.worker_crashed")

    @staticmethod
    def _capture_error(errors, backend, audio) -> None:
        try:
            backend.transcribe(audio)
        except BackendError as exc:
            errors.append(exc)


if __name__ == "__main__":
    unittest.main()
