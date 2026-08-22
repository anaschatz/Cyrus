from __future__ import annotations

import sys
import threading
import time
import unittest
from array import array

import _bootstrap  # noqa: F401

from jarvis_voice.asr_factory import AsrBackendFactory, FallbackAsrBackend
from jarvis_voice.backends import AsrScore, AudioChunk, LanguageMetadata, TranscriptResult
from jarvis_voice.config import voice_config_from_mapping
from jarvis_voice.errors import BackendError
from jarvis_voice.gpu import GpuDevice
from jarvis_voice.transcription import AsyncTranscriptionCoordinator


RESULT = TranscriptResult("Open it.", LanguageMetadata("en", 0.9), AsrScore(0.8, -0.2))


class Backend:
    def __init__(self, result=RESULT, error: BackendError | None = None) -> None:
        self.result = result; self.error = error; self.calls = 0; self.cancelled = False
    def transcribe(self, audio):
        del audio; self.calls += 1
        if self.error: raise self.error
        return self.result
    def cancel_current(self): self.cancelled = True


class BlockingBackend(Backend):
    def __init__(self) -> None:
        super().__init__(); self.started = threading.Event(); self.release = threading.Event()
    def transcribe(self, audio):
        del audio; self.started.set(); self.release.wait(2); return self.result
    def cancel_current(self): self.cancelled = True


class Inventory:
    def devices(self): return (GpuDevice(0, "GPU", 10_000, 9_000, "driver"),)


class AsrFactoryTests(unittest.TestCase):
    def test_cpu_profile_does_not_import_cuda_worker(self) -> None:
        previous = sys.modules.pop("jarvis_voice.cuda_worker", None)
        try:
            cpu = Backend()
            factory = AsrBackendFactory(cpu_factory=lambda *a, **k: cpu)
            config = voice_config_from_mapping({"whisper_model_directory": "explicit-local-directory"})
            self.assertIs(factory.create(config), cpu)
            self.assertNotIn("jarvis_voice.cuda_worker", sys.modules)
        finally:
            if previous is not None:
                sys.modules["jarvis_voice.cuda_worker"] = previous

    def test_cuda_factory_passes_explicit_profile_without_changing_device(self) -> None:
        cpu = Backend(); captured = []
        def cuda_factory(worker_config, **kwargs):
            captured.append((worker_config, kwargs)); return Backend()
        config = voice_config_from_mapping({
            "whisper_model_directory": "explicit-local-directory",
            "asr": {"execution_profile": "cuda", "cuda": {"device_index": 0, "compute_type": "int8_float16"}},
        })
        backend = AsrBackendFactory(inventory_factory=Inventory, cpu_factory=lambda *a, **k: cpu, cuda_backend_factory=cuda_factory).create(config)
        worker, kwargs = captured[0]
        self.assertEqual((worker.model_directory, worker.device_index, worker.compute_type), ("explicit-local-directory", 0, "int8_float16"))
        self.assertIsInstance(backend, FallbackAsrBackend)
        self.assertEqual(kwargs["residency"], "turn_scoped")

    def test_turbo_manifest_is_shared_by_cuda_and_cpu_fallback(self) -> None:
        cpu_arguments = []
        cuda_arguments = []

        def cpu_factory(*args, **kwargs):
            cpu_arguments.append((args, kwargs))
            return Backend()

        def cuda_factory(worker_config, **kwargs):
            cuda_arguments.append((worker_config, kwargs))
            return Backend()

        config = voice_config_from_mapping(
            {
                "whisper_model_directory": "explicit-local-turbo-directory",
                "asr": {
                    "model_profile": "large-v3-turbo",
                    "execution_profile": "cuda",
                    "cuda": {
                        "device_index": 0,
                        "minimum_free_vram_mb": 2_048,
                        "reserve_for_other_workloads_mb": 2_048,
                    },
                },
            }
        )
        AsrBackendFactory(
            inventory_factory=Inventory,
            cpu_factory=cpu_factory,
            cuda_backend_factory=cuda_factory,
        ).create(config)
        cpu_manifest = cpu_arguments[0][1]["manifest"]
        cuda_manifest = cuda_arguments[0][0].manifest
        self.assertIs(cpu_manifest, cuda_manifest)
        self.assertEqual(
            cuda_manifest.model_id,
            "dropbox-dash/faster-whisper-large-v3-turbo",
        )
        self.assertEqual(len(cuda_manifest.files), 5)

    def test_fallback_uses_same_audio_once_and_never_emits_partial_cuda_result(self) -> None:
        cuda = Backend(error=BackendError("cuda.oom", "secret cuda details", subsystem="cuda")); cpu = Backend()
        backend = FallbackAsrBackend(cuda, cpu, enabled=True, fallbackable_codes=frozenset({"cuda.oom"}))
        audio = AudioChunk(array("f", [1]), 16_000)
        self.assertEqual(backend.transcribe(audio), RESULT)
        self.assertEqual((cuda.calls, cpu.calls, backend.fallbacks), (1, 1, 1))

    def test_fallback_disabled_and_dual_failure_are_typed(self) -> None:
        cuda = Backend(error=BackendError("cuda.oom", "failed", subsystem="cuda"))
        with self.assertRaisesRegex(BackendError, "cuda.oom"):
            FallbackAsrBackend(cuda, Backend(), enabled=False, fallbackable_codes=frozenset({"cuda.oom"})).transcribe(object())
        cpu = Backend(error=BackendError("asr.cpu_failed", "failed", subsystem="asr"))
        with self.assertRaisesRegex(BackendError, "asr.cpu_failed"):
            FallbackAsrBackend(cuda, cpu, enabled=True, fallbackable_codes=frozenset({"cuda.oom"})).transcribe(object())
        self.assertEqual((cuda.calls, cpu.calls), (2, 1))

    def test_admission_denial_respects_fallback_policy(self) -> None:
        denial = BackendError("cuda.insufficient_free_vram", "denied", subsystem="cuda")
        cpu = Backend(); audio = AudioChunk(array("f", [1]), 16_000)
        enabled = FallbackAsrBackend(Backend(error=denial), cpu, enabled=True, fallbackable_codes=frozenset({denial.code}))
        self.assertEqual(enabled.transcribe(audio), RESULT)
        with self.assertRaisesRegex(BackendError, "cuda.insufficient_free_vram"):
            FallbackAsrBackend(Backend(error=denial), Backend(), enabled=False, fallbackable_codes=frozenset({denial.code})).transcribe(audio)

    def test_cancelled_async_result_is_retired_and_never_published(self) -> None:
        backend = BlockingBackend(); coordinator = AsyncTranscriptionCoordinator(backend)
        self.addCleanup(coordinator.close)
        coordinator.submit("turn-old", AudioChunk(array("f", [1]), 16_000))
        self.assertTrue(backend.started.wait(1))
        coordinator.cancel_current(); backend.release.set()
        for _ in range(20):
            if coordinator.poll() is None:
                time.sleep(0.001)
        self.assertTrue(backend.cancelled)
        self.assertIsNone(coordinator.poll())


if __name__ == "__main__":
    unittest.main()
