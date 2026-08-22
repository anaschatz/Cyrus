from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from jarvis_voice.backends import AsrScore, LanguageMetadata, TranscriptResult
from jarvis_voice.config import voice_config_from_mapping
from jarvis_voice.cuda_probe import run_active_cuda_probe
from jarvis_voice.gpu import GpuDevice


class Inventory:
    def devices(self): return (GpuDevice(0, "GPU", 8192, 7000, "driver"),)


class ProbeBackend:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.metrics = [
            {"event": "vram_before_worker", "value_mb": 7000},
            {"event": "vram_after_model_load", "value_mb": 6200},
        ]
    def transcribe(self, audio):
        del audio
        return TranscriptResult("must-not-escape", LanguageMetadata("en", 0.9), AsrScore(0.8, -0.2))
    def close(self): self.metrics.append({"event": "vram_after_worker_exit", "value_mb": 6990})


class ActiveCudaProbeTests(unittest.TestCase):
    def test_probe_is_local_redacted_and_marks_peak_as_sampled(self) -> None:
        config = voice_config_from_mapping({
            "whisper_model_directory": "explicit-local-directory",
            "asr": {"execution_profile": "cuda", "cuda": {"device_index": 0, "vram_poll_interval_ms": 10}},
        })
        report = run_active_cuda_probe(config, inventory=Inventory(), backend_factory=ProbeBackend)
        self.assertEqual(report["status"], "ready")
        self.assertFalse(report["transcript_exposed"])
        self.assertFalse(report["network_used"])
        self.assertTrue(report["peak_is_sampled_not_absolute"])
        self.assertNotIn("must-not-escape", str(report))
        self.assertEqual(report["vram_free_samples_mb"]["after_worker_exit"], 6990)

    def test_turbo_active_probe_receives_the_selected_manifest(self) -> None:
        captured = []

        class CapturingProbeBackend(ProbeBackend):
            def __init__(self, worker_config, **kwargs) -> None:
                captured.append(worker_config)
                super().__init__(worker_config, **kwargs)

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
        run_active_cuda_probe(
            config,
            inventory=Inventory(),
            backend_factory=CapturingProbeBackend,
        )
        self.assertEqual(
            captured[0].manifest.model_id,
            "dropbox-dash/faster-whisper-large-v3-turbo",
        )


if __name__ == "__main__":
    unittest.main()
