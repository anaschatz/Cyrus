from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.config import is_loopback_websocket_url, load_voice_config, voice_config_from_mapping
from jarvis_voice.doctor import doctor_report_config
from jarvis_voice.errors import BackendError
from jarvis_voice.gpu import GpuDevice


class StaticInventory:
    def devices(self):
        return (GpuDevice(0, "Test GPU", 8192, 6144, "driver"),)


class ConfigDoctorTests(unittest.TestCase):
    def test_only_loopback_websockets_are_allowed(self) -> None:
        for url in ("ws://127.0.0.1:8765", "ws://127.0.0.1:8765/"):
            self.assertTrue(is_loopback_websocket_url(url))
        for url in (
            "https://127.0.0.1",
            "wss://127.0.0.1:8765",
            "ws://localhost:8765",
            "ws://[::1]:8765",
            "ws://127.0.0.2:8765",
            "ws://192.168.1.10:8765",
            "ws://user:password@127.0.0.1:8765",
            "ws://127.0.0.1:8765/voice",
            "ws://127.0.0.1:8765/?mode=voice",
            "ws://127.0.0.1:8765/#voice",
            "ws://127.0.0.1:0",
            "ws://127.0.0.1:65536",
        ):
            self.assertFalse(is_loopback_websocket_url(url))
            with self.assertRaisesRegex(BackendError, "config.non_loopback_orchestrator"):
                voice_config_from_mapping({"orchestrator_url": url})

    def test_voice_auth_key_tilde_is_normalized_to_an_absolute_path(self) -> None:
        configured = "~/.jarvis-test/voice-auth.json"
        config = voice_config_from_mapping({"voice_auth_key_file": configured})

        self.assertEqual(
            config.voice_auth_key_file,
            str(Path(configured).expanduser()),
        )
        self.assertTrue(Path(config.voice_auth_key_file).is_absolute())

    def test_doctor_distinguishes_missing_configuration_from_invalid_key(self) -> None:
        missing = doctor_report_config(voice_config_from_mapping({}))
        invalid = doctor_report_config(
            voice_config_from_mapping(
                {"voice_auth_key_file": "/absolute/missing/voice-auth.json"}
            )
        )

        self.assertEqual(
            missing["checks"]["transport"]["code"],
            "transport.authentication_key_required",
        )
        self.assertEqual(
            invalid["checks"]["transport"]["code"],
            "transport.authentication_key_invalid",
        )

    def test_safe_example_shape_round_trips(self) -> None:
        config = load_voice_config(Path(__file__).resolve().parents[1] / "config.example.json")
        self.assertEqual(config.ptt_key, "F9")
        self.assertEqual(config.orchestrator_url, "ws://127.0.0.1:8765")
        self.assertEqual(config.asr.model_profile, "base")
        self.assertEqual(config.asr.execution_profile, "cpu")
        self.assertIsNone(config.asr.cuda.device_index)
        self.assertEqual(config.asr.cuda.residency, "turn_scoped")
        self.assertEqual(config.application.shutdown_timeout_seconds, 5.0)
        self.assertEqual(config.application.log_maximum_bytes, 1_048_576)

    def test_invalid_capacity_is_typed(self) -> None:
        with self.assertRaisesRegex(BackendError, "config.invalid_field"):
            voice_config_from_mapping({"ptt_queue_capacity": 0})
        with self.assertRaisesRegex(BackendError, "config.invalid_field"):
            voice_config_from_mapping({"application": {"shutdown_timeout_seconds": 0}})

    def test_ptt_function_key_is_configurable(self) -> None:
        self.assertEqual(voice_config_from_mapping({"ptt_key": "f24"}).ptt_key, "F24")
        self.assertEqual(
            voice_config_from_mapping({"ptt_key": "control_option"}).ptt_key,
            "CONTROL_OPTION",
        )
        self.assertEqual(
            voice_config_from_mapping({"ptt_key": "right_option"}).ptt_key,
            "RIGHT_OPTION",
        )
        with self.assertRaisesRegex(BackendError, "config.unsupported_ptt_key"):
            voice_config_from_mapping({"ptt_key": "F25"})

    def test_cuda_profile_requires_explicit_device_and_valid_residency(self) -> None:
        with self.assertRaisesRegex(BackendError, "config.cuda_device_required"):
            voice_config_from_mapping({"asr": {"execution_profile": "cuda"}})
        with self.assertRaisesRegex(BackendError, "config.invalid_field"):
            voice_config_from_mapping({"asr": {"cuda": {"residency": "automatic"}}})
        config = voice_config_from_mapping({"asr": {"execution_profile": "cuda", "cuda": {"device_index": 0}}})
        self.assertEqual(config.asr.cuda.device_index, 0)

    def test_large_v3_turbo_cuda_profile_is_closed_and_llm_safe(self) -> None:
        with self.assertRaisesRegex(BackendError, "config.invalid_field"):
            voice_config_from_mapping({"asr": {"model_profile": "latest"}})
        with self.assertRaisesRegex(
            BackendError,
            "config.large_v3_turbo_cuda_profile_required",
        ):
            voice_config_from_mapping(
                {
                    "asr": {
                        "model_profile": "large-v3-turbo",
                        "execution_profile": "cuda",
                        "cuda": {"device_index": 0},
                    }
                }
            )
        config = voice_config_from_mapping(
            {
                "asr": {
                    "model_profile": "large-v3-turbo",
                    "execution_profile": "cuda",
                    "cuda": {
                        "device_index": 0,
                        "compute_type": "int8_float16",
                        "minimum_free_vram_mb": 2_048,
                        "reserve_for_other_workloads_mb": 2_048,
                        "fallback_to_cpu": True,
                    },
                }
            }
        )
        self.assertEqual(config.asr.model_profile, "large-v3-turbo")
        self.assertEqual(config.asr.cuda.minimum_free_vram_mb, 2_048)

    def test_passive_cuda_doctor_never_runs_active_probe(self) -> None:
        config = voice_config_from_mapping({
            "whisper_model_directory": "/private/secret-model",
            "asr": {"execution_profile": "cuda", "cuda": {"device_index": 0, "reserve_for_other_workloads_mb": 2048, "minimum_free_vram_mb": 4096}},
        })
        report = doctor_report_config(config, gpu_inventory=StaticInventory(), active_probe_runner=lambda *a, **k: self.fail("passive doctor loaded CUDA"))
        encoded = json.dumps(report)
        self.assertFalse(report["gpu_used"])
        self.assertEqual(report["checks"]["cuda"]["active_probe"]["status"], "deferred")
        self.assertEqual(report["checks"]["cuda"]["admission"]["available_for_voice_mb"], 4096)
        self.assertNotIn("secret-model", encoded)

    def test_doctor_reports_selected_turbo_manifest_without_discovery(self) -> None:
        config = voice_config_from_mapping(
            {
                "whisper_model_directory": "/private/missing-turbo-pack",
                "asr": {"model_profile": "large-v3-turbo"},
            }
        )
        report = doctor_report_config(config)
        model = report["checks"]["model"]
        self.assertEqual(model["profile"], "large-v3-turbo")
        self.assertEqual(
            model["id"],
            "dropbox-dash/faster-whisper-large-v3-turbo",
        )
        self.assertEqual(
            model["revision"],
            "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
        )
        self.assertEqual(model["manifest"]["code"], "model.directory_missing")

    def test_active_probe_requires_cuda_profile(self) -> None:
        report = doctor_report_config(voice_config_from_mapping({}), active_cuda_probe=True)
        self.assertEqual(report["checks"]["cuda"]["code"], "cuda.active_probe_requires_cuda_profile")
        self.assertFalse(report["ready"])

    def test_milestone_one_doctor_is_redacted_and_blocked_off_windows(self) -> None:
        secret = "/private/very-sensitive/model"
        config = voice_config_from_mapping({"whisper_model_directory": secret})
        report = doctor_report_config(config)
        encoded = json.dumps(report)
        self.assertNotIn(secret, encoded)
        self.assertFalse(report["network_used"])
        self.assertFalse(report["gpu_used"])
        self.assertFalse(report["ready"])
        self.assertEqual(report["checks"]["cuda"]["code"], "cuda.not_required")
        self.assertEqual(report["milestone"], "windows-packaging-tray-hardening")
        self.assertEqual(report["checks"]["packaging"]["bundle_mode"], "source")
        self.assertFalse(report["checks"]["packaging"]["frozen"])


if __name__ == "__main__":
    unittest.main()
