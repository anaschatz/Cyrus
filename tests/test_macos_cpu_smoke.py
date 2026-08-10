from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.config import voice_config_from_mapping
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.asr_selection import create_macos_asr_backend, with_faster_whisper_cpu_smoke
from jarvis_voice.macos.doctor import macos_doctor_report
from jarvis_voice.manifest import ManifestValidationResult


class MacCpuSmokeSelectionTests(unittest.TestCase):
    def test_profile_is_explicit_and_absolute(self) -> None:
        base = voice_config_from_mapping({})
        self.assertEqual(base.macos.asr_backend, "whisper_cpp")
        selected = with_faster_whisper_cpu_smoke(base, "/private/local/model")
        self.assertEqual(selected.macos.asr_backend, "faster_whisper_cpu_smoke")
        self.assertEqual(selected.asr.execution_profile, "cpu")
        self.assertEqual(selected.whisper_model_directory, "/private/local/model")
        self.assertEqual(base.macos.asr_backend, "whisper_cpp")
        with self.assertRaisesRegex(BackendError, "asr.absolute_model_directory_required"):
            with_faster_whisper_cpu_smoke(base, "relative/model")

    def test_strict_config_rejects_unknown_backend(self) -> None:
        with self.assertRaisesRegex(BackendError, "config.invalid_field"):
            voice_config_from_mapping({"macos": {"asr_backend": "automatic"}})

    def test_factory_constructs_only_the_selected_backend(self) -> None:
        calls: list[tuple[str, object]] = []

        def cpu_factory(model_directory: str, *, compute_type: str):
            calls.append(("cpu", (model_directory, compute_type)))
            return object()

        def cpp_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(("cpp", (args, kwargs)))
            return object()

        selected = with_faster_whisper_cpu_smoke(voice_config_from_mapping({}), "/private/local/model")
        backend = create_macos_asr_backend(selected, cpu_factory=cpu_factory, whisper_cpp_factory=cpp_factory)  # type: ignore[arg-type]
        self.assertIsNotNone(backend)
        self.assertEqual(calls, [("cpu", ("/private/local/model", "int8"))])

        calls.clear()
        create_macos_asr_backend(voice_config_from_mapping({}), cpu_factory=cpu_factory, whisper_cpp_factory=cpp_factory)  # type: ignore[arg-type]
        self.assertEqual(calls[0][0], "cpp")


class MacCpuSmokeDoctorTests(unittest.TestCase):
    def _config(self, vad_path: Path):
        raw = vad_path.read_bytes()
        import hashlib

        base = voice_config_from_mapping({
            "orchestrator_url": "ws://127.0.0.1:8765",
            "vad": {"artifact": {"path": str(vad_path), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}},
            "macos": {"system_voices": {"el-GR": "GreekVoice", "en-US": "EnglishVoice"}},
        })
        return with_faster_whisper_cpu_smoke(base, "/private/secret-local-model")

    def test_ready_report_is_redacted_and_never_checks_whisper_cpp(self) -> None:
        dependencies = {
            name: {"status": "ready", "code": "dependency.available", "package": name, "version": "test"}
            for name in ("sounddevice", "soxr", "onnxruntime", "websockets", "faster-whisper", "ctranslate2", "numpy")
        }
        with tempfile.TemporaryDirectory() as temp:
            vad = Path(temp) / "silero.onnx"
            vad.write_bytes(b"vad")
            config = self._config(vad)
            with mock.patch("jarvis_voice.macos.doctor._validate_artifact", side_effect=AssertionError("unused backend inspected")):
                report = macos_doctor_report(
                    config,
                    platform_name="darwin",
                    machine="arm64",
                    macos_version="14.6",
                    python_version=(3, 11),
                    pointer_bits=64,
                    permission_probe=lambda: True,
                    audio_probe=lambda device: True,
                    voice_probe=lambda: {"GreekVoice", "EnglishVoice"},
                    dependencies=dependencies,
                    model_validator=lambda path: ManifestValidationResult(True, "model.manifest_valid", "valid", files_checked=4),
                )
        encoded = json.dumps(report)
        self.assertTrue(report["ready"])
        self.assertEqual(report["milestone"], "macos-cpu-smoke")
        self.assertEqual(report["checks"]["asr"]["backend"], "faster_whisper_cpu_smoke")
        self.assertFalse(report["checks"]["asr"]["model_loaded"])
        self.assertFalse(report["checks"]["asr"]["network_fallback"])
        self.assertEqual(report["checks"]["whisper_cpp"]["status"], "deferred")
        self.assertNotIn("secret-local-model", encoded)
        self.assertNotIn(str(vad), encoded)
        self.assertNotIn("ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66", encoded)
        self.assertFalse(report["network_used"])
        self.assertFalse(report["content_captured"])

    def test_invalid_model_and_missing_dependencies_are_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            vad = Path(temp) / "silero.onnx"
            vad.write_bytes(b"vad")
            config = self._config(vad)
            report = macos_doctor_report(
                config,
                platform_name="darwin",
                machine="arm64",
                macos_version="14.0",
                python_version=(3, 11),
                pointer_bits=64,
                permission_probe=lambda: True,
                audio_probe=lambda device: True,
                voice_probe=lambda: {"GreekVoice", "EnglishVoice"},
                dependencies={"sounddevice": {"status": "blocked", "code": "dependency.missing"}},
                model_validator=lambda path: ManifestValidationResult(False, "model.hash_mismatch", "invalid", artifact_name="model.bin"),
            )
        self.assertFalse(report["ready"])
        self.assertEqual(report["checks"]["asr"]["code"], "model.hash_mismatch")
        self.assertEqual(report["checks"]["audio"]["code"], "audio.sounddevice_missing")

    def test_cli_exposes_explicit_cpu_commands(self) -> None:
        from jarvis_voice.cli import build_parser

        doctor = build_parser().parse_args(["doctor-macos-cpu", "--config", "voice.json", "--model-dir", "/model"])
        run = build_parser().parse_args(["run-macos-cpu", "--config", "voice.json", "--model-dir", "/model"])
        self.assertEqual((doctor.command, doctor.model_dir), ("doctor-macos-cpu", "/model"))
        self.assertEqual((run.command, run.model_dir), ("run-macos-cpu", "/model"))


if __name__ == "__main__":
    unittest.main()
