from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.macos.materialization import (
    DIRECT_REQUIREMENTS,
    SILERO_ONNX_MEMBER,
    SILERO_WHEEL_NAME,
    blocked_materialization_summary,
    completed_materialization_summary,
    disk_space_gate,
    extract_silero_onnx,
    lock_wheelhouse,
    materialize_private_config,
    probe_local_backends,
)
from jarvis_voice.macos.staging import validate_wheelhouse
from jarvis_voice.errors import BackendError
from jarvis_voice.manifest import LARGE_V3_TURBO_MODEL_MANIFEST, ManifestValidationResult
from jarvis_voice.voice_auth import create_voice_auth_key_file


def fake_wheel(directory: Path, name: str, version: str) -> Path:
    stem = name.replace("-", "_")
    path = directory / f"{stem}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{stem}-{version}.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n")
    return path


class WheelhouseLockTests(unittest.TestCase):
    def test_lock_reads_metadata_hashes_every_wheel_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            for name, version in DIRECT_REQUIREMENTS.items():
                fake_wheel(wheelhouse, name, version)
            fake_wheel(wheelhouse, "dependency", "9.0")
            manifest = root / "wheelhouse.lock.json"
            result = lock_wheelhouse(str(wheelhouse), str(manifest))
            validated = validate_wheelhouse(str(wheelhouse), str(manifest))
            raw = json.loads(manifest.read_text())
        self.assertTrue(result["ready"])
        self.assertEqual(validated["status"], "ready")
        self.assertEqual(len(raw["wheels"]), 7)
        self.assertEqual(raw["top_level"], dict(DIRECT_REQUIREMENTS))

    def test_missing_direct_requirement_and_wrong_platform_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            fake_wheel(wheelhouse, "dependency", "1.0")
            with self.assertRaisesRegex(BackendError, "materialization.direct_requirement_mismatch"):
                lock_wheelhouse(str(wheelhouse), str(root / "lock.json"))
            for path in wheelhouse.iterdir():
                path.unlink()
            invalid = wheelhouse / "package-1.0-cp311-cp311-win_amd64.whl"
            with zipfile.ZipFile(invalid, "w") as archive:
                archive.writestr("package-1.0.dist-info/METADATA", "Name: package\nVersion: 1.0\n\n")
            with self.assertRaisesRegex(BackendError, "materialization.wheel_platform_mismatch"):
                lock_wheelhouse(str(wheelhouse), str(root / "lock.json"))


class DiskGateTests(unittest.TestCase):
    def test_disk_gate_is_typed_and_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ready = disk_space_gate(temp, minimum_free_bytes=1)
            blocked = disk_space_gate(temp, minimum_free_bytes=2**63)
        self.assertTrue(ready["ready"])
        self.assertEqual(blocked["code"], "materialization.disk_space_insufficient")
        self.assertFalse(blocked["installation_performed"])
        self.assertNotIn(temp, json.dumps(blocked))

    def test_blocked_summary_records_network_acquisition_without_paths(self) -> None:
        report = blocked_materialization_summary(
            python_version="3.11.14",
            wheels_locked=29,
            silero_bytes=2_327_524,
            available_mib=1_011,
        )
        self.assertFalse(report["ready"])
        self.assertTrue(report["network_acquisition_performed"])
        self.assertFalse(report["runtime_dependencies_installed"])
        self.assertFalse(report["offline_install_attempted"])

    def test_completed_summary_distinguishes_acquisition_from_offline_install(self) -> None:
        report = completed_materialization_summary(
            python_version="3.11.14",
            wheels_locked=29,
            silero_bytes=2_327_524,
        )
        self.assertTrue(report["ready"])
        self.assertTrue(report["network_acquisition_performed"])
        self.assertTrue(report["offline_install"])
        self.assertTrue(report["backend_probe_ready"])
        self.assertFalse(report["microphone_opened"])


class SileroExtractionTests(unittest.TestCase):
    def test_extracts_only_bounded_onnx_and_writes_private_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheel = root / SILERO_WHEEL_NAME
            onnx = b"o" * 100_000
            with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(SILERO_ONNX_MEMBER, onnx)
                archive.writestr("silero_vad/ignored.py", "ignored")
            manifest = root / "silero.private.json"
            with mock.patch("jarvis_voice.macos.materialization.SILERO_WHEEL_SHA256", hashlib.sha256(wheel.read_bytes()).hexdigest()):
                result = extract_silero_onnx(str(wheel), str(root), str(manifest))
            raw = json.loads(manifest.read_text())
            artifact = Path(raw["artifact"]["path"])
            extracted = artifact.read_bytes()
        self.assertTrue(result["ready"])
        self.assertEqual(extracted, onnx)
        self.assertEqual(raw["artifact"]["sha256"], hashlib.sha256(onnx).hexdigest())
        self.assertNotIn("ignored.py", json.dumps(raw))


class PrivateConfigTests(unittest.TestCase):
    def test_materializes_explicit_model_vad_and_installed_voices(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base.json"
            base.write_text(json.dumps({
                "orchestrator_url": "ws://127.0.0.1:8765",
                "asr": {"execution_profile": "cpu"},
                "vad": {"artifact": {"path": None, "size": None, "sha256": None}},
                "macos": {"asr_backend": "whisper_cpp", "system_voices": {}},
            }))
            vad = root / "silero.onnx"
            vad.write_bytes(b"vad")
            manifest = root / "vad.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "artifact": {"path": str(vad), "size": 3, "sha256": hashlib.sha256(b"vad").hexdigest()},
                "source": {"package": "silero-vad", "version": "6.2.1", "wheel_sha256": "source"},
            }))
            output = root / "voice.json"
            auth_key = root / "voice-auth.json"
            create_voice_auth_key_file(auth_key, key=b"k" * 32)
            with (
                mock.patch("jarvis_voice.macos.materialization.validate_model_directory", return_value=ManifestValidationResult(True, "model.manifest_valid", "valid", files_checked=4)),
                mock.patch("jarvis_voice.macos.materialization._default_voice_probe", return_value={"Melina", "Samantha"}),
            ):
                result = materialize_private_config(
                    str(base),
                    "/private/model",
                    str(manifest),
                    str(output),
                    greek_voice="Melina",
                    english_voice="Samantha",
                    voice_auth_key=str(auth_key),
                )
            raw = json.loads(output.read_text())
        self.assertTrue(result["ready"])
        self.assertEqual(raw["macos"]["asr_backend"], "faster_whisper_cpu_smoke")
        self.assertEqual(raw["macos"]["system_voices"], {"el-GR": "Melina", "en-US": "Samantha"})
        self.assertEqual(raw["whisper_model_directory"], "/private/model")
        self.assertEqual(raw["voice_auth_key_file"], str(auth_key))

    def test_materialization_validates_selected_turbo_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base.json"
            base.write_text(json.dumps({
                "asr": {"model_profile": "large-v3-turbo"},
                "vad": {"artifact": {"path": None, "size": None, "sha256": None}},
            }))
            vad = root / "silero.onnx"
            vad.write_bytes(b"vad")
            manifest = root / "vad.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "artifact": {"path": str(vad), "size": 3, "sha256": hashlib.sha256(b"vad").hexdigest()},
                "source": {"package": "silero-vad", "version": "6.2.1", "wheel_sha256": "source"},
            }))
            auth_key = root / "voice-auth.json"
            create_voice_auth_key_file(auth_key, key=b"k" * 32)
            validator = mock.Mock(return_value=ManifestValidationResult(True, "model.manifest_valid", "valid", files_checked=5))
            with (
                mock.patch("jarvis_voice.macos.materialization.validate_model_directory", validator),
                mock.patch("jarvis_voice.macos.materialization._default_voice_probe", return_value={"Melina", "Samantha"}),
            ):
                materialize_private_config(
                    str(base),
                    "/private/turbo-model",
                    str(manifest),
                    str(root / "voice.json"),
                    greek_voice="Melina",
                    english_voice="Samantha",
                    voice_auth_key=str(auth_key),
                )

        self.assertIs(validator.call_args.args[1], LARGE_V3_TURBO_MODEL_MANIFEST)


class BackendProbeTests(unittest.TestCase):
    def test_probe_uses_speech_detected_and_never_opens_microphone(self) -> None:
        fake_config = SimpleNamespace(
            asr=SimpleNamespace(cpu_compute_type="int8", model_profile="base"),
            vad=SimpleNamespace(
                artifact=SimpleNamespace(path="/vad", size=3, sha256="hash"),
                threshold=0.5,
                minimum_speech_ms=80,
                padding_ms=80,
            ),
            audio_input_device=None,
        )
        fake_asr = mock.Mock()
        fake_vad = mock.Mock()
        fake_vad.trim.return_value = SimpleNamespace(speech_detected=False)
        sounddevice = SimpleNamespace(query_devices=lambda device, kind: {"max_input_channels": 1})
        with (
            mock.patch("jarvis_voice.macos.materialization.load_voice_config", return_value=fake_config),
            mock.patch("jarvis_voice.macos.materialization.FasterWhisperCpuBackend", return_value=fake_asr),
            mock.patch("jarvis_voice.macos.materialization.SileroOnnxVad", return_value=fake_vad),
            mock.patch.dict("sys.modules", {"sounddevice": sounddevice}),
        ):
            report = probe_local_backends("/config", "/model")
        self.assertTrue(report["ready"])
        self.assertFalse(report["microphone_opened"])
        fake_asr.load.assert_called_once_with()
        self.assertEqual(len(fake_vad.trim.call_args.args[0].samples), 512)


if __name__ == "__main__":
    unittest.main()
