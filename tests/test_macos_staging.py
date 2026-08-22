from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import _bootstrap  # noqa: F401

from jarvis_voice.config import voice_config_from_mapping
from jarvis_voice.macos.asr_selection import with_faster_whisper_cpu_smoke
from jarvis_voice.macos.staging import macos_staging_report, probe_python, validate_wheelhouse
from jarvis_voice.manifest import ManifestValidationResult


def ready_python_runner(*args, **kwargs):  # type: ignore[no-untyped-def]
    del args, kwargs
    return SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"platform": "darwin", "machine": "arm64", "macos": "14.6", "version": [3, 11, 9], "bits": 64}),
    )


def write_wheelhouse(root: Path) -> tuple[Path, Path]:
    wheelhouse = root / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "example-1.0-py3-none-any.whl"
    wheel.write_bytes(b"sealed-wheel")
    manifest = root / "wheelhouse.lock.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "target": {"python": "3.11", "platform": "macos_arm64"},
        "top_level": {
            "faster-whisper": "1.2.1",
            "sounddevice": "0.5.3",
            "soxr": "1.1.0",
            "onnxruntime": "1.27.0",
            "kokoro-onnx": "0.6.1",
            "websockets": "15.0.1",
        },
        "wheels": [{
            "filename": wheel.name,
            "size": wheel.stat().st_size,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        }],
    }), encoding="utf-8")
    return wheelhouse, manifest


class PythonProbeTests(unittest.TestCase):
    def test_isolated_python_311_arm64_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            executable = Path(temp) / "python3.11"
            executable.write_bytes(b"fake")
            executable.chmod(0o700)
            result = probe_python(str(executable), runner=ready_python_runner)
        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["isolated_probe"])

    def test_relative_missing_and_wrong_version_are_blocked(self) -> None:
        self.assertEqual(probe_python("python3.11")["code"], "staging.python_absolute_path_required")
        self.assertEqual(probe_python("/definitely/missing/python3.11")["code"], "staging.python_executable_missing")
        with tempfile.TemporaryDirectory() as temp:
            executable = Path(temp) / "python"
            executable.write_bytes(b"fake")
            executable.chmod(0o700)

            def wrong(*args, **kwargs):  # type: ignore[no-untyped-def]
                del args, kwargs
                return SimpleNamespace(returncode=0, stdout=json.dumps({"platform": "darwin", "machine": "arm64", "macos": "14.6", "version": [3, 14, 0], "bits": 64}))

            result = probe_python(str(executable), runner=wrong)
        self.assertEqual(result["code"], "staging.python_311_arm64_required")


class WheelhouseTests(unittest.TestCase):
    def test_hash_locked_wheelhouse_and_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            wheelhouse, manifest = write_wheelhouse(Path(temp))
            result = validate_wheelhouse(str(wheelhouse), str(manifest))
            self.assertEqual(result["status"], "ready")
            self.assertTrue(result["network_install_forbidden"])
            next(wheelhouse.iterdir()).write_bytes(b"changed")
            changed = validate_wheelhouse(str(wheelhouse), str(manifest))
        self.assertEqual(changed["code"], "staging.wheelhouse_artifact_mismatch")

    def test_wheel_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheelhouse, manifest = write_wheelhouse(root)
            wheel = next(wheelhouse.iterdir())
            outside = root / "outside.whl"
            outside.write_bytes(wheel.read_bytes())
            wheel.unlink()
            wheel.symlink_to(outside)
            result = validate_wheelhouse(str(wheelhouse), str(manifest))
        self.assertEqual(result["code"], "staging.wheelhouse_artifact_mismatch")

    def test_missing_partial_and_relative_inputs_are_blocked(self) -> None:
        self.assertEqual(validate_wheelhouse(None, None)["code"], "staging.wheelhouse_not_supplied")
        self.assertEqual(validate_wheelhouse("/wheelhouse", None)["code"], "staging.wheelhouse_and_manifest_required")
        self.assertEqual(validate_wheelhouse("relative", "manifest.json")["code"], "staging.wheelhouse_absolute_paths_required")


class StagingReportTests(unittest.TestCase):
    def test_complete_staging_is_redacted_and_still_requires_install_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "python3.11"
            executable.write_bytes(b"fake")
            executable.chmod(0o700)
            vad = root / "silero.onnx"
            vad.write_bytes(b"vad")
            wheelhouse, manifest = write_wheelhouse(root)
            config = with_faster_whisper_cpu_smoke(voice_config_from_mapping({
                "vad": {"artifact": {"path": str(vad), "size": 3, "sha256": hashlib.sha256(b"vad").hexdigest()}},
                "macos": {"system_voices": {"el-GR": "GreekVoice", "en-US": "EnglishVoice"}},
            }), "/private/secret-model")
            report = macos_staging_report(
                config,
                python_executable=str(executable),
                wheelhouse=str(wheelhouse),
                wheelhouse_manifest=str(manifest),
                python_runner=ready_python_runner,
                model_validator=lambda path: ManifestValidationResult(True, "model.manifest_valid", "valid", files_checked=4),
                permission_probe=lambda: True,
                voice_probe=lambda: {"GreekVoice", "EnglishVoice"},
            )
            encoded = json.dumps(report)
        self.assertTrue(report["ready"])
        self.assertTrue(report["approval_required"])
        self.assertEqual(report["next_action"], "staging.offline_install_requires_explicit_approval")
        self.assertFalse(report["installation_performed"])
        self.assertFalse(report["network_used"])
        self.assertEqual(report["checks"]["model"]["model_profile"], "base")
        self.assertEqual(
            report["checks"]["model"]["model_id"],
            "Systran/faster-whisper-base",
        )
        self.assertNotIn(str(root), encoded)
        self.assertNotIn("secret-model", encoded)
        self.assertNotRegex(encoded, r"[0-9a-f]{64}")

    def test_current_blocker_priority_starts_with_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            executable = Path(temp) / "python"
            executable.write_bytes(b"fake")
            executable.chmod(0o700)
            config = with_faster_whisper_cpu_smoke(voice_config_from_mapping({}), "/private/model")

            def current(*args, **kwargs):  # type: ignore[no-untyped-def]
                del args, kwargs
                return SimpleNamespace(returncode=0, stdout=json.dumps({"platform": "darwin", "machine": "arm64", "macos": "26.3", "version": [3, 14, 6], "bits": 64}))

            report = macos_staging_report(
                config,
                python_executable=str(executable),
                python_runner=current,
                model_validator=lambda path: ManifestValidationResult(True, "model.manifest_valid", "valid", files_checked=4),
                permission_probe=lambda: False,
                voice_probe=set,
            )
        self.assertFalse(report["ready"])
        self.assertEqual(report["next_action"], "staging.approval_python_311_required")
        self.assertFalse(report["permission_prompted"])
        self.assertFalse(report["runtime_started"])


if __name__ == "__main__":
    unittest.main()
