from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.macos import materialization
from jarvis_voice.macos.materialization import (
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


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "materialize_macos_cpu.py"
SPEC = importlib.util.spec_from_file_location("materialize_macos_cpu", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT_MODULE)


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
            requirements = Path(__file__).resolve().parents[1] / "requirements" / "macos-cpu.in"
            approved = dict(line.split("==") for line in requirements.read_text().splitlines())
            for name, version in approved.items():
                fake_wheel(wheelhouse, name, version)
            fake_wheel(wheelhouse, "dependency", "9.0")
            manifest = root / "wheelhouse.lock.json"
            result = lock_wheelhouse(str(wheelhouse), str(manifest))
            validated = validate_wheelhouse(str(wheelhouse), str(manifest))
            for target in (manifest, next(wheelhouse.iterdir())):
                original = target.read_bytes()
                with self.subTest(target=target.name):
                    try:
                        with self.assertRaises(BackendError):
                            lock_wheelhouse(str(wheelhouse), str(target))
                        self.assertEqual(target.read_bytes(), original)
                    finally:
                        target.write_bytes(original)
            raw = json.loads(manifest.read_text())
        self.assertTrue(result["ready"])
        self.assertEqual(validated["status"], "ready")
        self.assertEqual(len(raw["wheels"]), len(approved) + 1)
        self.assertEqual(raw["top_level"], approved)

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
    def test_publication_preserves_existing_files_and_rolls_back_failed_manifest(self) -> None:
        for collision in ("existing_model", "dangling_model", "racing_model", "racing_manifest", "manifest_failure", "model_alias", "wheel_alias", "replacement_model"):
            with self.subTest(collision=collision), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                wheel = root / SILERO_WHEEL_NAME
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr(SILERO_ONNX_MEMBER, b"o" * 100_000)
                original_wheel = wheel.read_bytes()
                model = root / "silero_vad-6.2.1.onnx"
                manifest = root / "manifest.json"
                if collision == "existing_model":
                    model.write_bytes(b"existing")
                elif collision == "dangling_model":
                    model.symlink_to(root / "missing")
                elif collision == "model_alias":
                    manifest = model
                elif collision == "wheel_alias":
                    manifest = wheel
                real_fsync = os.fsync
                real_report = materialization.write_report

                def flush(descriptor):
                    real_fsync(descriptor)
                    if collision == "racing_model" and not model.exists():
                        model.write_bytes(b"existing")
                    elif collision == "racing_manifest" and model.exists() and not manifest.exists():
                        manifest.write_bytes(b"existing")

                def report(path, value):
                    if collision == "replacement_model":
                        model.unlink()
                        model.write_bytes(b"replacement")
                    if collision in {"manifest_failure", "replacement_model"}:
                        raise BackendError("qualification.output_write_failed", "private details", subsystem="qualification")
                    return real_report(path, value)

                with (mock.patch.object(materialization, "SILERO_WHEEL_SHA256", hashlib.sha256(original_wheel).hexdigest()),
                      mock.patch.object(materialization.os, "fsync", side_effect=flush),
                      mock.patch.object(materialization, "write_report", side_effect=report)):
                    with self.assertRaises(BackendError):
                        extract_silero_onnx(str(wheel), str(root), str(manifest))
                self.assertEqual(wheel.read_bytes(), original_wheel)
                if collision in {"existing_model", "racing_model", "replacement_model"}:
                    self.assertEqual(model.read_bytes(), b"replacement" if collision == "replacement_model" else b"existing")
                elif collision == "dangling_model":
                    self.assertTrue(model.is_symlink())
                    self.assertFalse((root / "missing").exists())
                else:
                    self.assertFalse(model.exists())
                if collision == "racing_manifest":
                    self.assertEqual(manifest.read_bytes(), b"existing")
                elif manifest not in {wheel, model}:
                    self.assertFalse(manifest.exists())
                self.assertFalse(list(root.glob(".silero-*")))
                self.assertFalse(list(root.glob(".materialization-*")))

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
            self.assertEqual(artifact.stat().st_mode & 0o777, 0o600)
            self.assertEqual(manifest.stat().st_mode & 0o777, 0o600)
            self.assertFalse(list(root.glob(".silero-*")))
            self.assertFalse(list(root.glob(".materialization-*")))
        self.assertTrue(result["ready"])
        self.assertEqual(extracted, onnx)
        self.assertEqual(raw["artifact"]["sha256"], hashlib.sha256(onnx).hexdigest())
        self.assertNotIn("ignored.py", json.dumps(raw))


class PrivateConfigTests(unittest.TestCase):
    def test_malformed_json_inputs_raise_safe_backend_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base, manifest, output = (Path(temp) / name for name in ("base.json", "vad.json", "voice.json"))
            for malformed in (b"\xff", b"[" * 2000 + b"]" * 2000, b"1" * 5000):
                for invalid in (base, manifest):
                    with self.subTest(payload=malformed[:8], input=invalid.name):
                        base.write_bytes(b"{}")
                        manifest.write_bytes(b"{}")
                        invalid.write_bytes(malformed)
                        with self.assertRaisesRegex(BackendError, "materialization.config_input_unreadable"):
                            materialize_private_config(
                                str(base), "/unused", str(manifest), str(output),
                                greek_voice="unused", english_voice="unused", voice_auth_key="unused",
                            )
                        self.assertFalse(output.exists())

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
                for target in (output, base, vad, manifest, auth_key):
                    original = target.read_bytes()
                    with self.subTest(target=target.name):
                        try:
                            with self.assertRaises(BackendError):
                                materialize_private_config(
                                    str(base), "/private/model", str(manifest), str(target),
                                    greek_voice="Melina", english_voice="Samantha", voice_auth_key=str(auth_key),
                                )
                            self.assertEqual(target.read_bytes(), original)
                        finally:
                            target.write_bytes(original)
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
        for failure in (None, "manifest", "vad_init", "load", "trim", "query", "interrupt"):
            with self.subTest(failure=failure):
                fake_asr = mock.Mock()
                fake_vad = mock.Mock()
                fake_vad.trim.return_value = SimpleNamespace(speech_detected=False)
                query = mock.Mock(return_value={"max_input_channels": 1})
                error = (KeyboardInterrupt() if failure == "interrupt" else
                         BackendError("materialization.test_failure", "private details", subsystem="materialization"))
                if failure == "load":
                    fake_asr.load.side_effect = error
                if failure in {"trim", "interrupt"}:
                    fake_vad.trim.side_effect = error
                if failure == "query":
                    query.side_effect = error
                fake_config.vad.artifact.path = None if failure == "manifest" else "/vad"
                with (
                    mock.patch("jarvis_voice.macos.materialization.load_voice_config", return_value=fake_config),
                    mock.patch("jarvis_voice.macos.materialization.FasterWhisperCpuBackend", return_value=fake_asr),
                    mock.patch("jarvis_voice.macos.materialization.SileroOnnxVad", return_value=fake_vad,
                               side_effect=error if failure == "vad_init" else None),
                    mock.patch.dict("sys.modules", {"sounddevice": SimpleNamespace(query_devices=query)}),
                ):
                    if failure:
                        with self.assertRaises(KeyboardInterrupt if failure == "interrupt" else BackendError) as caught:
                            probe_local_backends("/config", "/model")
                        if failure not in {"manifest", "query"}:
                            self.assertIs(caught.exception, error)
                        elif failure == "query":
                            self.assertEqual(caught.exception.code, "materialization.audio_probe_failed")
                    else:
                        report = probe_local_backends("/config", "/model")
                        self.assertTrue(report["ready"])
                        self.assertFalse(report["microphone_opened"])
                        fake_asr.load.assert_called_once_with()
                        self.assertEqual(len(fake_vad.trim.call_args.args[0].samples), 512)
                fake_asr.close.assert_called_once_with()


class MaterializationCliTests(unittest.TestCase):
    def test_report_failures_always_print_safe_json(self) -> None:
        commands = (
            ("probe-backends", ["--config", "unused", "--model-dir", "unused"], "probe_local_backends"),
            ("disk-gate", ["--path", "unused"], "disk_space_gate"),
            ("blocked-summary", ["--python-version", "3.11.9", "--wheels-locked", "1", "--silero-bytes", "1", "--available-mib", "1"], "blocked_materialization_summary"),
            ("complete-summary", ["--python-version", "3.11.9", "--wheels-locked", "1", "--silero-bytes", "1"], "completed_materialization_summary"),
        )
        with tempfile.TemporaryDirectory() as temp:
            for command, arguments, operation in commands:
                for operation_fails, output_exists in ((False, False), (True, False), (True, True)):
                    with self.subTest(command=command, operation_fails=operation_fails, output_exists=output_exists):
                        output = Path(temp) / ("report.json" if output_exists else "missing/report.json")
                        failure = BackendError("materialization.test_failure", "private details", subsystem="materialization")
                        stdout = io.StringIO()
                        with (mock.patch.object(SCRIPT_MODULE, operation, return_value={"ready": True},
                                                side_effect=failure if operation_fails else None),
                              contextlib.redirect_stdout(stdout)):
                            status = SCRIPT_MODULE.main([command, *arguments, "--output", str(output)])
                        report = json.loads(stdout.getvalue())
                        self.assertEqual(status, 2)
                        self.assertFalse(report["ready"])
                        self.assertEqual(report["error"]["code"], "materialization.test_failure" if operation_fails else "qualification.output_directory_missing")
                        if output_exists:
                            self.assertEqual(json.loads(output.read_text()), report)
                        else:
                            self.assertEqual(report["output_error_code"], "qualification.output_directory_missing")
                            self.assertFalse(output.exists())
                        self.assertNotIn("private", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
