from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from jarvis_inference.strict_json import canonical_bytes
from jarvis_inference.windows_asr_corpus import (
    WindowsAsrCorpusError,
    _is_local_windows_absolute,
    _windows_drive_is_local,
    build_corpus_manifest,
    coexistence_entries,
    validate_corpus_manifest,
    write_corpus_manifest,
)
from jarvis_inference.windows_fixture import (
    FixtureFailure,
    _collector_environment,
    _write_json,
    load_windows_assets,
)

ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "fixtures" / "windows-asr-audio-corpus-v1.template.json"
FIXTURE = ROOT / "fixtures" / "windows-asr-coexistence-v1.json"


def _write_wav(path: Path, *, frames: int = 24_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(os.fspath(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\0\0" * frames)


def _corpus(root: Path) -> Path:
    for language in ("el", "en"):
        for ordinal in range(1, 101):
            case_id = f"asr-{language}-{ordinal:03d}"
            _write_wav(root / "audio" / language / f"{case_id}.wav")
            reference = root / "references" / language / f"{case_id}.txt"
            reference.parent.mkdir(parents=True, exist_ok=True)
            text = f"δοκιμή αναφοράς {ordinal}" if language == "el" else f"reference test {ordinal}"
            reference.write_text(text, encoding="utf-8")
    document = build_corpus_manifest(root, TEMPLATE)
    output = root / "audio-corpus-v1.json"
    write_corpus_manifest(output, document)
    return output


class WindowsAsrCorpusTests(unittest.TestCase):
    def test_windows_paths_are_local_drive_only(self) -> None:
        self.assertTrue(_is_local_windows_absolute(r"D:\CYRUS\qualification"))
        self.assertFalse(_is_local_windows_absolute(r"\\server\share\qualification"))
        self.assertFalse(_is_local_windows_absolute(r"\\?\D:\qualification"))
        self.assertFalse(
            _windows_drive_is_local(
                r"Z:\qualification",
                drive_type_reader=lambda _root: 4,
            )
        )

    def test_builder_creates_exact_content_free_200_pair_inventory(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            root = Path(directory)
            output = _corpus(root)
            document = validate_corpus_manifest(output, corpus_root=root)
            self.assertEqual(len(document["entries"]), 200)
            self.assertEqual(document["entries"][0]["case_id"], "asr-el-001")
            self.assertEqual(document["entries"][-1]["case_id"], "asr-en-100")
            selected = coexistence_entries(document)
            self.assertEqual(len(selected), 100)
            self.assertEqual(selected[49]["case_id"], "asr-el-050")
            self.assertEqual(selected[50]["case_id"], "asr-en-001")
            self.assertEqual(document["schedule"]["inference_lead_in_ms"], 2000)
            self.assertTrue(all(entry["duration_ms"] == 1500 for entry in document["entries"]))
            self.assertTrue(all(len(entry["sha256"]) == 64 for entry in document["entries"]))
            self.assertTrue(all(entry["reference_word_count"] == 3 for entry in document["entries"]))
            self.assertNotIn("transcript", output.read_text(encoding="utf-8").casefold())
            self.assertNotIn("δοκιμή", output.read_text(encoding="utf-8").casefold())
            original = output.read_bytes()
            with self.assertRaises(WindowsAsrCorpusError):
                write_corpus_manifest(output, document)
            self.assertEqual(output.read_bytes(), original)

            raced = root / "raced-corpus.json"

            def publish_racer(_temporary: Path, destination: Path) -> None:
                Path(destination).write_bytes(b"racer-won")
                raise FileExistsError

            with (
                patch(
                    "jarvis_inference.windows_asr_corpus.os.link",
                    side_effect=publish_racer,
                ),
                self.assertRaises(FileExistsError),
            ):
                write_corpus_manifest(raced, document)
            self.assertEqual(raced.read_bytes(), b"racer-won")

    def test_identity_change_or_unsafe_audio_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            root = Path(directory)
            output = _corpus(root)
            target = root / "audio" / "el" / "asr-el-001.wav"
            target.write_bytes(target.read_bytes() + b"x")
            with self.assertRaisesRegex(WindowsAsrCorpusError, "identity changed"):
                validate_corpus_manifest(output, corpus_root=root)

    def test_reference_identity_controls_and_intermediate_links_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            root = Path(directory)
            output = _corpus(root)
            reference = root / "references" / "el" / "asr-el-001.txt"
            reference.write_text("unsafe\ncontrol", encoding="utf-8")
            with self.assertRaises(WindowsAsrCorpusError):
                validate_corpus_manifest(output, corpus_root=root)
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            audio = root / "audio"
            try:
                audio.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                return
            with self.assertRaises(WindowsAsrCorpusError):
                build_corpus_manifest(root, TEMPLATE)

    def test_windows_runner_asset_loader_requires_private_complete_corpus(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            root = Path(directory)
            corpus = _corpus(root)
            fixture = root / "windows-asr-coexistence-v1.json"
            fixture.write_bytes(FIXTURE.read_bytes())
            collector = root / "collector.exe"
            collector.write_bytes(b"collector")
            power = root / "power-profile-v1.json"
            power.write_text(
                '{"windows_power_scheme_guid":"12345678-1234-1234-1234-123456789abc","ac_state":"ac","gpu_power_limit_watts":360,"gpu_clock_policy":"applications_graphics_mhz=2500;applications_memory_mhz=14000","driver_version":"test","compute_runtime":"CUDA 12.8","require_zero_thermal_power_throttle_reasons":true}',
                encoding="utf-8",
            )
            asr = root / "asr-config-v1.json"
            model_directory = root / "model"
            model_directory.mkdir()
            asr_document = {
                "schema_version": 1,
                "backend": "faster-whisper",
                "model_pack": {
                    "id": "dropbox-dash/faster-whisper-large-v3-turbo",
                    "revision": "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
                    "artifacts_sha256": "c4cd79451ca5f1fb41221ee186d1579b1d7f2196f9d573dde5eecf2a6e5dbafe",
                },
                "model_directory": str(model_directory),
                "model_profile": "large-v3-turbo",
                "execution_profile": "cuda_turn_scoped",
                "cpu_compute_type": "int8",
                "cuda": {
                    "device_index": 0,
                    "compute_type": "int8_float16",
                    "minimum_free_vram_mb": 2048,
                    "reserve_for_other_workloads_mb": 2048,
                    "fallback_to_cpu": True,
                    "worker_start_timeout_seconds": 120,
                    "transcription_timeout_seconds": 180,
                    "warm_restart_limit": 0,
                    "vram_poll_interval_ms": 100,
                },
                "sample_rate_hz": 16000,
                "channels": 1,
                "sample_width_bytes": 2,
                "language_mode": "multilingual-auto",
                "maximum_utterance_seconds": 30,
            }
            asr.write_bytes(canonical_bytes(asr_document))
            assets = {
                "FIXTURE": fixture,
                "COLLECTOR": collector,
                "AUDIO_CORPUS": corpus,
                "POWER_PROFILE": power,
                "ASR_CONFIG": asr,
            }
            environment: dict[str, str] = {}
            for name, path in assets.items():
                environment["JARVIS_QUAL_WINDOWS_" + name] = str(path)
                environment["JARVIS_QUAL_WINDOWS_" + name + "_SHA256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = load_windows_assets(environment)
            self.assertEqual(loaded.corpus_digest, hashlib.sha256(corpus.read_bytes()).hexdigest())
            self.assertEqual(
                loaded.asr_config_digest,
                hashlib.sha256(canonical_bytes(asr_document)).hexdigest(),
            )
            environment["JARVIS_QUAL_WINDOWS_AUDIO_CORPUS_SHA256"] = "0" * 64
            with self.assertRaises(FixtureFailure):
                load_windows_assets(environment)

            environment["JARVIS_QUAL_WINDOWS_AUDIO_CORPUS_SHA256"] = hashlib.sha256(
                corpus.read_bytes()
            ).hexdigest()
            system_root = root / "Windows"
            system32 = system_root / "System32"
            system32.mkdir(parents=True)
            nvidia_smi = system32 / "nvidia-smi.exe"
            powercfg = system32 / "powercfg.exe"
            nvidia_smi.write_bytes(b"trusted-nvidia-smi")
            powercfg.write_bytes(b"trusted-powercfg")
            environment.update(
                {
                    "SYSTEMROOT": str(system_root),
                    "PATH": str(root / "attacker-controlled"),
                    "TEMP": str(root / "ambient-temp"),
                    "TMP": str(root / "ambient-tmp"),
                    "TMPDIR": str(root / "ambient-tmpdir"),
                    "JARVIS_QUAL_BASE_URL": "http://127.0.0.1:8766/v1",
                    "JARVIS_QUAL_API_KEY": "k" * 32,
                    "JARVIS_QUAL_FIXTURE_DIGEST": "f" * 64,
                }
            )
            child = _collector_environment(
                root / "collector-output.json",
                loaded,
                environment,
                private_temp=root,
                windows_directory_reader=lambda: system_root,
            )
            self.assertNotIn("PATH", child)
            self.assertEqual(child["TEMP"], str(root))
            self.assertEqual(child["TMP"], str(root))
            self.assertEqual(child["TMPDIR"], str(root))
            self.assertEqual(child["SYSTEMROOT"], str(system_root))
            self.assertEqual(child["WINDIR"], str(system_root))
            self.assertEqual(
                child["CYRUS_ASR_QUALIFICATION_NVIDIA_SMI"],
                str(nvidia_smi),
            )
            self.assertEqual(
                child["CYRUS_ASR_QUALIFICATION_POWERCFG"],
                str(powercfg),
            )

            nvidia_smi.unlink()
            program_files = root / "Program Files"
            standard_nv_smi = (
                program_files
                / "NVIDIA Corporation"
                / "NVSMI"
                / "nvidia-smi.exe"
            )
            standard_nv_smi.parent.mkdir(parents=True)
            standard_nv_smi.write_bytes(b"trusted-standard-nvidia-smi")
            standard_child = _collector_environment(
                root / "collector-output-standard.json",
                loaded,
                environment,
                private_temp=root,
                windows_directory_reader=lambda: system_root,
                program_files_reader=lambda: program_files,
            )
            self.assertEqual(
                standard_child["CYRUS_ASR_QUALIFICATION_NVIDIA_SMI"],
                str(standard_nv_smi),
            )
            self.assertNotIn("PATH", standard_child)

            linked_root = root / "linked-Windows"
            try:
                linked_root.symlink_to(system_root, target_is_directory=True)
            except (OSError, NotImplementedError):
                pass
            else:
                with self.assertRaises(FixtureFailure):
                    _collector_environment(
                        root / "unsafe-collector-output.json",
                        loaded,
                        environment,
                        private_temp=root,
                        windows_directory_reader=lambda: linked_root,
                    )

            linked_program_files = root / "linked-Program-Files"
            try:
                linked_program_files.symlink_to(
                    program_files,
                    target_is_directory=True,
                )
            except (OSError, NotImplementedError):
                pass
            else:
                with self.assertRaises(FixtureFailure):
                    _collector_environment(
                        root / "unsafe-program-files-output.json",
                        loaded,
                        environment,
                        private_temp=root,
                        windows_directory_reader=lambda: system_root,
                        program_files_reader=lambda: linked_program_files,
                    )

    def test_windows_evidence_publication_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            root = Path(directory)
            output = root / "evidence.json"
            _write_json(output, {"value": 1})
            original = output.read_bytes()
            with self.assertRaises(FixtureFailure):
                _write_json(output, {"value": 2})
            self.assertEqual(output.read_bytes(), original)

            raced = root / "raced-evidence.json"

            def publish_racer(_temporary: Path, destination: Path) -> None:
                Path(destination).write_bytes(b"racer-won")
                raise FileExistsError

            with (
                patch(
                    "jarvis_inference.windows_fixture.os.link",
                    side_effect=publish_racer,
                ),
                self.assertRaises(FixtureFailure),
            ):
                _write_json(raced, {"value": 1})
            self.assertEqual(raced.read_bytes(), b"racer-won")

    def test_runner_source_has_no_native_pass_or_sensitive_command_line(self) -> None:
        source = (ROOT / "src" / "jarvis_inference" / "windows_fixture.py").read_text(encoding="utf-8")
        self.assertIn("CYRUS_ASR_QUALIFICATION_OUTPUT", source)
        self.assertIn("stdin=subprocess.DEVNULL", source)
        self.assertIn("stdout=subprocess.DEVNULL", source)
        self.assertIn("stderr=subprocess.DEVNULL", source)
        self.assertNotIn("status\": \"pass", source)
        self.assertNotIn("audio bytes", source.casefold())


if __name__ == "__main__":
    unittest.main()
