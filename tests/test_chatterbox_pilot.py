from __future__ import annotations

import json
import socket
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.chatterbox_pilot import (
    MINIMUM_ACQUISITION_FREE_BYTES,
    MINIMUM_RUN_FREE_BYTES,
    F1_MAX_MEAN_REALTIME_FACTOR,
    F1_MAX_STANDALONE_RSS_BYTES,
    F1_QUALIFICATION_FIXTURES,
    F1_REFERENCE_SHA256,
    MODEL_ARTIFACTS,
    MODEL_DOWNLOAD_BYTES,
    TOKENIZER_FILENAME,
    _add_f1_qualification_gates,
    _deny_network_during_pilot,
    _release_mps_unit_cache,
    _validated_f1_reference_audio,
    _validated_audio_path,
    chatterbox_pilot_preflight,
    score_f1_transcripts,
    validate_chatterbox_model_directory,
)


class ChatterboxPilotTests(unittest.TestCase):
    def test_network_guard_fails_closed_and_restores_socket(self) -> None:
        original_socket = socket.socket

        @_deny_network_during_pilot
        def guarded() -> dict[str, object]:
            self.assertIsNot(socket.socket, original_socket)
            with socket.socket() as stream:
                with self.assertRaises(BackendError) as raised:
                    stream.connect(("127.0.0.1", 9))
            self.assertEqual(raised.exception.code, "tts.chatterbox.network_forbidden")
            return {"ready": True}

        self.assertTrue(guarded()["ready"])
        self.assertIs(socket.socket, original_socket)

    def test_low_disk_blocks_acquisition_without_network_or_install(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = chatterbox_pilot_preflight(
                temp,
                Path(temp) / "missing-model",
                platform_name="darwin",
                machine="arm64",
                python_version=(3, 11, 14),
                pointer_bits=64,
                disk_usage=lambda path: SimpleNamespace(free=4 * 1024**3),
                module_probe=lambda name: False,
                source_revision_probe=lambda: False,
            )
        self.assertFalse(report["ready"])
        self.assertFalse(report["acquisition_ready"])
        self.assertEqual(report["code"], "tts.chatterbox.disk_space_insufficient")
        self.assertFalse(report["network_used"])
        self.assertFalse(report["installation_performed"])
        self.assertEqual(report["disk"]["acquisition_required_mib"], MINIMUM_ACQUISITION_FREE_BYTES // 1024**2)

    def test_host_gate_requires_darwin_arm64_python_311(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = chatterbox_pilot_preflight(
                temp,
                Path(temp) / "model",
                platform_name="darwin",
                machine="arm64",
                python_version=(3, 14, 0),
                pointer_bits=64,
                disk_usage=lambda path: SimpleNamespace(free=16 * 1024**3),
                module_probe=lambda name: True,
                source_revision_probe=lambda: True,
            )
        self.assertEqual(report["code"], "tts.chatterbox.host_unsupported")
        self.assertFalse(report["acquisition_ready"])

    def test_installed_dependencies_reduce_acquisition_gate_to_model_plus_reserve(self) -> None:
        required = MODEL_DOWNLOAD_BYTES + MINIMUM_RUN_FREE_BYTES
        with tempfile.TemporaryDirectory() as temp:
            report = chatterbox_pilot_preflight(
                temp,
                Path(temp) / "model",
                platform_name="darwin",
                machine="arm64",
                python_version=(3, 11, 14),
                pointer_bits=64,
                disk_usage=lambda path: SimpleNamespace(free=required),
                module_probe=lambda name: True,
                source_revision_probe=lambda: True,
            )
        self.assertTrue(report["acquisition_ready"])
        self.assertEqual(report["disk"]["acquisition_required_mib"], required // 1024**2)

    def test_model_validation_is_strict_and_tokenizer_requires_greek(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for artifact in MODEL_ARTIFACTS:
                path = root / artifact.filename
                if artifact.filename != TOKENIZER_FILENAME:
                    path.write_bytes(b"fixture")
            tokenizer = json.dumps({"added_tokens": [{"content": "[el]"}]})
            (root / TOKENIZER_FILENAME).write_text(tokenizer.ljust(69_989))
            with mock.patch("jarvis_voice.macos.chatterbox_pilot._sha256", return_value="wrong"):
                result = validate_chatterbox_model_directory(root)
        self.assertFalse(result.ready)
        self.assertEqual(result.code, "tts.chatterbox.model_artifact_invalid")

    def test_model_validation_accepts_pinned_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hashes: dict[str, str] = {}
            for artifact in MODEL_ARTIFACTS:
                path = root / artifact.filename
                if artifact.filename == TOKENIZER_FILENAME:
                    tokenizer = json.dumps({"added_tokens": [{"content": "[el]"}]})
                    path.write_text(tokenizer.ljust(artifact.size))
                else:
                    with path.open("wb") as stream:
                        stream.truncate(artifact.size)
                hashes[str(path)] = artifact.sha256

            def fake_hash(path: Path) -> str:
                return hashes[str(path)]

            with mock.patch("jarvis_voice.macos.chatterbox_pilot._sha256", side_effect=fake_hash):
                result = validate_chatterbox_model_directory(root)
        self.assertTrue(result.ready)
        self.assertEqual(result.files_present, len(MODEL_ARTIFACTS))

    def test_acquisition_report_uses_post_download_model_validation(self) -> None:
        from jarvis_voice.macos.chatterbox_pilot import acquire_chatterbox_model

        before = SimpleNamespace(ready=False, code="tts.chatterbox.model_directory_missing")
        after = SimpleNamespace(
            ready=True,
            code="tts.chatterbox.model_ready",
            files_present=len(MODEL_ARTIFACTS),
            files_required=len(MODEL_ARTIFACTS),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            with (
                mock.patch(
                    "jarvis_voice.macos.chatterbox_pilot.chatterbox_pilot_preflight",
                    return_value={"acquisition_ready": True, "code": "ready", "model": {}},
                ),
                mock.patch(
                    "jarvis_voice.macos.chatterbox_pilot.validate_chatterbox_model_directory",
                    side_effect=[before, after],
                ),
                mock.patch.dict(
                    "sys.modules",
                    {"huggingface_hub": SimpleNamespace(snapshot_download=mock.Mock())},
                ),
                mock.patch("os.chmod"),
            ):
                report = acquire_chatterbox_model(root, model)
        self.assertTrue(report["model"]["ready"])
        self.assertEqual(report["model"]["code"], "tts.chatterbox.model_ready")
        self.assertEqual(report["model"]["files_present"], len(MODEL_ARTIFACTS))

    def test_voice_conversion_audio_validation_rejects_unsafe_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            valid = root / "reference.wav"
            with wave.open(str(valid), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(24_000)
                stream.writeframes(b"\0\0")
            self.assertEqual(_validated_audio_path(valid, role="target"), valid)
            invalid = root / "reference.txt"
            invalid.write_text("not audio")
            with self.assertRaises(BackendError) as raised:
                _validated_audio_path(invalid, role="target")
        self.assertEqual(raised.exception.code, "tts.chatterbox.target_audio_invalid")

    def test_voice_conversion_audio_validation_rejects_empty_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            empty = Path(temp) / "empty.wav"
            with wave.open(str(empty), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(24_000)
            with self.assertRaises(BackendError) as raised:
                _validated_audio_path(empty, role="target")
        self.assertEqual(raised.exception.code, "tts.chatterbox.target_audio_invalid")

    def test_f1_qualification_fixture_set_is_fixed_and_complete(self) -> None:
        fixture_ids = [fixture_id for fixture_id, _ in F1_QUALIFICATION_FIXTURES]
        self.assertEqual(len(F1_QUALIFICATION_FIXTURES), 20)
        self.assertEqual(len(set(fixture_ids)), 20)
        self.assertTrue(all(text.strip() for _, text in F1_QUALIFICATION_FIXTURES))
        self.assertEqual(F1_MAX_STANDALONE_RSS_BYTES, 6 * 1024**3)
        self.assertEqual(F1_MAX_MEAN_REALTIME_FACTOR, 1.0)

    def test_f1_reference_requires_the_pinned_stoma_recording(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            reference = Path(temp) / "reference.wav"
            with wave.open(str(reference), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(44_100)
                stream.writeframes(b"\0\0")
            with mock.patch("jarvis_voice.macos.chatterbox_pilot._sha256", return_value=F1_REFERENCE_SHA256):
                self.assertEqual(_validated_f1_reference_audio(reference), reference)
            with mock.patch("jarvis_voice.macos.chatterbox_pilot._sha256", return_value="wrong"):
                with self.assertRaises(BackendError) as raised:
                    _validated_f1_reference_audio(reference)
        self.assertEqual(raised.exception.code, "tts.chatterbox.f1_reference_mismatch")

    def test_f1_qualification_stays_blocked_until_streaming_and_cancel_exist(self) -> None:
        report = _add_f1_qualification_gates(
            {
                "voice": {"reference_sha256": F1_REFERENCE_SHA256},
                "peak_rss_bytes": F1_MAX_STANDALONE_RSS_BYTES,
                "samples": [
                    {"realtime_factor": 0.5}
                    for _ in F1_QUALIFICATION_FIXTURES
                ],
            }
        )
        qualification = report["qualification"]
        self.assertFalse(qualification["automated_gates_passed"])
        self.assertFalse(qualification["production_promotion_allowed"])
        self.assertFalse(qualification["automated_gates"]["incremental_audio_streaming"])
        self.assertFalse(qualification["automated_gates"]["interrupt_during_generation"])
        self.assertFalse(report["production_backend_changed"])

    def test_f1_transcript_scoring_reports_micro_word_error_rate(self) -> None:
        transcripts = {fixture_id: text for fixture_id, text in F1_QUALIFICATION_FIXTURES}
        transcripts["greeting"] = "Καλημέρα. Είμαι έτοιμη να βοηθήσω."
        report = score_f1_transcripts(transcripts)
        self.assertEqual(report["fixture_count"], 20)
        self.assertEqual(report["total_word_errors"], 1)
        self.assertGreater(report["micro_wer_percent"], 0.0)
        self.assertFalse(report["production_promotion_allowed"])

    def test_f1_transcript_scoring_rejects_incomplete_sets(self) -> None:
        with self.assertRaises(BackendError) as raised:
            score_f1_transcripts({"greeting": "Καλημέρα"})
        self.assertEqual(raised.exception.code, "tts.chatterbox.f1_transcripts_incomplete")

    def test_mps_unit_cache_release_collects_empties_and_synchronizes(self) -> None:
        fake_mps = SimpleNamespace(empty_cache=mock.Mock(), synchronize=mock.Mock())
        with mock.patch("gc.collect") as collect:
            _release_mps_unit_cache(SimpleNamespace(mps=fake_mps))
        collect.assert_called_once_with()
        fake_mps.empty_cache.assert_called_once_with()
        fake_mps.synchronize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
