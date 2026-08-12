from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
import wave
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.compact_greek_tts_acquisition import (
    _download_artifact,
    _download_url_allowed,
    acquire_pinned_artifacts,
    compact_acquisition_plan,
    compact_f1_training_plan,
    extract_stoma_f1,
)
from jarvis_voice.macos.compact_greek_tts_pilot import (
    ArtifactSpec,
    STOMA_NON_F1_ROWS_TO_REJECT,
    STOMA_SECTION_COUNTS,
    STOMA_SHARDS,
    ValidationResult,
)


class _Response:
    def __init__(self, data: bytes, url: str, *, status: int = 200, headers: dict[str, str] | None = None):
        self._data = data
        self._offset = 0
        self._url = url
        self.status = status
        self.headers = {} if headers is None else headers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def geturl(self) -> str:
        return self._url

    def getcode(self) -> int:
        return self.status

    def read(self, size: int) -> bytes:
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class _Batch:
    def __init__(self, rows: list[dict[str, object]]):
        self._rows = rows

    def to_pylist(self) -> list[dict[str, object]]:
        return self._rows


def _wav_fixture() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(44_100)
        stream.writeframes(b"\0\0")
    return output.getvalue()


def _dataset_rows() -> list[dict[str, object]]:
    wav_bytes = _wav_fixture()
    rows: list[dict[str, object]] = []
    index = 0
    for section, count in STOMA_SECTION_COUNTS.items():
        for _ in range(count):
            index += 1
            session = ((index - 1) // 99) + 1
            sentence = ((index - 1) % 99) + 1
            rows.append(
                {
                    "speaker_id": "F1",
                    "section": section,
                    "text": "Ελληνικό κείμενο.",
                    "audio": {"bytes": wav_bytes, "path": None},
                    "file_name": f"Greek-F1-{section}-{session:02d}-{sentence:02d}.wav",
                }
            )
    rows.extend(
        {
            "speaker_id": "F2",
            "section": "Harvard",
            "text": "Δεν πρέπει να εξαχθεί.",
            "audio": {"bytes": wav_bytes, "path": None},
            "file_name": f"Greek-F2-Harvard-{index:02d}-{offset:02d}.wav",
        }
        for offset in range(STOMA_NON_F1_ROWS_TO_REJECT)
    )
    return rows


class _ParquetModule:
    rows = _dataset_rows()

    class ParquetFile:
        def __init__(self, path: Path):
            shard_index = 0 if path.name.startswith("train-00000") else 1
            split = 971
            self._rows = _ParquetModule.rows[:split] if shard_index == 0 else _ParquetModule.rows[split:]

        def iter_batches(self, **kwargs):
            yield _Batch(self._rows)


class CompactGreekTtsAcquisitionTests(unittest.TestCase):
    def test_download_redirect_allowlist_is_exact(self) -> None:
        self.assertTrue(_download_url_allowed("https://us.aws.cdn.hf.co/object"))
        self.assertFalse(_download_url_allowed("https://us.aws.cdn.hf.co.evil.example/object"))

    def test_plan_is_read_only_and_stoma_group_is_minimal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before = list(root.iterdir())
            report = compact_acquisition_plan(root, group="stoma")
            after = list(root.iterdir())
        self.assertEqual(report["download_bytes"], sum(item.size for item in STOMA_SHARDS))
        self.assertFalse(report["network_used"])
        self.assertEqual(before, after)

    def test_acquisition_requires_exact_byte_confirmation_before_downloader(self) -> None:
        calls: list[object] = []
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(BackendError) as raised:
                acquire_pinned_artifacts(
                    temp,
                    group="stoma",
                    confirm_download_bytes=1,
                    downloader=lambda spec, path: calls.append((spec, path)) or {},
                )
        self.assertEqual(raised.exception.code, "tts.f1_compact.download_confirmation_invalid")
        self.assertEqual(calls, [])

    def test_downloader_validates_hash_and_publishes_atomically(self) -> None:
        data = b"pinned fixture"
        source = "https://huggingface.co/example/artifact"
        spec = ArtifactSpec("artifact.bin", len(data), hashlib.sha256(data).hexdigest(), source)
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "artifact.bin"
            result = _download_artifact(spec, destination, urlopen=lambda request, timeout: _Response(data, source))
            self.assertEqual(destination.read_bytes(), data)
            self.assertFalse(destination.with_name("artifact.bin.part").exists())
        self.assertEqual(result["status"], "downloaded")

    def test_extractor_emits_only_f1_and_an_offline_manifest(self) -> None:
        valid = ValidationResult(True, "tts.f1_compact.artifacts_ready", 2, 2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "f1"
            report = extract_stoma_f1(
                root,
                output,
                parquet_module=_ParquetModule,
                artifact_validator=lambda path, specs: valid,
            )
            metadata = (output / "metadata.csv").read_text(encoding="utf-8")
            manifest = (output / "bundle-manifest.json").read_text(encoding="utf-8")
            audio_count = len(list((output / "audio").iterdir()))
        self.assertTrue(report["ready"])
        self.assertNotIn("Greek-F2", metadata)
        self.assertIn('"network_used": false', manifest)
        self.assertEqual(audio_count, 1_301)

    def test_training_plan_does_not_start_training(self) -> None:
        plan = compact_f1_training_plan("/data/f1", "/artifacts", "/work/f1")
        self.assertFalse(plan["ready"])
        self.assertFalse(plan["training_started"])
        self.assertTrue(plan["gpu_probe_required"])
        self.assertIn("--trainer.max_epochs", plan["training_argv"])
        self.assertIn("--ckpt_path", plan["training_argv"])
        self.assertIn("scripts/export_piper_mps_checkpoint.py", plan["export_argv"])


if __name__ == "__main__":
    unittest.main()
