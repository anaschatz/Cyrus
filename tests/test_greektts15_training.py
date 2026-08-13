from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.greektts15_training import (
    DATASET_SCHEMA,
    EXPECTED_ROWS,
    build_training_plan,
    normalize_transcript,
)


class GreekTts15TrainingTests(unittest.TestCase):
    def test_normalization_repairs_only_leading_greek_confusables(self) -> None:
        self.assertEqual(normalize_transcript("Tην  ίδια μέρα"), ("Την ίδια μέρα", True))
        self.assertEqual(normalize_transcript("To παιδί"), ("Το παιδί", True))
        self.assertEqual(normalize_transcript("Η CPU λειτουργεί"), ("Η CPU λειτουργεί", False))

    def test_plan_has_deterministic_stratified_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = self._bundle(Path(temp))
            with mock.patch("jarvis_voice.macos.greektts15_training.EXPECTED_ROWS", 20):
                first = build_training_plan(root)
                second = build_training_plan(root)
        self.assertEqual(first, second)
        self.assertEqual(first["split"]["counts"], {"train": 16, "validation": 2, "control": 2})
        self.assertFalse(first["training_policy"]["merge_into_quantized_base"])

    def test_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = self._bundle(Path(temp))
            target = next((root / "audio").iterdir())
            target.write_bytes(b"changed")
            with mock.patch("jarvis_voice.macos.greektts15_training.EXPECTED_ROWS", 20):
                with self.assertRaises(BackendError) as raised:
                    build_training_plan(root)
        self.assertEqual(raised.exception.code, "tts.greektts15.dataset_audio_hash_invalid")

    @staticmethod
    def _bundle(root: Path) -> Path:
        audio = root / "audio"
        audio.mkdir()
        rows = []
        hashes = {}
        for corpus in ("B2", "C1"):
            for index in range(10):
                filename = f"Greek-F1-{corpus}-01-{index:02d}.wav"
                path = audio / filename
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(44_100)
                    wav.writeframes(b"\0\0" * (44_100 * 2))
                hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
                rows.append(f"{filename}|Tο δείγμα {index}.")
        metadata = root / "metadata.csv"
        metadata.write_text("\n".join(rows) + "\n", encoding="utf-8")
        (root / "bundle-manifest.json").write_text(
            json.dumps(
                {
                    "schema": DATASET_SCHEMA,
                    "rows": len(rows),
                    "dataset_revision": "test",
                    "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                    "audio_sha256": hashes,
                }
            ),
            encoding="utf-8",
        )
        return root


if __name__ == "__main__":
    unittest.main()
