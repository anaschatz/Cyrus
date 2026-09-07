from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.compact_greek_tts_acquisition import extract_stoma_f1
from jarvis_voice.macos.compact_greek_tts_pilot import ValidationResult, validate_f1_rows
from test_compact_greek_tts_acquisition import _ParquetModule, _dataset_rows
from test_compact_greek_tts_pilot import _valid_rows


class CompactExtractionBoundariesTests(unittest.TestCase):
    def extract(self, root):
        return extract_stoma_f1(
            root, root / "output", parquet_module=_ParquetModule,
            artifact_validator=lambda *_: ValidationResult(True, "ready", 2, 2),
        )

    def test_truncated_wav_is_rejected_before_publication(self):
        rows = _dataset_rows()
        rows[0]["audio"]["bytes"] = rows[0]["audio"]["bytes"][:-1]
        with tempfile.TemporaryDirectory() as temporary, patch.object(_ParquetModule, "rows", rows):
            root = Path(temporary)
            with self.assertRaisesRegex(BackendError, "audio_wav_invalid"):
                self.extract(root)
            self.assertEqual(list(root.iterdir()), [])

    def test_preparation_failure_and_interruption_remove_temporary_tree(self):
        original_mkdir = Path.mkdir
        for phase in ("mkdir", "read"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                def mkdir(path, *args, **kwargs):
                    if phase == "mkdir" and path.name == "audio":
                        raise OSError("synthetic directory failure")
                    return original_mkdir(path, *args, **kwargs)
                with patch.object(Path, "mkdir", mkdir), patch.object(_ParquetModule.ParquetFile, "iter_batches", side_effect=KeyboardInterrupt):
                    with self.assertRaises((OSError, KeyboardInterrupt)):
                        self.extract(root)
                self.assertEqual(list(root.iterdir()), [])

    def test_section_must_match_filename_even_when_section_counts_match(self):
        rows = _valid_rows()
        first, second = rows[0], next(row for row in rows if row["section"] == "C1")
        first["section"], second["section"] = second["section"], first["section"]
        result = validate_f1_rows(rows)
        self.assertFalse(result.ready)
        self.assertEqual(result.code, "tts.f1_compact.dataset_contaminated")


if __name__ == "__main__":
    unittest.main()
