from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import _bootstrap  # noqa: F401

from jarvis_voice.macos.compact_greek_tts_pilot import (
    MINIMUM_ACQUISITION_FREE_BYTES,
    PINNED_DOWNLOAD_BYTES,
    PIPER_CHECKPOINT_ARTIFACTS,
    PIPER_RUNTIME_ARTIFACTS,
    STOMA_NON_F1_ROWS_TO_REJECT,
    STOMA_SECTION_COUNTS,
    STOMA_SHARDS,
    STOMA_SPEAKER_CLIPS,
    compact_greek_tts_preflight,
    pinned_f1_manifest,
    validate_f1_rows,
)


def _valid_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 0
    for section, count in STOMA_SECTION_COUNTS.items():
        for offset in range(count):
            index += 1
            session = ((index - 1) // 99) + 1
            sentence = ((index - 1) % 99) + 1
            rows.append(
                {
                    "speaker_id": "F1",
                    "section": section,
                    "file_name": f"Greek-F1-{section}-{session:02d}-{sentence:02d}.wav",
                    "text": "Ελληνικό κείμενο.",
                }
            )
    return rows


class CompactGreekTtsPilotTests(unittest.TestCase):
    def test_manifest_pins_minimal_subset_and_runtime(self) -> None:
        manifest = pinned_f1_manifest()
        self.assertEqual(manifest["dataset"]["clips"], STOMA_SPEAKER_CLIPS)
        self.assertEqual(manifest["dataset"]["non_f1_rows_to_reject"], STOMA_NON_F1_ROWS_TO_REJECT)
        self.assertEqual(len(manifest["dataset"]["shards"]), 2)
        self.assertEqual(manifest["runtime"]["version"], "1.6.0")
        self.assertEqual(manifest["runtime"]["license"], "GPL-3.0-or-later")
        self.assertFalse(manifest["qualification"]["network_allowed"])

    def test_download_total_is_derived_only_from_allow_list(self) -> None:
        expected = sum(
            item.size for item in STOMA_SHARDS + PIPER_CHECKPOINT_ARTIFACTS + PIPER_RUNTIME_ARTIFACTS
        )
        self.assertEqual(PINNED_DOWNLOAD_BYTES, expected)

    def test_preflight_is_read_only_and_does_not_install(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before = list(root.iterdir())
            report = compact_greek_tts_preflight(
                root,
                platform_name="darwin",
                machine="arm64",
                python_version=(3, 11, 14),
                pointer_bits=64,
                disk_usage=lambda path: SimpleNamespace(free=MINIMUM_ACQUISITION_FREE_BYTES),
                dependency_probe=lambda name, version: False,
            )
            after = list(root.iterdir())
        self.assertTrue(report["ready"])
        self.assertFalse(report["installation_performed"])
        self.assertFalse(report["model_downloaded"])
        self.assertFalse(report["network_used"])
        self.assertFalse(report["production_backend_changed"])
        self.assertEqual(before, after)

    def test_low_disk_and_wrong_python_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            low_disk = compact_greek_tts_preflight(
                root,
                platform_name="darwin",
                machine="arm64",
                python_version=(3, 11, 0),
                pointer_bits=64,
                disk_usage=lambda path: SimpleNamespace(free=MINIMUM_ACQUISITION_FREE_BYTES - 1),
            )
            wrong_python = compact_greek_tts_preflight(
                root,
                platform_name="darwin",
                machine="arm64",
                python_version=(3, 14, 0),
                pointer_bits=64,
                disk_usage=lambda path: SimpleNamespace(free=20 * 1024**3),
            )
        self.assertEqual(low_disk["code"], "tts.f1_compact.disk_space_insufficient")
        self.assertFalse(low_disk["ready"])
        self.assertEqual(wrong_python["code"], "tts.f1_compact.host_unsupported")
        self.assertFalse(wrong_python["ready"])

    def test_dataset_validation_accepts_only_complete_f1_subset(self) -> None:
        result = validate_f1_rows(_valid_rows())
        self.assertTrue(result.ready)
        self.assertEqual(result.rows_seen, STOMA_SPEAKER_CLIPS)

    def test_dataset_validation_rejects_f2_contamination_and_missing_rows(self) -> None:
        contaminated = _valid_rows()
        contaminated[-1] = {**contaminated[-1], "speaker_id": "F2"}
        self.assertEqual(
            validate_f1_rows(contaminated).code,
            "tts.f1_compact.dataset_contaminated",
        )
        self.assertEqual(
            validate_f1_rows(_valid_rows()[:-1]).code,
            "tts.f1_compact.dataset_incomplete",
        )


if __name__ == "__main__":
    unittest.main()
