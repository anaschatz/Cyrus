from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jarvis_voice.errors import BackendError
from jarvis_voice.piper_mps_export import (
    LANGUAGE_METADATA,
    finalize_piper_mps_config,
    validate_piper_mps_export,
)


class PiperMpsExportTests(unittest.TestCase):
    def _paths(self, root: Path) -> tuple[Path, Path]:
        checkpoint = (
            root
            / "runs/lightning_logs/version_3/checkpoints/epoch=22-val_mel=0.4752.ckpt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"x" * 32)
        output = root / "export/el_GR-jarvis-f1-medium.onnx"
        output.parent.mkdir()
        output.with_name(f"{output.name}.json").write_text("{}", encoding="utf-8")
        return checkpoint, output

    def test_accepts_expected_local_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            checkpoint, output = self._paths(Path(temp))
            observed = validate_piper_mps_export(
                checkpoint,
                output,
                lambda _: (),
                minimum_size=1,
            )
        self.assertEqual(observed, (checkpoint, output))

    def test_accepts_only_bounded_protected_step_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = (
                root
                / "protected-mini/runs/lightning_logs/version_0/checkpoints/step=100.ckpt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"protected")
            output = root / "export/el_GR-jarvis-f1-medium.onnx"
            output.parent.mkdir()
            output.with_name(f"{output.name}.json").write_text(
                "{}", encoding="utf-8"
            )
            observed = validate_piper_mps_export(
                checkpoint,
                output,
                lambda _: (),
                minimum_size=1,
            )
        self.assertEqual(observed, (checkpoint, output))

    def test_rejects_unapproved_step_or_wrong_protected_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "export/el_GR-jarvis-f1-medium.onnx"
            output.parent.mkdir()
            output.with_name(f"{output.name}.json").write_text(
                "{}", encoding="utf-8"
            )
            for relative in (
                "protected-mini/runs/lightning_logs/version_0/checkpoints/step=150.ckpt",
                "other/runs/lightning_logs/version_0/checkpoints/step=100.ckpt",
            ):
                checkpoint = root / relative
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"rejected")
                with self.assertRaises(BackendError) as raised:
                    validate_piper_mps_export(
                        checkpoint,
                        output,
                        lambda _: (),
                        minimum_size=1,
                    )
                self.assertEqual(
                    raised.exception.code,
                    "tts.f1_mps.export_checkpoint_invalid",
                )

    def test_rejects_unsafe_checkpoint_globals(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            checkpoint, output = self._paths(Path(temp))
            with self.assertRaises(BackendError) as raised:
                validate_piper_mps_export(
                    checkpoint,
                    output,
                    lambda _: ("malicious.Type",),
                    minimum_size=1,
                )
        self.assertEqual(
            raised.exception.code,
            "tts.f1_mps.export_checkpoint_globals_invalid",
        )

    def test_rejects_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            checkpoint, output = self._paths(Path(temp))
            output.write_bytes(b"partial")
            with self.assertRaises(BackendError) as raised:
                validate_piper_mps_export(
                    checkpoint,
                    output,
                    lambda _: (),
                    minimum_size=1,
                )
        self.assertEqual(raised.exception.code, "tts.f1_mps.export_output_invalid")

    def test_finalizes_generated_config_with_canonical_greek_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "el_GR-jarvis-f1-medium.onnx.json"
            config.write_text(
                '{"audio":{"sample_rate":22050},"espeak":{"voice":"el"},'
                '"phoneme_type":"espeak","num_speakers":1,"num_symbols":256,'
                '"phoneme_id_map":{"_ ":[0]}}',
                encoding="utf-8",
            )
            finalize_piper_mps_config(config)
            import json

            payload = json.loads(config.read_text(encoding="utf-8"))
        self.assertEqual(payload["language"], LANGUAGE_METADATA)

    def test_rejects_conflicting_language_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "el_GR-jarvis-f1-medium.onnx.json"
            config.write_text(
                '{"audio":{"sample_rate":22050},"espeak":{"voice":"el"},'
                '"phoneme_type":"espeak","num_speakers":1,"num_symbols":256,'
                '"phoneme_id_map":{"_ ":[0]},"language":{"code":"en"}}',
                encoding="utf-8",
            )
            with self.assertRaises(BackendError) as raised:
                finalize_piper_mps_config(config)
        self.assertEqual(raised.exception.code, "tts.f1_mps.export_language_conflict")


if __name__ == "__main__":
    unittest.main()
