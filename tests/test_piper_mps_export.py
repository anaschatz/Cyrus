from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jarvis_voice.errors import BackendError
from jarvis_voice.piper_mps_export import validate_piper_mps_export


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


if __name__ == "__main__":
    unittest.main()
