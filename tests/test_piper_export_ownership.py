from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
from jarvis_voice.errors import BackendError
from jarvis_voice import piper_mps_export as export


class PiperExportOwnershipTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "runs/lightning_logs/version_0/checkpoints/epoch=1-val_mel=0.5.ckpt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"checkpoint")
        self.output = self.root / export.EXPECTED_OUTPUT_NAME
        self.config = self.output.with_name(self.output.name + ".json")
        self.config_bytes = json.dumps({"audio": {"sample_rate": 22050}, "espeak": {"voice": "el"}, "phoneme_type": "espeak", "num_speakers": 1, "num_symbols": 256, "phoneme_id_map": {"_": [0]}}).encode()
        self.config.write_bytes(self.config_bytes)
        self.partial = self.output.with_name("." + self.output.name + ".part")

    @contextmanager
    def competing_writer(self, partial):
        real_io, real_os = io.open, os.open
        injected = False
        def inject(path):
            nonlocal injected
            if not injected and not isinstance(path, int) and Path(path) == partial:
                injected = True
                with real_io(partial, "wb") as stream:
                    stream.write(b"another export owns this")
        def io_open(path, *args, **kwargs):
            inject(path)
            return real_io(path, *args, **kwargs)
        def os_open(path, *args, **kwargs):
            inject(path)
            return real_os(path, *args, **kwargs)
        with patch.object(io, "open", io_open), patch.object(os, "open", os_open):
            yield
        self.assertTrue(injected)

    def fake_runtime(self):
        model = Mock(n_vocab=256, n_speakers=1)
        torch = SimpleNamespace(load=Mock(), serialization=SimpleNamespace(get_unsafe_globals_in_checkpoint=lambda _: ()), no_grad=nullcontext, long="long", randint=Mock(return_value=Mock(size=Mock(return_value=50))), LongTensor=Mock(), FloatTensor=Mock(), onnx=SimpleNamespace(export=Mock(side_effect=lambda **kwargs: Path(kwargs["f"]).write_bytes(b"validated model"))))
        loader = Mock(side_effect=lambda path, **kwargs: (torch.load(path), SimpleNamespace(model_g=model))[1])
        onnx = SimpleNamespace(load=lambda path: Path(path).read_bytes(), checker=SimpleNamespace(check_model=Mock()))
        modules = {"torch": torch, "piper.train.vits.lightning": SimpleNamespace(VitsModel=SimpleNamespace(load_from_checkpoint=loader)), "onnx": onnx}
        self.enterContext(patch.object(export.importlib, "import_module", side_effect=modules.__getitem__))
        validate = export.validate_piper_mps_export
        self.enterContext(patch.object(export, "validate_piper_mps_export", side_effect=lambda checkpoint, output, probe: validate(checkpoint, output, probe, minimum_size=1)))
        return torch, loader, onnx

    def test_config_cannot_overwrite_another_writers_partial(self):
        partial = self.config.with_name("." + self.config.name + ".part")
        with self.competing_writer(partial):
            with self.assertRaisesRegex(BackendError, "export_config_partial_exists"):
                export.finalize_piper_mps_config(self.config)
        self.assertEqual(partial.read_bytes(), b"another export owns this")
        self.assertEqual(self.config.read_bytes(), self.config_bytes)

    def test_export_cannot_overwrite_another_writers_partial(self):
        torch, loader, _ = self.fake_runtime()
        original_load = torch.load
        with self.competing_writer(self.partial):
            with self.assertRaisesRegex(BackendError, "export_partial_exists"):
                export.run_piper_mps_export(self.checkpoint, self.output)
        self.assertEqual(self.partial.read_bytes(), b"another export owns this")
        self.assertFalse(self.output.exists())
        self.assertIs(torch.load, original_load)

    def test_owned_partial_is_cleaned_on_failure_and_success_publishes(self):
        torch, loader, onnx = self.fake_runtime()
        original_load = torch.load
        for operation in (loader, torch.onnx.export, onnx.checker.check_model):
            with self.subTest(operation=operation):
                original_effect = operation.side_effect
                operation.side_effect = KeyboardInterrupt("synthetic interruption")
                try:
                    with self.assertRaises(KeyboardInterrupt):
                        export.run_piper_mps_export(self.checkpoint, self.output)
                finally:
                    operation.side_effect = original_effect
                self.assertFalse(self.partial.exists())
                self.assertFalse(self.output.exists())
                self.assertIs(torch.load, original_load)
        self.partial.unlink(missing_ok=True)
        export.run_piper_mps_export(self.checkpoint, self.output)
        self.assertEqual(self.output.read_bytes(), b"validated model")
        self.assertFalse(self.partial.exists())
        self.assertEqual(json.loads(self.config.read_text())["language"], export.LANGUAGE_METADATA)
        self.assertIs(torch.load, original_load)
        original_load.assert_called_with(self.checkpoint, weights_only=True)

    def test_completed_output_after_validation_is_preserved(self):
        self.fake_runtime()
        validator = export.validate_piper_mps_export
        def delayed_validation(*args):
            result = validator(*args)
            self.output.write_bytes(b"another export completed")
            return result
        with patch.object(export, "validate_piper_mps_export", side_effect=delayed_validation):
            with self.assertRaisesRegex(BackendError, "export_output_invalid"):
                export.run_piper_mps_export(self.checkpoint, self.output)
        self.assertEqual(self.output.read_bytes(), b"another export completed")
        self.assertFalse(self.partial.exists())

    def test_config_interruption_cleans_its_reserved_partial(self):
        partial = self.config.with_name("." + self.config.name + ".part")
        with patch.object(export.json, "dumps", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            export.finalize_piper_mps_config(self.config)
        self.assertFalse(partial.exists())
        self.assertEqual(self.config.read_bytes(), self.config_bytes)

    def test_success_does_not_delete_a_subsequent_writers_partial(self):
        self.fake_runtime()
        real_replace = os.replace
        for configuration in (True, False):
            with self.subTest(configuration=configuration):
                partial = self.config.with_name("." + self.config.name + ".part") if configuration else self.partial
                def publish(source, destination):
                    real_replace(source, destination)
                    if source == partial:
                        partial.write_bytes(b"subsequent writer")
                with patch.object(export.os, "replace", side_effect=publish):
                    if configuration:
                        export.finalize_piper_mps_config(self.config)
                    else:
                        export.run_piper_mps_export(self.checkpoint, self.output)
                self.assertEqual(partial.read_bytes(), b"subsequent writer")


if __name__ == "__main__":
    unittest.main()
