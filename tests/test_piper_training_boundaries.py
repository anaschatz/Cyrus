from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
from jarvis_voice.errors import BackendError
from jarvis_voice import piper_mps_checkpoint as checkpoint
from jarvis_voice.piper_mps_training import piper_mps_training_commands


class PiperTrainingBoundariesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "lightning_logs/version_0/checkpoints/last.ckpt"
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"checkpoint")
        self.callbacks = [
            SimpleNamespace(monitor="val_mel"),
            SimpleNamespace(monitor="val_mos"),
        ]
        self.model_class = SimpleNamespace(
            configure_optimizers=Mock(),
            on_fit_start=Mock(),
            on_train_batch_start=Mock(),
        )
        self.training = SimpleNamespace(
            _DEFAULT_CALLBACKS=self.callbacks, VitsModel=self.model_class, main=Mock()
        )
        self.torch = SimpleNamespace(
            load=Mock(),
            serialization=SimpleNamespace(
                get_unsafe_globals_in_checkpoint=Mock(),
                safe_globals=lambda _: nullcontext(),
            ),
        )

    def protected_arguments(self):
        entrypoint = self.root / "entry.py"
        entrypoint.write_bytes(b"")
        return piper_mps_training_commands(
            self.root / "python",
            entrypoint,
            self.root / "data",
            self.root / "artifacts",
            self.root / "work",
        )["protected_mini_training_argv"][2:]

    def test_checkpoint_validators_handle_inline_values_and_mixed_duplicates(self):
        validators = (
            (
                checkpoint.WARM_START_ARGUMENT,
                lambda args: checkpoint.validate_piper_mps_checkpoint(
                    args,
                    lambda _: ["pathlib.PosixPath"],
                    expected_size=10,
                    expected_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
                ),
            ),
            (
                checkpoint.RESUME_ARGUMENT,
                lambda args: checkpoint.validate_piper_mps_resume_checkpoint(
                    args, lambda _: [], minimum_size=1
                ),
            ),
        )
        for argument, validate in validators:
            with self.subTest(argument=argument, mode="inline"):
                self.assertEqual(
                    validate(["fit", f"{argument}={self.path}"]), self.path
                )
            with self.subTest(argument=argument, mode="duplicate"):
                with self.assertRaises(BackendError):
                    validate(
                        ["fit", argument, str(self.path), f"{argument}={self.path}"]
                    )

    def test_inline_overrides_cannot_bypass_protected_training_checks(self):
        arguments = self.protected_arguments()
        for override in (
            "--trainer.max_steps=10000",
            "--trainer.max_epochs=999",
            "--trainer.fast_dev_run=true",
        ):
            with (
                self.subTest(override=override),
                patch.object(
                    checkpoint.importlib, "import_module", return_value=self.torch
                ),
                patch.object(
                    checkpoint, "validate_piper_mps_checkpoint", return_value=self.path
                ) as validator,
                patch.object(
                    checkpoint,
                    "_prepare_piper_main",
                    return_value=(self.training, tuple(self.callbacks), None, {}),
                ),
            ):
                with self.assertRaises(BackendError):
                    checkpoint.run_piper_mps_fit([*arguments, override])
                validator.assert_not_called()
        self.training.main.assert_not_called()

    def test_failed_callback_preparation_leaves_original_defaults_and_hooks(self):
        arguments = ["--model.mos_metric", "none"]
        originals = vars(self.model_class).copy()
        for phase in ("import", "construct"):
            with self.subTest(phase=phase):
                self.training._DEFAULT_CALLBACKS = self.callbacks

                def load(name):
                    if name == "piper.train.__main__":
                        return self.training
                    if phase == "import":
                        raise ImportError("synthetic callback import failure")
                    return SimpleNamespace(
                        ModelCheckpoint=Mock(
                            side_effect=RuntimeError(
                                "synthetic callback construction failure"
                            )
                        )
                    )

                with (
                    patch.object(
                        checkpoint.importlib, "import_module", side_effect=load
                    ),
                    self.assertRaises((ImportError, RuntimeError)),
                ):
                    checkpoint._prepare_piper_main(arguments, protected_f1=True)
                self.assertIs(self.training._DEFAULT_CALLBACKS, self.callbacks)
                self.assertEqual(vars(self.model_class), originals)

    def test_resume_load_remains_weights_only_and_restores_process_state(self):
        original_load, original_argv = self.torch.load, sys.argv
        self.training.main = lambda: self.torch.load(self.path, weights_only=False)
        modules = {"torch": self.torch, "piper.train.__main__": self.training}
        with (
            patch.object(
                checkpoint.importlib, "import_module", side_effect=modules.__getitem__
            ),
            patch.object(
                checkpoint,
                "validate_piper_mps_resume_checkpoint",
                return_value=self.path,
            ),
        ):
            checkpoint.run_piper_mps_fit(["fit", "--ckpt_path", str(self.path)])
        original_load.assert_called_once_with(self.path, weights_only=True)
        self.assertIs(self.torch.load, original_load)
        self.assertIs(sys.argv, original_argv)
        self.assertEqual(self.training._DEFAULT_CALLBACKS, self.callbacks)


if __name__ == "__main__":
    unittest.main()
