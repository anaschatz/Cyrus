from __future__ import annotations

import hashlib
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.piper_mps_checkpoint import (
    run_piper_mps_fit,
    validate_piper_mps_checkpoint,
    validate_piper_mps_resume_checkpoint,
)


class PiperMpsCheckpointTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, list[str], str]:
        checkpoint = root / "warm-start.ckpt"
        checkpoint.write_bytes(b"pinned checkpoint")
        argv = ["fit", "--model.warmstart_ckpt", str(checkpoint)]
        return checkpoint, argv, hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def test_accepts_only_pinned_checkpoint_and_posix_path_global(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, argv, checksum = self.fixture(Path(temporary))
            result = validate_piper_mps_checkpoint(
                argv,
                lambda _path: ["pathlib.PosixPath"],
                expected_size=checkpoint.stat().st_size,
                expected_sha256=checksum,
            )
        self.assertEqual(result, checkpoint)

    def test_rejects_any_additional_unsafe_global(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, argv, checksum = self.fixture(Path(temporary))
            with self.assertRaisesRegex(
                BackendError, "tts.f1_mps.checkpoint_globals_mismatch"
            ):
                validate_piper_mps_checkpoint(
                    argv,
                    lambda _path: ["pathlib.PosixPath", "unexpected.Global"],
                    expected_size=checkpoint.stat().st_size,
                    expected_sha256=checksum,
                )

    def test_rejects_modified_checkpoint_before_global_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, argv, _checksum = self.fixture(Path(temporary))
            scanned = False

            def probe(_path: str) -> list[str]:
                nonlocal scanned
                scanned = True
                return ["pathlib.PosixPath"]

            with self.assertRaisesRegex(
                BackendError, "tts.f1_mps.checkpoint_hash_mismatch"
            ):
                validate_piper_mps_checkpoint(
                    argv,
                    probe,
                    expected_size=checkpoint.stat().st_size,
                    expected_sha256="0" * 64,
                )
        self.assertFalse(scanned)

    def test_rejects_duplicate_checkpoint_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, argv, checksum = self.fixture(Path(temporary))
            with self.assertRaisesRegex(
                BackendError, "tts.f1_mps.checkpoint_argument_invalid"
            ):
                validate_piper_mps_checkpoint(
                    [*argv, "--model.warmstart_ckpt", str(checkpoint)],
                    lambda _path: ["pathlib.PosixPath"],
                    expected_size=checkpoint.stat().st_size,
                    expected_sha256=checksum,
                )

    def test_runner_scopes_safe_global_and_restores_argv(self) -> None:
        state = {"inside": False, "called": False}

        @contextmanager
        def safe_globals(_values: list[type[Path]]) -> object:
            state["inside"] = True
            try:
                yield
            finally:
                state["inside"] = False

        serialization = SimpleNamespace(
            get_unsafe_globals_in_checkpoint=lambda _path: ["pathlib.PosixPath"],
            safe_globals=safe_globals,
        )
        load_calls: list[tuple[object, dict[str, object]]] = []

        def torch_load(source: object, *_args: object, **kwargs: object) -> object:
            load_calls.append((source, kwargs))
            return {}

        fake_torch = SimpleNamespace(serialization=serialization, load=torch_load)
        original_argv = list(__import__("sys").argv)
        checkpoint = Path("/private/tmp/pinned.ckpt")
        callbacks = [
            SimpleNamespace(monitor="val_mel"),
            SimpleNamespace(monitor="val_mos"),
        ]
        training_module = SimpleNamespace(
            _DEFAULT_CALLBACKS=callbacks,
            main=lambda: None,
        )

        def run_training() -> None:
            state["called"] = True
            self.assertTrue(state["inside"])
            self.assertEqual(
                __import__("sys").argv,
                [
                    "piper.train",
                    "fit",
                    "--model.mos_metric",
                    "none",
                    "--model.warmstart_ckpt",
                    str(checkpoint),
                ],
            )
            self.assertEqual(
                [callback.monitor for callback in training_module._DEFAULT_CALLBACKS],
                ["val_mel"],
            )
            fake_torch.load(str(checkpoint), weights_only=False)
            fake_torch.load("/private/tmp/other.ckpt", weights_only=False)

        training_module.main = run_training

        with (
            mock.patch(
                "jarvis_voice.piper_mps_checkpoint.validate_piper_mps_checkpoint",
                return_value=checkpoint,
            ),
            mock.patch(
                "jarvis_voice.piper_mps_checkpoint.importlib.import_module",
                side_effect=lambda name: (
                    fake_torch if name == "torch" else training_module
                ),
            ),
        ):
            run_piper_mps_fit(
                [
                    "fit",
                    "--model.mos_metric",
                    "none",
                    "--model.warmstart_ckpt",
                    str(checkpoint),
                ]
            )
        self.assertTrue(state["called"])
        self.assertFalse(state["inside"])
        self.assertEqual(__import__("sys").argv, original_argv)
        self.assertIs(fake_torch.load, torch_load)
        self.assertEqual(training_module._DEFAULT_CALLBACKS, callbacks)
        self.assertEqual(
            load_calls,
            [
                (str(checkpoint), {"weights_only": True}),
                ("/private/tmp/other.ckpt", {"weights_only": False}),
            ],
        )

    def test_accepts_local_last_checkpoint_with_no_unsafe_globals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = (
                Path(temporary)
                / "lightning_logs/version_0/checkpoints/last.ckpt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"resume")
            result = validate_piper_mps_resume_checkpoint(
                [
                    "fit",
                    "--ckpt_path",
                    str(checkpoint),
                    "--model.warmstart_ckpt",
                    "null",
                ],
                lambda _path: [],
                minimum_size=1,
                maximum_size=1024,
            )
        self.assertEqual(result, checkpoint)

    def test_resume_rejects_warmstart_or_unsafe_globals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = (
                Path(temporary)
                / "lightning_logs/version_1/checkpoints/last.ckpt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"resume")
            with self.assertRaisesRegex(
                BackendError, "tts.f1_mps.resume_argument_invalid"
            ):
                validate_piper_mps_resume_checkpoint(
                    [
                        "fit",
                        "--ckpt_path",
                        str(checkpoint),
                        "--model.warmstart_ckpt",
                        "/tmp/warm.ckpt",
                    ],
                    lambda _path: [],
                    minimum_size=1,
                    maximum_size=1024,
                )
            with self.assertRaisesRegex(
                BackendError, "tts.f1_mps.resume_checkpoint_globals_invalid"
            ):
                validate_piper_mps_resume_checkpoint(
                    ["fit", "--ckpt_path", str(checkpoint)],
                    lambda _path: ["unexpected.Global"],
                    minimum_size=1,
                    maximum_size=1024,
                )

    def test_runner_treats_explicit_null_warmstart_as_resume(self) -> None:
        checkpoint = Path("/tmp/lightning_logs/version_0/checkpoints/last.ckpt")
        fake_torch = SimpleNamespace(
            serialization=SimpleNamespace(
                get_unsafe_globals_in_checkpoint=lambda _path: [],
                safe_globals=lambda _values: mock.MagicMock(),
            ),
            load=lambda *_args, **_kwargs: {},
        )
        training_module = SimpleNamespace(
            _DEFAULT_CALLBACKS=[
                SimpleNamespace(monitor="val_mel"),
                SimpleNamespace(monitor="val_mos"),
            ],
            main=lambda: None,
        )
        with (
            mock.patch(
                "jarvis_voice.piper_mps_checkpoint.validate_piper_mps_resume_checkpoint",
                return_value=checkpoint,
            ) as resume_probe,
            mock.patch(
                "jarvis_voice.piper_mps_checkpoint.importlib.import_module",
                side_effect=lambda name: (
                    fake_torch if name == "torch" else training_module
                ),
            ),
        ):
            run_piper_mps_fit(
                [
                    "fit",
                    "--model.mos_metric",
                    "none",
                    "--model.warmstart_ckpt",
                    "null",
                    "--ckpt_path",
                    str(checkpoint),
                ]
            )
        resume_probe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
