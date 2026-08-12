from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.piper_mps_training import (
    GIB,
    LOCKED_LIGHTNING_VERSION,
    LOCKED_TORCH_VERSION,
    PIPER_REVISION,
    PIPER_VERSION,
    piper_mps_training_commands,
    piper_mps_training_readiness,
)


class PiperMpsTrainingTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        data = root / "data"
        artifacts = root / "artifacts"
        work = root / "work"
        source = root / "source"
        for directory in (data, artifacts / "runtime", work, source):
            directory.mkdir(parents=True)
        return data, artifacts, work, source

    def report(self, root: Path, **overrides: object) -> dict[str, object]:
        data, artifacts, work, source = self.fixture(root)
        versions = {
            "piper-tts": PIPER_VERSION,
            "torch": LOCKED_TORCH_VERSION,
            "lightning": LOCKED_LIGHTNING_VERSION,
        }
        arguments = {
            "source_root": source,
            "platform_name": "darwin",
            "machine": "arm64",
            "python_version": (3, 11, 14),
            "total_memory_bytes": 16 * GIB,
            "disk_usage": lambda _path: SimpleNamespace(free=24 * GIB),
            "compiler_path": "/usr/bin/clang",
            "distribution_probe": versions.get,
            "mps_probe": lambda: (True, True),
            "source_revision_probe": lambda _path: PIPER_REVISION,
        }
        arguments.update(overrides)
        with mock.patch(
            "jarvis_voice.piper_mps_training.build_piper_f1_handoff",
            return_value={"totals": {"files": 1306, "bytes": 1_400_000_000}},
        ), mock.patch(
            "jarvis_voice.piper_mps_training.validate_artifact_set",
            return_value=SimpleNamespace(ready=True),
        ), mock.patch(
            "jarvis_voice.piper_mps_training._wheel_has_training_core",
            return_value=False,
        ):
            return piper_mps_training_readiness(data, artifacts, work, **arguments)

    def test_ready_mps_profile_is_strictly_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self.report(Path(temporary))
        self.assertTrue(report["ready"])
        self.assertTrue(report["smoke_ready"])
        self.assertEqual(report["profile"]["batch_size"], 1)
        self.assertEqual(report["profile"]["precision"], "32-true")
        self.assertTrue(report["profile"]["main_llm_must_be_stopped"])

    def test_missing_source_is_materializable_but_not_smoke_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self.report(
                Path(temporary), source_revision_probe=lambda _path: None
            )
        self.assertTrue(report["materialization_ready"])
        self.assertFalse(report["smoke_ready"])
        self.assertEqual(report["code"], "tts.f1_mps.source_checkout_required")

    def test_disk_allows_smoke_but_blocks_full_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self.report(
                Path(temporary),
                disk_usage=lambda _path: SimpleNamespace(free=16 * GIB),
            )
        self.assertTrue(report["smoke_ready"])
        self.assertFalse(report["ready"])
        self.assertEqual(report["code"], "tts.f1_mps.full_training_disk_insufficient")

    def test_command_profile_uses_mps_batch_one_and_fast_dev_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("data", "artifacts", "work"):
                (root / name).mkdir()
            python = root / "python"
            commands = piper_mps_training_commands(
                python, root / "data", root / "artifacts", root / "work"
            )
        smoke = commands["smoke_argv"]
        initial = commands["initial_training_argv"]
        self.assertEqual(smoke[smoke.index("--trainer.accelerator") + 1], "mps")
        self.assertEqual(smoke[smoke.index("--data.batch_size") + 1], "1")
        self.assertIn("--trainer.fast_dev_run", smoke)
        self.assertNotIn("--trainer.fast_dev_run", initial)

    def test_command_profile_preserves_virtualenv_python_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("data", "artifacts", "work"):
                (root / name).mkdir()
            base_python = root / "base-python"
            base_python.write_bytes(b"")
            environment_python = root / "environment/bin/python"
            environment_python.parent.mkdir(parents=True)
            environment_python.symlink_to(base_python)
            commands = piper_mps_training_commands(
                environment_python,
                root / "data",
                root / "artifacts",
                root / "work",
            )
        self.assertEqual(commands["smoke_argv"][0], str(environment_python))


if __name__ == "__main__":
    unittest.main()
