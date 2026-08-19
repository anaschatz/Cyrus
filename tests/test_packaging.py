from __future__ import annotations

import contextlib
import io
import json
import tempfile
import tomllib
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.app_paths import ApplicationPaths, require_external_mutable_path
from jarvis_voice.cli import main
from jarvis_voice.distribution import build_distribution_manifest, verify_distribution, write_distribution_manifest
from jarvis_voice.errors import BackendError
from jarvis_voice.safe_log import SafeEventLogger
from jarvis_voice import __version__


class ApplicationPathTests(unittest.TestCase):
    def test_release_metadata_versions_are_consistent(self) -> None:
        root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text())
        self.assertEqual(project["project"]["version"], __version__)
        self.assertIn("version=\"0.8.0.0\"", (root / "packaging" / "windows-manifest.xml").read_text())
        self.assertIn("filevers=(0, 8, 0, 0)", (root / "packaging" / "windows-version-info.txt").read_text())

    def test_native_builds_gate_on_voice_and_full_orchestrator_suites(self) -> None:
        root = Path(__file__).resolve().parents[1]
        macos = (root / "scripts" / "build_macos.sh").read_text()
        windows = (root / "scripts" / "build_windows.ps1").read_text()

        self.assertIn("python3.11 -m unittest discover -s tests -v", macos)
        self.assertIn(
            "python3.11 -m unittest discover -s orchestrator/tests -v",
            macos,
        )
        self.assertIn("orchestrator/src orchestrator/tests", macos)
        self.assertIn("py -3.11 -m unittest discover -s tests -v", windows)
        self.assertIn(
            "py -3.11 -m unittest discover -s orchestrator/tests -v",
            windows,
        )
        self.assertIn("orchestrator/src orchestrator/tests", windows)

    def test_per_user_tree_contains_no_machine_or_user_derived_name(self) -> None:
        paths = ApplicationPaths.from_environment({"LOCALAPPDATA": str(Path.cwd() / ".test-local-appdata")})
        self.assertEqual(paths.root.name, "JARVIS Voice")
        self.assertEqual(paths.default_config_file.name, "voice.json")
        self.assertEqual({paths.config.name, paths.models.name, paths.logs.name, paths.state.name}, {"config", "models", "logs", "state"})

    def test_missing_local_app_data_is_typed(self) -> None:
        with self.assertRaisesRegex(BackendError, "application.local_app_data_missing"):
            ApplicationPaths.from_environment({})
        with self.assertRaisesRegex(BackendError, "application.local_app_data_invalid"):
            ApplicationPaths.from_environment({"LOCALAPPDATA": "relative/appdata"})

    def test_mutable_data_inside_bundle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(BackendError, "packaging.mutable_path_inside_bundle"):
                require_external_mutable_path(root / "config" / "voice.json", root)
            outside = root.parent / "voice.json"
            self.assertEqual(require_external_mutable_path(outside, root), outside.resolve())


class SafeLogTests(unittest.TestCase):
    def test_log_accepts_only_content_free_tokens_and_rotates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            logger = SafeEventLogger(path, maximum_bytes=130, backup_count=1)
            logger.write("voice.lifecycle.start", status="starting")
            logger.write("voice.lifecycle.error", status="error", code="runtime.test_failure")
            self.assertTrue(path.exists())
            self.assertTrue(path.with_name("events.jsonl.1").exists())
            combined = path.read_text() + path.with_name("events.jsonl.1").read_text()
            self.assertNotIn("transcript", combined)
            with self.assertRaisesRegex(BackendError, "logging.unsafe_token"):
                logger.write("voice.lifecycle.error", code="/private/secret transcript")


class DistributionManifestTests(unittest.TestCase):
    def _distribution(self, root: Path) -> None:
        (root / "_internal").mkdir()
        (root / "JARVIS Voice.exe").write_bytes(b"exe")
        (root / "_internal" / "runtime.dll").write_bytes(b"dll")

    def test_manifest_is_sorted_deterministic_and_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._distribution(root)
            first = build_distribution_manifest(root)
            second = build_distribution_manifest(root)
            self.assertEqual(first, second)
            self.assertEqual([item["path"] for item in first["files"]], ["JARVIS Voice.exe", "_internal/runtime.dll"])
            write_distribution_manifest(root)
            result = verify_distribution(root)
            self.assertTrue(result.ok)
            self.assertEqual(result.files_checked, 2)

    def test_missing_unexpected_size_and_hash_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._distribution(root)
            write_distribution_manifest(root)
            target = root / "_internal" / "runtime.dll"
            target.unlink()
            self.assertEqual(verify_distribution(root).code, "packaging.artifact_missing")
            target.write_bytes(b"dll")
            (root / "unexpected.txt").write_text("x")
            self.assertEqual(verify_distribution(root).code, "packaging.artifact_unexpected")
            (root / "unexpected.txt").unlink()
            target.write_bytes(b"different-size")
            self.assertEqual(verify_distribution(root).code, "packaging.artifact_size_mismatch")
            target.write_bytes(b"bad")
            self.assertEqual(verify_distribution(root).code, "packaging.artifact_hash_mismatch")

    def test_traversal_duplicate_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._distribution(root)
            manifest = write_distribution_manifest(root)
            data = json.loads(manifest.read_text())
            data["files"][0]["path"] = "../escape.exe"
            manifest.write_text(json.dumps(data))
            self.assertEqual(verify_distribution(root).code, "packaging.manifest_schema_invalid")
            manifest.unlink()
            try:
                (root / "link.dll").symlink_to(root / "_internal" / "runtime.dll")
            except (OSError, NotImplementedError):
                return
            with self.assertRaisesRegex(BackendError, "packaging.symlink_forbidden"):
                build_distribution_manifest(root)

    def test_cli_writes_and_verifies_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._distribution(root)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["dist-manifest", "--directory", str(root)]), 0)
            self.assertFalse(json.loads(output.getvalue())["network_used"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["verify-dist", "--directory", str(root)]), 0)
            self.assertTrue(json.loads(output.getvalue())["ready"])


if __name__ == "__main__":
    unittest.main()
