from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import threading
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_voice.app_paths import ApplicationPaths, require_external_mutable_path
from jarvis_voice.cli import main
from jarvis_voice.distribution import _sha256, build_distribution_manifest, verify_distribution, write_distribution_manifest
from jarvis_voice.errors import BackendError, ProtocolError, StateTransitionError
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
        self.assertIn("& $VoicePython -m unittest discover -s tests -v", windows)
        self.assertIn(
            "& $OrchestratorPython -m unittest discover -s orchestrator/tests -v",
            windows,
        )
        self.assertIn(
            "& $InferencePython -m unittest discover -s inference/tests -v",
            windows,
        )
        self.assertIn(
            "& $InferencePython .\\inference\\tests\\sdk_conformance.py -v",
            windows,
        )
        self.assertIn("orchestrator/src orchestrator/tests", windows)
        self.assertIn("windows_asr_collector.spec", windows)
        self.assertIn("cyrus-windows-asr-collector.exe", windows)
        self.assertIn("1.2.1:4.8.1:6.21.0", windows)
        self.assertIn('[ValidateSet("QualificationRunners", "VoicePackage")]', windows)
        self.assertIn("-CudaRuntimeManifest is required", windows)
        self.assertIn("verify-runners --manifest", windows)
        self.assertIn("changed during Voice packaging", windows)
        self.assertIn('"boot-smoke"', windows)
        self.assertIn("Assert-CyrusTrackedTreeClean", windows)
        self.assertIn("-IncludeUntracked", windows)
        self.assertIn("Assert-CyrusPathOutsideRepository", windows)
        self.assertIn("existing Voice package output directory was rejected", windows)
        self.assertIn('inference\\.venv\\Scripts\\python.exe', windows)
        self.assertNotIn("py -3.11 -m", windows)
        self.assertNotIn("$env:PYTHONPATH", windows)
        self.assertNotIn("$env:CUDA_PATH", windows)
        self.assertIn('"OPENROUTER_API_KEY"', windows)
        self.assertIn('"JARVIS_INFERENCE_API_KEY"', windows)

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
    def test_concurrent_rotation_keeps_each_record_in_one_bounded_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            logger = SafeEventLogger(path, maximum_bytes=1, backup_count=2)
            logger.write("initial")
            rotated, resume, second_rotated = threading.Event(), threading.Event(), threading.Event()
            original_rotate = logger._rotate_if_needed
            errors = []

            def rotate(size):
                original_rotate(size)
                if threading.current_thread().name == "first":
                    rotated.set()
                    resume.wait(1)
                else:
                    second_rotated.set()

            def write(name):
                try:
                    logger.write(name)
                except Exception as exc:
                    errors.append(exc)

            with patch.object(logger, "_rotate_if_needed", side_effect=rotate):
                first = threading.Thread(target=write, args=("first",), name="first")
                second = threading.Thread(target=write, args=("second",), name="second")
                first.start()
                try:
                    self.assertTrue(rotated.wait(1))
                    second.start()
                    second_rotated.wait(0.1)
                finally:
                    resume.set()
                    first.join(1)
                    if second.ident is not None:
                        second.join(1)
            self.assertFalse(first.is_alive() or second.is_alive())
            self.assertFalse(errors)
            events = []
            for file in path.parent.glob("events.jsonl*"):
                lines = file.read_text().splitlines()
                self.assertEqual(len(lines), 1)
                events.append(json.loads(lines[0])["event"])
            self.assertCountEqual(events, ["initial", "first", "second"])

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
    def test_cli_reports_protocol_and_state_errors_without_private_messages(self) -> None:
        for error in (ProtocolError("protocol.test", "private detail"), StateTransitionError("state.test", "private detail")):
            stdout = io.StringIO()
            with self.subTest(code=error.code), patch("jarvis_voice.cli.doctor_report", side_effect=error), contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["doctor", "--model-dir", "/unused"]), 2)
            self.assertEqual(json.loads(stdout.getvalue())["error"]["code"], error.code)
            self.assertNotIn("private detail", stdout.getvalue())

    def test_cli_output_failure_is_reported_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "missing" / "report.json"
            stdout = io.StringIO()
            with patch("jarvis_voice.cli.doctor_report", return_value={"ready": False}), contextlib.redirect_stdout(stdout):
                result = main(["doctor", "--model-dir", temporary, "--output", str(output)])
            self.assertEqual(result, 2)
            self.assertEqual(json.loads(stdout.getvalue())["error"]["code"], "qualification.output_directory_missing")

    def test_manifest_write_error_survives_failed_temporary_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._distribution(root)
            with patch.object(Path, "replace", side_effect=PermissionError("private replace failure")), patch.object(Path, "unlink", side_effect=PermissionError("private cleanup failure")):
                with self.assertRaisesRegex(BackendError, "packaging.manifest_write_failed"):
                    write_distribution_manifest(root)

    def test_artifact_read_failures_are_typed_for_build_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._distribution(root)
            write_distribution_manifest(root)
            real_open = Path.open

            def unreadable(path, *args, **kwargs):
                if path.name == "runtime.dll":
                    raise PermissionError("private path must not escape")
                return real_open(path, *args, **kwargs)

            for operation in (build_distribution_manifest, verify_distribution):
                with self.subTest(operation=operation.__name__), patch.object(Path, "open", unreadable):
                    with self.assertRaisesRegex(BackendError, "packaging.distribution_unreadable"):
                        operation(root)

    def test_windows_reparse_directory_is_not_traversed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._distribution(root)
            write_distribution_manifest(root)
            real_lstat = Path.lstat

            def reparse(path):
                metadata = real_lstat(path)
                if path.name == "_internal":
                    return SimpleNamespace(st_mode=metadata.st_mode, st_file_attributes=0x400)
                return metadata

            for operation in (build_distribution_manifest, verify_distribution):
                with self.subTest(operation=operation.__name__), patch.object(Path, "lstat", reparse):
                    with self.assertRaisesRegex(BackendError, "packaging.symlink_forbidden"):
                        operation(root)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires FIFO support")
    def test_manifest_fifo_is_rejected_before_opening(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.mkfifo(root / "distribution-manifest.json")
            with patch.object(Path, "read_bytes", side_effect=AssertionError("must not open FIFO")):
                with self.assertRaisesRegex(BackendError, "packaging.manifest_unreadable"):
                    verify_distribution(root)

    def test_unreadable_subdirectory_cannot_be_silently_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._distribution(root)
            real_scandir = os.scandir

            def unreadable(path):
                if Path(path).name == "_internal":
                    raise PermissionError("private path must not escape")
                return real_scandir(path)

            with patch.object(os, "scandir", side_effect=unreadable):
                with self.assertRaisesRegex(BackendError, "packaging.distribution_unreadable"):
                    write_distribution_manifest(root)
            self.assertFalse((root / "distribution-manifest.json").exists())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires FIFO support")
    def test_nonregular_extra_file_cannot_pass_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._distribution(root)
            write_distribution_manifest(root)
            os.mkfifo(root / "unexpected.pipe")
            with self.assertRaisesRegex(BackendError, "packaging.artifact_type_invalid"):
                verify_distribution(root)

    def test_malformed_json_and_schema_types_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._distribution(root)
            manifest = write_distribution_manifest(root)
            for raw in (b"[" * 2000 + b"0" + b"]" * 2000, b"9" * 5000, b"\xff"):
                with self.subTest(prefix=raw[:10]):
                    manifest.write_bytes(raw)
                    with self.assertRaisesRegex(BackendError, "packaging.manifest_unreadable"):
                        verify_distribution(root)
            data = build_distribution_manifest(root)
            for version in (True, 1.0):
                with self.subTest(version=version):
                    data["schema_version"] = version
                    manifest.write_text(json.dumps(data))
                    self.assertEqual(verify_distribution(root).code, "packaging.manifest_schema_invalid")

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
            for path in ("../escape.exe", "./JARVIS Voice.exe"):
                data["files"][0]["path"] = path
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
            with contextlib.redirect_stdout(output), patch("jarvis_voice.distribution._sha256", wraps=_sha256) as hasher:
                self.assertEqual(main(["dist-manifest", "--directory", str(root)]), 0)
            self.assertEqual(hasher.call_count, 2)
            self.assertEqual(json.loads(output.getvalue())["files"], 2)
            self.assertFalse(json.loads(output.getvalue())["network_used"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["verify-dist", "--directory", str(root)]), 0)
            self.assertTrue(json.loads(output.getvalue())["ready"])


if __name__ == "__main__":
    unittest.main()
