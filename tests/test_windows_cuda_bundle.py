from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
from jarvis_inference.windows_process import WindowsProcessFailure

from jarvis_voice.windows.cuda_bundle import (
    CUDA_BUNDLE_SCHEMA,
    QUALIFICATION_RUNNERS,
    REQUIRED_CUDA_DLLS,
    MAX_MANIFEST_BYTES,
    CudaBundleError,
    _hash_regular_file,
    build_cuda_manifest,
    canonical_json_bytes,
    create_runner_manifest,
    load_cuda_bundle,
    verify_runner_manifest,
)

BUILD_GUARD_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "windows_build_guard.py"
)
BUILD_GUARD_SPEC = importlib.util.spec_from_file_location(
    "windows_build_guard_test_module", BUILD_GUARD_SCRIPT
)
assert BUILD_GUARD_SPEC is not None and BUILD_GUARD_SPEC.loader is not None
BUILD_GUARD = importlib.util.module_from_spec(BUILD_GUARD_SPEC)
BUILD_GUARD_SPEC.loader.exec_module(BUILD_GUARD)


def _portable_validator(path: Path, kind: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise CudaBundleError("test.path_not_absolute")
    if path.is_symlink():
        raise CudaBundleError("test.reparse")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise CudaBundleError("test.unreadable") from exc
    if kind == "file" and not stat.S_ISREG(metadata.st_mode):
        raise CudaBundleError("test.not_file")
    if kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise CudaBundleError("test.not_directory")
    return path


class WindowsCudaBundleTests(unittest.TestCase):
    def test_invalid_manifest_json_has_a_typed_error_for_both_loaders(self) -> None:
        malformed = (
            b'{"value":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}\n",
            b'{"value":"\\ud800"}\n',
            b'{"value":1e400}\n',
            b'{"value":' + b"9" * 5000 + b"}\n",
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            for loader in (load_cuda_bundle, verify_runner_manifest):
                for raw in malformed:
                    with self.subTest(loader=loader.__name__, prefix=raw[:25]):
                        manifest.write_bytes(raw)
                        with self.assertRaisesRegex(
                            CudaBundleError, "cuda_bundle.manifest_json_invalid"
                        ):
                            loader(manifest, path_validator=_portable_validator)

    def test_manifest_read_is_bounded_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            manifest.write_bytes(b"{}\n")
            stream = io.BytesIO(b" " * (MAX_MANIFEST_BYTES + 100))
            read = Mock(wraps=stream.read)
            stream.read = read
            with patch.object(Path, "open", return_value=stream):
                self.assert_code("cuda_bundle.manifest_size_invalid", manifest)
            self.assertEqual(read.call_args.args, (MAX_MANIFEST_BYTES + 1,))

    def test_hash_rejects_same_size_mutation_during_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.dll"
            path.write_bytes(b"original")
            before = path.stat()
            digest = hashlib.sha256()

            def mutate(chunk: bytes) -> None:
                digest.update(chunk)
                path.write_bytes(b"modified")
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 10**9))

            wrapped = Mock(wraps=digest)
            wrapped.update.side_effect = mutate
            with patch(
                "jarvis_voice.windows.cuda_bundle.hashlib.sha256", return_value=wrapped
            ):
                with self.assertRaisesRegex(
                    CudaBundleError, "cuda_bundle.file_changed_during_hash"
                ):
                    _hash_regular_file(path, len(b"original"))

    def test_hash_stops_reading_a_continuously_growing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.dll"
            path.write_bytes(b"original")
            consumed = 0
            digest = Mock()

            def grow(chunk: bytes) -> None:
                nonlocal consumed
                consumed += len(chunk)
                self.assertLessEqual(consumed, len(b"original") + 1)
                with path.open("ab") as handle:
                    handle.write(b"more")

            digest.update.side_effect = grow
            with patch(
                "jarvis_voice.windows.cuda_bundle.hashlib.sha256", return_value=digest
            ):
                with self.assertRaisesRegex(
                    CudaBundleError, "cuda_bundle.file_changed_during_hash"
                ):
                    _hash_regular_file(path, len(b"original"))

    def test_hash_allows_windows_stat_executable_permission_bits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runner.exe"
            payload = b"runner"
            path.write_bytes(payload)
            metadata = path.stat()
            by_path = SimpleNamespace(
                **{
                    name: getattr(metadata, name)
                    for name in (
                        "st_dev",
                        "st_ino",
                        "st_mode",
                        "st_size",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                }
            )
            # Windows stat adds execute bits by suffix; fstat has no filename.
            by_path.st_mode |= 0o111
            with patch.object(Path, "stat", return_value=by_path):
                self.assertEqual(
                    _hash_regular_file(path, len(payload)),
                    hashlib.sha256(payload).hexdigest(),
                )

    def _make_bundle(self, parent: Path) -> tuple[Path, Path, dict[str, object]]:
        root = parent / "runtime"
        root.mkdir()
        files: list[dict[str, object]] = []
        for index, name in enumerate(sorted(REQUIRED_CUDA_DLLS), start=1):
            payload = (name + ":" + str(index)).encode("ascii")
            (root / name).write_bytes(payload)
            files.append(
                {
                    "name": name,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        document: dict[str, object] = {
            "schema": CUDA_BUNDLE_SCHEMA,
            "root": str(root),
            "files": files,
        }
        manifest = parent / "cuda-runtime-manifest.json"
        manifest.write_bytes(canonical_json_bytes(document))
        return root, manifest, document

    def assert_code(self, expected: str, manifest: Path) -> None:
        with self.assertRaises(CudaBundleError) as raised:
            load_cuda_bundle(manifest, path_validator=_portable_validator)
        self.assertEqual(raised.exception.code, expected)

    def test_exact_canonical_bundle_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, _document = self._make_bundle(Path(temporary))
            bundle = load_cuda_bundle(manifest, path_validator=_portable_validator)
            self.assertEqual(bundle.root, root)
            self.assertEqual(
                {item.name.casefold() for item in bundle.files}, REQUIRED_CUDA_DLLS
            )
            self.assertEqual(
                bundle.manifest_sha256,
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
            generated = build_cuda_manifest(
                root,
                path_validator=_portable_validator,
            )
            self.assertEqual(generated, _document)

    def test_missing_required_and_extra_directory_entries_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, document = self._make_bundle(Path(temporary))
            removed = document["files"].pop()
            (root / removed["name"]).unlink()
            manifest.write_bytes(canonical_json_bytes(document))
            self.assert_code("cuda_bundle.required_dll_missing", manifest)

        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, _document = self._make_bundle(Path(temporary))
            (root / "undeclared.dll").write_bytes(b"x")
            self.assert_code("cuda_bundle.directory_not_exact", manifest)

    def test_hash_noncanonical_duplicate_and_reparse_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, document = self._make_bundle(Path(temporary))
            (root / document["files"][0]["name"]).write_bytes(b"changed")
            self.assert_code("cuda_bundle.file_size_mismatch", manifest)

        with tempfile.TemporaryDirectory() as temporary:
            _root, manifest, document = self._make_bundle(Path(temporary))
            manifest.write_text(json.dumps(document, indent=2), encoding="utf-8")
            self.assert_code("cuda_bundle.manifest_not_canonical", manifest)

        with tempfile.TemporaryDirectory() as temporary:
            _root, manifest, _document = self._make_bundle(Path(temporary))
            manifest.write_bytes(
                b'{"files":[],"root":"x","schema":"one","schema":"two"}\n'
            )
            self.assert_code("cuda_bundle.duplicate_json_key", manifest)

        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, _document = self._make_bundle(Path(temporary))
            target = root / min(REQUIRED_CUDA_DLLS)
            payload = target.read_bytes()
            target.unlink()
            outside = Path(temporary) / "outside.dll"
            outside.write_bytes(payload)
            try:
                target.symlink_to(outside)
            except (OSError, NotImplementedError):
                return
            self.assert_code("test.reparse", manifest)

    def test_driver_dll_and_manifest_inside_bundle_are_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, document = self._make_bundle(Path(temporary))
            payload = b"driver"
            (root / "nvcuda.dll").write_bytes(payload)
            document["files"].append(
                {
                    "name": "nvcuda.dll",
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            document["files"].sort(key=lambda item: item["name"].casefold())
            manifest.write_bytes(canonical_json_bytes(document))
            self.assert_code("cuda_bundle.driver_dll_forbidden", manifest)

        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, document = self._make_bundle(Path(temporary))
            payload = b"shadow"
            (root / "version.dll").write_bytes(payload)
            document["files"].append(
                {
                    "name": "version.dll",
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            document["files"].sort(key=lambda item: item["name"].casefold())
            manifest.write_bytes(canonical_json_bytes(document))
            self.assert_code("cuda_bundle.dll_name_not_allowed", manifest)

        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, document = self._make_bundle(Path(temporary))
            inside = root / "manifest.json"
            inside.write_bytes(canonical_json_bytes(document))
            self.assert_code("cuda_bundle.manifest_inside_bundle", inside)

    def test_runner_seal_is_canonical_and_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, name in enumerate(QUALIFICATION_RUNNERS, start=1):
                (root / name).write_bytes((name * index).encode("ascii"))
            payload = create_runner_manifest(
                root,
                "a" * 64,
                path_validator=_portable_validator,
            )
            manifest = root / "qualification-runners-manifest.json"
            manifest.write_bytes(canonical_json_bytes(payload))
            verified = verify_runner_manifest(
                manifest,
                path_validator=_portable_validator,
            )
            self.assertEqual(verified, payload)
            (root / QUALIFICATION_RUNNERS[0]).write_bytes(b"mutated")
            with self.assertRaisesRegex(
                CudaBundleError, "runner_bundle.file_hash_mismatch"
            ):
                verify_runner_manifest(manifest, path_validator=_portable_validator)

    def test_boot_smoke_uses_job_api_and_only_derived_environment(self) -> None:
        runner = Path("/sealed/cyrus-windows-asr-collector.exe")
        temporary = Path("/private/runtime")
        with (
            patch.object(
                BUILD_GUARD, "validate_private_temp_directory", return_value=temporary
            ),
            patch.object(
                BUILD_GUARD,
                "trusted_windows_directory",
                return_value=Path("/native/windows"),
            ),
            patch.object(
                BUILD_GUARD,
                "private_temp_environment",
                return_value={"TEMP": str(temporary), "TMP": str(temporary)},
            ),
            patch.object(
                BUILD_GUARD, "require_native_windows_path", return_value=runner
            ),
            patch.object(
                BUILD_GUARD,
                "run_windows_job",
                return_value=subprocess.CompletedProcess([str(runner)], 2),
            ) as run,
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                result = BUILD_GUARD.main(
                    [
                        "boot-smoke",
                        "--runner",
                        str(runner),
                        "--temporary-directory",
                        str(temporary),
                    ]
                )
        self.assertEqual(result, 0)
        self.assertTrue(json.loads(output.getvalue())["ready"])
        environment = run.call_args.kwargs["environment"]
        self.assertEqual(
            environment,
            {
                "SystemRoot": str(Path("/native/windows")),
                "WINDIR": str(Path("/native/windows")),
                "OS": "Windows_NT",
                "TEMP": str(temporary),
                "TMP": str(temporary),
            },
        )
        self.assertNotIn("PATH", environment)
        self.assertNotIn("HTTP_PROXY", environment)

    def test_boot_smoke_native_failure_is_content_free(self) -> None:
        runner = Path("/sealed/cyrus-windows-asr-collector.exe")
        temporary = Path("/private/runtime")
        with (
            patch.object(
                BUILD_GUARD, "validate_private_temp_directory", return_value=temporary
            ),
            patch.object(
                BUILD_GUARD,
                "trusted_windows_directory",
                return_value=Path("/native/windows"),
            ),
            patch.object(
                BUILD_GUARD,
                "private_temp_environment",
                return_value={"TEMP": str(temporary)},
            ),
            patch.object(
                BUILD_GUARD, "require_native_windows_path", return_value=runner
            ),
            patch.object(
                BUILD_GUARD,
                "run_windows_job",
                side_effect=WindowsProcessFailure("private path must not escape"),
            ),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                result = BUILD_GUARD.main(
                    [
                        "boot-smoke",
                        "--runner",
                        str(runner),
                        "--temporary-directory",
                        str(temporary),
                    ]
                )
        self.assertEqual(result, 2)
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "code": "runner_bundle.boot_smoke_failed",
                "ready": False,
                "schema": "cyrus.windows-build-guard-result/1",
            },
        )
        self.assertNotIn("private path", output.getvalue())

    def test_boot_smoke_path_failure_uses_the_same_stable_code(self) -> None:
        temporary = Path("/private/runtime")
        with (
            patch.object(
                BUILD_GUARD,
                "validate_private_temp_directory",
                return_value=temporary,
            ),
            patch.object(
                BUILD_GUARD,
                "trusted_windows_directory",
                return_value=Path("/native/windows"),
            ),
            patch.object(
                BUILD_GUARD,
                "private_temp_environment",
                return_value={"TEMP": str(temporary)},
            ),
            patch.object(
                BUILD_GUARD,
                "require_native_windows_path",
                side_effect=CudaBundleError("cuda_bundle.private_path"),
            ),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                result = BUILD_GUARD.main(
                    [
                        "boot-smoke",
                        "--runner",
                        "/private/runner.exe",
                        "--temporary-directory",
                        str(temporary),
                    ]
                )
        self.assertEqual(result, 2)
        self.assertEqual(
            json.loads(output.getvalue())["code"],
            "runner_bundle.boot_smoke_failed",
        )
        self.assertNotIn("private_path", output.getvalue())


if __name__ == "__main__":
    unittest.main()
