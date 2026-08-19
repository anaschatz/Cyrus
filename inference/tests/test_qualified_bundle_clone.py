from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from jarvis_inference.strict_json import canonical_bytes
from jarvis_inference.gateway import effective_configuration
from jarvis_inference.qualification import SUITE_VERSION


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "clone_qualified_macos_bundle.py"
)
SPEC = importlib.util.spec_from_file_location("qualified_bundle_clone", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
clone_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = clone_module
SPEC.loader.exec_module(clone_module)


def _digest(path: Path) -> tuple[int, str]:
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def _artifact(root: Path, path: Path) -> dict[str, object]:
    size, digest = _digest(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }


def _record_hash(raw: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


class QualifiedBundleFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.site = root / "python/lib/python3.11/site-packages"
        self.gateway = self.site / "jarvis_inference"
        self.gateway_dist = self.site / "jarvis_inference-1.0.0.dist-info"
        self.sdk = self.site / "openai"
        self.sdk_dist = self.site / "openai-2.46.0.dist-info"
        self._build()

    @staticmethod
    def _write(path: Path, raw: bytes, *, executable: bool = False) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o700 if executable else 0o600)
        return path

    def _wheel(
        self,
        name: str,
        version: str,
        filename: str,
        *,
        members: dict[str, bytes] | None = None,
    ) -> Path:
        path = self.root / "wheels" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        dist = name.replace("-", "_") + f"-{version}.dist-info"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member, raw in sorted((members or {}).items()):
                archive.writestr(member, raw)
            archive.writestr(
                f"{dist}/METADATA",
                f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            )
            archive.writestr(f"{dist}/WHEEL", "Wheel-Version: 1.0\n")
            archive.writestr(f"{dist}/RECORD", "")
        return path

    def _distribution(
        self,
        package: Path,
        dist_info: Path,
        name: str,
        version: str,
        package_files: dict[str, bytes],
    ) -> None:
        for filename, raw in package_files.items():
            self._write(package / filename, raw)
        metadata = self._write(
            dist_info / "METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode(),
        )
        record = dist_info / "RECORD"
        rows: list[list[str]] = []
        for path in sorted([*package.iterdir(), metadata]):
            relative = path.relative_to(self.site).as_posix()
            raw = path.read_bytes()
            rows.append([relative, _record_hash(raw), str(len(raw))])
        rows.append([record.relative_to(self.site).as_posix(), "", ""])
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerows(rows)
        self._write(record, stream.getvalue().encode())

    def _gateway_digest(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(self.gateway.glob("*.py"), key=lambda item: item.name):
            digest.update(path.name.encode() + b"\0")
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _build(self) -> None:
        executable = self._write(
            self.root / "python/bin/python3.11", b"sealed-python\n", executable=True
        )
        self._write(self.root / "python/pyvenv.cfg", b"sealed = true\n")
        worker = self._write(
            self.root / "runtime/mlx_worker_runtime.py", b"WORKER = True\n"
        )
        model = self._write(
            self.root / "models/qwen/config.json", b'{"model":"fixture"}\n'
        )
        fixture_runner = self._write(
            self.root / "fixtures/qualification_runner.py", b"RUNNER = True\n"
        )
        self._write(self.root / "README.txt", b"ordinary full-bundle file\n")
        gateway_sources = {
            "__init__.py": b'__version__ = "1.0.0"\n',
            "gateway.py": b"GATEWAY = True\n",
        }
        sdk_sources = {"__init__.py": b'__version__ = "2.46.0"\n'}
        self._distribution(
            self.gateway,
            self.gateway_dist,
            "jarvis-inference",
            "1.0.0",
            gateway_sources,
        )
        self._distribution(
            self.sdk,
            self.sdk_dist,
            "openai",
            "2.46.0",
            sdk_sources,
        )
        gateway_wheel = self._wheel(
            "jarvis-inference",
            "1.0.0",
            "jarvis_inference-1.0.0-py3-none-any.whl",
            members={
                f"jarvis_inference/{filename}": raw
                for filename, raw in gateway_sources.items()
            },
        )
        sdk_wheel = self._wheel(
            "openai",
            "2.46.0",
            "openai-2.46.0-py3-none-any.whl",
            members={
                f"openai/{filename}": raw
                for filename, raw in sdk_sources.items()
            },
        )
        gateway_wheel_artifact = _artifact(self.root, gateway_wheel)
        manifest = {
            "alias": "jarvis-local",
            "backend": {"type": "mlx-worker", "worker": _artifact(self.root, worker)},
            "capabilities": {
                "json_object": True,
                "json_schema": True,
                "stream": True,
                "text": True,
                "tools": True,
            },
            "manifest_schema_version": 1,
            "model": {
                "artifacts": [_artifact(self.root, model)],
                "context_window": 4096,
                "format": "mlx",
            },
            "platform": "macos-arm64",
            "python": {
                "executable": _artifact(self.root, executable),
                "implementation": "CPython",
                "packages": [
                    {
                        "name": "jarvis-inference",
                        "version": "1.0.0",
                        "wheel": gateway_wheel_artifact,
                        "wheel_filename": gateway_wheel.name,
                    },
                    {
                        "name": "openai",
                        "version": "2.46.0",
                        "wheel": _artifact(self.root, sdk_wheel),
                        "wheel_filename": sdk_wheel.name,
                    },
                ],
                "site_packages": "python/lib/python3.11/site-packages",
                "version": "3.11.9",
            },
            "qualification_report": "qualification-report.json",
        }
        (self.root / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        identity = {
            "architecture": "arm64",
            "cpu": "fixture-apple-silicon",
            "effective_configuration": dict(
                clone_module._EXPECTED_EFFECTIVE_CONFIGURATION
            ),
            "gateway_digest": self._gateway_digest(),
            "gateway_runtime": {
                "openai_sdk_version": "2.46.0",
                "version": "1.0.0",
                "wheel": gateway_wheel_artifact,
            },
            "hardware_runtime": {
                "compute_capability": "metal3",
                "compute_runtime": "mlx-0.31.3",
                "driver": "fixture-driver",
                "gpu_device": "Apple fixture",
                "gpu_memory_bytes": 16 * 1024**3,
                "ram_bytes": 16 * 1024**3,
            },
            "manifest_digest": hashlib.sha256(canonical_bytes(manifest)).hexdigest(),
            "os": {
                "release": "fixture-release",
                "system": "Darwin",
                "version": "fixture-version",
            },
            "python": {"implementation": "CPython", "version": "3.11.9"},
        }
        report = {
            "completed": 2,
            "deterministic_seeds": [1],
            "evidence_digest": "a" * 64,
            "fixture_digest": "b" * 64,
            "fixture_runner": _artifact(self.root, fixture_runner),
            "identity": identity,
            "provenance": {
                "challenge": "c" * 64,
                "completed_unix_ns": "2",
                "fixture_digest": "b" * 64,
                "identity_digest": hashlib.sha256(
                    canonical_bytes(identity)
                ).hexdigest(),
                "monotonic_elapsed_ns": 1,
                "started_unix_ns": "1",
            },
            "report_schema_version": 1,
            "results": {},
            "started": 1,
            "status": "pass",
            "suite_version": clone_module.QUALIFICATION_SUITE_VERSION,
        }
        (self.root / "qualification-report.json").write_text(
            json.dumps(report, sort_keys=True), encoding="utf-8"
        )

    def _refresh_manifest_report(self, manifest: dict[str, object]) -> None:
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        report_path = self.root / "qualification-report.json"
        report = json.loads(report_path.read_text())
        report["identity"]["manifest_digest"] = hashlib.sha256(
            canonical_bytes(manifest)
        ).hexdigest()
        report["provenance"]["identity_digest"] = hashlib.sha256(
            canonical_bytes(report["identity"])
        ).hexdigest()
        report_path.write_text(
            json.dumps(report, sort_keys=True), encoding="utf-8"
        )

    def add_shared_mlx_owners(
        self,
        *,
        second_name: str = "mlx-metal",
        first_version: str = "0.32.0",
        second_version: str = "0.32.0",
        installed_members: dict[str, bytes] | None = None,
        first_wheel_members: dict[str, bytes] | None = None,
        second_wheel_members: dict[str, bytes] | None = None,
    ) -> dict[str, Path]:
        default_installed = {
            PurePosixPath(member).name: f"fixture:{member}\n".encode()
            for member in clone_module._MLX_SHARED_WHEEL_MEMBERS
        }
        installed = dict(
            default_installed if installed_members is None else installed_members
        )
        default_wheel = {
            f"mlx/{filename}": raw for filename, raw in installed.items()
        }
        first_wheel = dict(
            default_wheel
            if first_wheel_members is None
            else first_wheel_members
        )
        second_wheel = dict(
            default_wheel
            if second_wheel_members is None
            else second_wheel_members
        )
        package = self.site / "mlx"
        first_dist = self.site / f"mlx-{first_version}.dist-info"
        second_slug = second_name.replace("-", "_")
        second_dist = self.site / f"{second_slug}-{second_version}.dist-info"
        self._distribution(
            package,
            first_dist,
            "mlx",
            first_version,
            installed,
        )
        self._distribution(
            package,
            second_dist,
            second_name,
            second_version,
            installed,
        )
        first_wheel_path = self._wheel(
            "mlx",
            first_version,
            f"mlx-{first_version}-py3-none-any.whl",
            members=first_wheel,
        )
        second_wheel_path = self._wheel(
            second_name,
            second_version,
            f"{second_slug}-{second_version}-py3-none-any.whl",
            members=second_wheel,
        )
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["python"]["packages"].extend(
            [
                {
                    "name": "mlx",
                    "version": first_version,
                    "wheel": _artifact(self.root, first_wheel_path),
                    "wheel_filename": first_wheel_path.name,
                },
                {
                    "name": second_name,
                    "version": second_version,
                    "wheel": _artifact(self.root, second_wheel_path),
                    "wheel_filename": second_wheel_path.name,
                },
            ]
        )
        manifest["python"]["packages"].sort(key=lambda value: value["name"])
        self._refresh_manifest_report(manifest)
        return {"mlx": first_dist, second_name: second_dist}

    @staticmethod
    def rewrite_record_row(
        dist_info: Path,
        row_path: str,
        *,
        encoded_digest: str | None = None,
        encoded_size: str | None = None,
    ) -> None:
        record = dist_info / "RECORD"
        rows = list(csv.reader(io.StringIO(record.read_text()), strict=True))
        changed = False
        for row in rows:
            if row[0] == row_path:
                if encoded_digest is not None:
                    row[1] = encoded_digest
                if encoded_size is not None:
                    row[2] = encoded_size
                changed = True
        if not changed:
            raise AssertionError("fixture RECORD row not found")
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerows(rows)
        record.write_text(stream.getvalue(), encoding="utf-8")

    def add_python_cache(
        self,
        *,
        source_relative: str = "jarvis_inference/gateway.py",
        cache_name: str | None = None,
        raw: bytes = b"synthetic generated bytecode\n",
    ) -> Path:
        source = self.site / source_relative
        selected_name = cache_name or (
            f"{source.stem}.cpython-311.pyc"
        )
        return self._write(source.parent / "__pycache__" / selected_name, raw)

    def record_file(self, dist_info: Path, path: Path) -> None:
        record = dist_info / "RECORD"
        rows = list(csv.reader(io.StringIO(record.read_text()), strict=True))
        raw = path.read_bytes()
        rows.insert(
            -1,
            [
                path.relative_to(self.site).as_posix(),
                _record_hash(raw),
                str(len(raw)),
            ],
        )
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerows(rows)
        record.write_text(stream.getvalue(), encoding="utf-8")

    def set_report_gateway_wheel(self, descriptor: dict[str, object]) -> None:
        report_path = self.root / "qualification-report.json"
        report = json.loads(report_path.read_text())
        report["identity"]["gateway_runtime"]["wheel"] = descriptor
        report["provenance"]["identity_digest"] = hashlib.sha256(
            canonical_bytes(report["identity"])
        ).hexdigest()
        report_path.write_text(
            json.dumps(report, sort_keys=True), encoding="utf-8"
        )

    def add_gateway_report_wheel_alias(
        self,
        *,
        relative: str = "dist/jarvis_inference-1.0.0-py3-none-any.whl",
        raw: bytes | None = None,
    ) -> Path:
        manifest = json.loads((self.root / "manifest.json").read_text())
        package = next(
            item
            for item in manifest["python"]["packages"]
            if item["name"] == "jarvis-inference"
        )
        source = self.root / package["wheel"]["path"]
        alias = self._write(
            self.root / relative,
            source.read_bytes() if raw is None else raw,
        )
        self.set_report_gateway_wheel(_artifact(self.root, alias))
        return alias

    def add_sanitizable_contamination(self) -> None:
        self._write(self.root / ".DS_Store", b"finder-root\n")
        self._write(self.site / ".DS_Store", b"finder-site\n")
        self._write(self.gateway_dist / "PKG-INFO 2", b"conflict\n")
        self._write(self.root / "python/readme 2.txt", b"conflict\n")
        self._write(
            self.site / "openai-2.46.0 2.dist-info/METADATA",
            b"Name: openai\nVersion: 2.46.0\n",
        )

    def add_standard_python_convenience_links(self) -> None:
        for relative, target in clone_module._PYTHON_CONVENIENCE_LINKS.items():
            link = self.root / relative
            target_path = link.parent / target
            if not target_path.exists():
                self._write(
                    target_path,
                    f"target:{relative}\n".encode(),
                    executable=relative.startswith("python/bin/"),
                )
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target)

    def record_convenience_link(self, relative: str) -> None:
        link = self.root / relative
        record = self.gateway_dist / "RECORD"
        rows = list(csv.reader(io.StringIO(record.read_text(), newline="")))
        record_path = os.path.relpath(link, self.site).replace(os.sep, "/")
        raw = link.read_bytes()
        rows.insert(-1, [record_path, _record_hash(raw), str(len(raw))])
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerows(rows)
        record.write_text(stream.getvalue(), encoding="utf-8")

    def add_recorded_conflict(self) -> Path:
        conflict = self._write(self.gateway_dist / "PKG-INFO 2", b"recorded\n")
        record = self.gateway_dist / "RECORD"
        rows = list(csv.reader(io.StringIO(record.read_text(), newline="")))
        relative = conflict.relative_to(self.site).as_posix()
        raw = conflict.read_bytes()
        rows.insert(-1, [relative, _record_hash(raw), str(len(raw))])
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerows(rows)
        record.write_text(stream.getvalue(), encoding="utf-8")
        return conflict

    def refresh_gateway_identity(self) -> None:
        report_path = self.root / "qualification-report.json"
        report = json.loads(report_path.read_text())
        report["identity"]["gateway_digest"] = self._gateway_digest()
        report["provenance"]["identity_digest"] = hashlib.sha256(
            canonical_bytes(report["identity"])
        ).hexdigest()
        report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    def source_hashes(self) -> dict[str, str]:
        return {
            path.relative_to(self.root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in self.root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }


class QualifiedBundleCloneTests(unittest.TestCase):
    def fixture(self, parent: Path, name: str = "source") -> QualifiedBundleFixture:
        root = parent / name
        root.mkdir()
        return QualifiedBundleFixture(root)

    def test_clone_contract_tracks_qualified_gateway_contract(self) -> None:
        self.assertEqual(
            clone_module._EXPECTED_EFFECTIVE_CONFIGURATION,
            effective_configuration(),
        )
        self.assertEqual(clone_module.QUALIFICATION_SUITE_VERSION, SUITE_VERSION)

    def test_clean_clone_excludes_only_unrecorded_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            fixture = self.fixture(parent)
            fixture.add_standard_python_convenience_links()
            fixture.add_sanitizable_contamination()
            before = fixture.source_hashes()
            destination = parent / "destination"

            report = clone_module.clone_qualified_bundle(
                fixture.root, destination
            )

            self.assertTrue(report["ready"])
            self.assertTrue(report["qualification_preserved"])
            self.assertFalse(report["requalification_required"])
            self.assertEqual(
                report["excluded"],
                {
                    "conflict_copy_files": 3,
                    "finder_metadata_files": 2,
                    "generated_python_cache_files": 0,
                    "python_runtime_convenience_links": 9,
                },
            )
            self.assertEqual(before, fixture.source_hashes())
            self.assertTrue((fixture.root / ".DS_Store").is_file())
            self.assertFalse((destination / ".DS_Store").exists())
            self.assertFalse((destination / "python/readme 2.txt").exists())
            self.assertFalse(
                (destination / "python/lib/python3.11/site-packages/.DS_Store").exists()
            )
            self.assertTrue((destination / "README.txt").is_file())
            for relative, target in clone_module._PYTHON_CONVENIENCE_LINKS.items():
                self.assertTrue(os.path.lexists(fixture.root / relative))
                self.assertEqual(os.readlink(fixture.root / relative), target)
                self.assertFalse(os.path.lexists(destination / relative))
                self.assertTrue(
                    (destination / relative).parent.joinpath(target).is_file()
                )
            self.assertEqual(
                (destination / "manifest.json").read_bytes(),
                (fixture.root / "manifest.json").read_bytes(),
            )
            self.assertEqual(
                (destination / "qualification-report.json").read_bytes(),
                (fixture.root / "qualification-report.json").read_bytes(),
            )
            destination_plan = clone_module._inspect_bundle(
                destination,
                "manifest.json",
                allow_sanitizable_contamination=False,
            )
            self.assertFalse(destination_plan.excluded_files)

    def test_gateway_wheel_path_alias_preserves_both_identity_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            fixture = self.fixture(parent)
            alias = fixture.add_gateway_report_wheel_alias()
            manifest_before = (fixture.root / "manifest.json").read_bytes()
            report_before = (
                fixture.root / "qualification-report.json"
            ).read_bytes()
            destination = parent / "destination"

            report = clone_module.clone_qualified_bundle(
                fixture.root, destination
            )

            self.assertTrue(report["ready"])
            manifest = json.loads(manifest_before)
            manifest_wheel = next(
                item["wheel"]
                for item in manifest["python"]["packages"]
                if item["name"] == "jarvis-inference"
            )
            manifest_wheel_path = destination / manifest_wheel["path"]
            alias_path = destination / alias.relative_to(fixture.root)
            self.assertNotEqual(alias_path, manifest_wheel_path)
            self.assertEqual(alias_path.read_bytes(), manifest_wheel_path.read_bytes())
            self.assertEqual(
                (destination / "manifest.json").read_bytes(), manifest_before
            )
            self.assertEqual(
                (destination / "qualification-report.json").read_bytes(),
                report_before,
            )

    def test_gateway_wheel_path_alias_fails_closed_on_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()

            def assert_invalid(
                fixture: QualifiedBundleFixture,
                destination_name: str,
            ) -> None:
                destination = parent / destination_name
                with self.assertRaises(clone_module.BundleCloneError) as caught:
                    clone_module.clone_qualified_bundle(
                        fixture.root, destination
                    )
                self.assertEqual(
                    caught.exception.code,
                    "bundle_gateway_identity_invalid",
                )
                self.assertFalse(destination.exists())

            renamed = self.fixture(parent, "renamed")
            renamed.add_gateway_report_wheel_alias(
                relative="dist/renamed-inference.whl"
            )
            assert_invalid(renamed, "renamed-clone")

            changed = self.fixture(parent, "changed")
            manifest = json.loads((changed.root / "manifest.json").read_text())
            package = next(
                item
                for item in manifest["python"]["packages"]
                if item["name"] == "jarvis-inference"
            )
            wheel_raw = (changed.root / package["wheel"]["path"]).read_bytes()
            changed_raw = bytes([wheel_raw[0] ^ 1]) + wheel_raw[1:]
            changed.add_gateway_report_wheel_alias(raw=changed_raw)
            assert_invalid(changed, "changed-clone")

            missing = self.fixture(parent, "missing")
            manifest = json.loads((missing.root / "manifest.json").read_text())
            package = next(
                item
                for item in manifest["python"]["packages"]
                if item["name"] == "jarvis-inference"
            )
            missing_descriptor = dict(package["wheel"])
            missing_descriptor["path"] = (
                "dist/jarvis_inference-1.0.0-py3-none-any.whl"
            )
            missing.set_report_gateway_wheel(missing_descriptor)
            assert_invalid(missing, "missing-clone")

            outside = self.fixture(parent, "outside")
            manifest = json.loads((outside.root / "manifest.json").read_text())
            package = next(
                item
                for item in manifest["python"]["packages"]
                if item["name"] == "jarvis-inference"
            )
            outside_descriptor = dict(package["wheel"])
            outside_descriptor["path"] = (
                "../jarvis_inference-1.0.0-py3-none-any.whl"
            )
            outside.set_report_gateway_wheel(outside_descriptor)
            assert_invalid(outside, "outside-clone")

            omitted = self.fixture(parent, "omitted")
            omitted.add_gateway_report_wheel_alias(
                relative=(
                    "dist 2/jarvis_inference-1.0.0-py3-none-any.whl"
                )
            )
            assert_invalid(omitted, "omitted-clone")

            byte_gate = self.fixture(parent, "byte-gate")
            byte_gate.add_gateway_report_wheel_alias()
            with patch.object(
                clone_module,
                "_artifact_bytes_equal",
                return_value=False,
            ):
                assert_invalid(byte_gate, "byte-gate-clone")

    def test_exact_mlx_shared_ownership_is_preserved_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            fixture = self.fixture(parent)
            fixture.add_shared_mlx_owners()
            before = fixture.source_hashes()
            destination = parent / "destination"

            report = clone_module.clone_qualified_bundle(
                fixture.root, destination
            )

            self.assertTrue(report["ready"])
            self.assertTrue(report["qualification_preserved"])
            self.assertFalse(report["requalification_required"])
            self.assertEqual(report["package_count"], 4)
            self.assertEqual(report["record_file_count"], 16)
            self.assertEqual(before, fixture.source_hashes())
            for member in clone_module._MLX_SHARED_WHEEL_MEMBERS:
                source = fixture.site / member
                copied = (
                    destination
                    / "python/lib/python3.11/site-packages"
                    / member
                )
                self.assertEqual(copied.read_bytes(), source.read_bytes())
            destination_plan = clone_module._inspect_bundle(
                destination,
                "manifest.json",
                allow_sanitizable_contamination=False,
            )
            self.assertFalse(destination_plan.excluded_files)

    def test_mlx_shared_ownership_rejects_every_non_exact_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()

            arbitrary_pair = self.fixture(parent, "arbitrary-pair")
            arbitrary_pair.add_shared_mlx_owners(second_name="mlx-runtime")
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    arbitrary_pair.root,
                    parent / "arbitrary-pair-clone",
                )
            self.assertEqual(caught.exception.code, "bundle_record_duplicate_path")

            default_installed = {
                PurePosixPath(member).name: f"fixture:{member}\n".encode()
                for member in clone_module._MLX_SHARED_WHEEL_MEMBERS
            }
            unexpected_path = self.fixture(parent, "unexpected-path")
            unexpected_members = {
                **default_installed,
                "unexpected.py": b"unexpected shared payload\n",
            }
            unexpected_path.add_shared_mlx_owners(
                installed_members=unexpected_members
            )
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    unexpected_path.root,
                    parent / "unexpected-path-clone",
                )
            self.assertEqual(caught.exception.code, "bundle_record_duplicate_path")

            missing_member = self.fixture(parent, "missing-wheel-member")
            missing_wheel_members = {
                f"mlx/{filename}": raw
                for filename, raw in default_installed.items()
            }
            missing_wheel_members.pop("mlx/utils.py")
            missing_member.add_shared_mlx_owners(
                second_wheel_members=missing_wheel_members
            )
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    missing_member.root,
                    parent / "missing-wheel-member-clone",
                )
            self.assertEqual(
                caught.exception.code,
                "bundle_record_shared_ownership_invalid",
            )

            mismatched_member = self.fixture(parent, "mismatched-wheel-member")
            mismatched_wheel_members = {
                f"mlx/{filename}": raw
                for filename, raw in default_installed.items()
            }
            mismatched_wheel_members["mlx/utils.py"] = b"different wheel bytes\n"
            mismatched_member.add_shared_mlx_owners(
                second_wheel_members=mismatched_wheel_members
            )
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    mismatched_member.root,
                    parent / "mismatched-wheel-member-clone",
                )
            self.assertEqual(
                caught.exception.code,
                "bundle_record_shared_ownership_invalid",
            )

            missing_size = self.fixture(parent, "missing-record-size")
            dist_infos = missing_size.add_shared_mlx_owners()
            missing_size.rewrite_record_row(
                dist_infos["mlx-metal"],
                "mlx/utils.py",
                encoded_size="",
            )
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    missing_size.root,
                    parent / "missing-record-size-clone",
                )
            self.assertEqual(
                caught.exception.code,
                "bundle_record_shared_ownership_invalid",
            )

            mismatched_digest = self.fixture(parent, "mismatched-record-digest")
            dist_infos = mismatched_digest.add_shared_mlx_owners()
            mismatched_digest.rewrite_record_row(
                dist_infos["mlx-metal"],
                "mlx/utils.py",
                encoded_digest=_record_hash(b"different installed bytes\n"),
            )
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    mismatched_digest.root,
                    parent / "mismatched-record-digest-clone",
                )
            self.assertEqual(
                caught.exception.code,
                "bundle_record_digest_mismatch",
            )

            mismatched_size = self.fixture(parent, "mismatched-record-size")
            dist_infos = mismatched_size.add_shared_mlx_owners()
            mismatched_size.rewrite_record_row(
                dist_infos["mlx-metal"],
                "mlx/utils.py",
                encoded_size="999999",
            )
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    mismatched_size.root,
                    parent / "mismatched-record-size-clone",
                )
            self.assertEqual(
                caught.exception.code,
                "bundle_record_size_mismatch",
            )

            incomplete_overlap = self.fixture(parent, "incomplete-overlap")
            incomplete_members = dict(default_installed)
            incomplete_members.pop("utils.py")
            full_wheel_members = {
                f"mlx/{filename}": raw
                for filename, raw in default_installed.items()
            }
            incomplete_overlap.add_shared_mlx_owners(
                installed_members=incomplete_members,
                first_wheel_members=full_wheel_members,
                second_wheel_members=full_wheel_members,
            )
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    incomplete_overlap.root,
                    parent / "incomplete-overlap-clone",
                )
            self.assertEqual(
                caught.exception.code,
                "bundle_record_shared_ownership_invalid",
            )

            wrong_version = self.fixture(parent, "wrong-version")
            wrong_version.add_shared_mlx_owners(second_version="0.32.1")
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    wrong_version.root,
                    parent / "wrong-version-clone",
                )
            self.assertEqual(caught.exception.code, "bundle_record_duplicate_path")

    def test_generated_cpython311_cache_is_omitted_only_inside_site_packages(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            fixture = self.fixture(parent)
            fixture.add_python_cache()
            fixture.add_python_cache(
                cache_name="gateway.cpython-311.opt-2.pyc",
                raw=b"synthetic optimized bytecode\n",
            )
            standard_library_cache = fixture._write(
                fixture.root
                / "python/lib/python3.11/__pycache__/os.cpython-311.pyc",
                b"standard-library cache remains ordinary bundle data\n",
            )
            before = fixture.source_hashes()
            destination = parent / "destination"

            report = clone_module.clone_qualified_bundle(
                fixture.root, destination
            )

            self.assertTrue(report["ready"])
            self.assertEqual(
                report["excluded"]["generated_python_cache_files"], 2
            )
            self.assertEqual(before, fixture.source_hashes())
            destination_site = (
                destination / "python/lib/python3.11/site-packages"
            )
            self.assertEqual(list(destination_site.rglob("*.pyc")), [])
            self.assertEqual(list(destination_site.rglob("__pycache__")), [])
            copied_standard_library_cache = (
                destination / standard_library_cache.relative_to(fixture.root)
            )
            self.assertEqual(
                copied_standard_library_cache.read_bytes(),
                standard_library_cache.read_bytes(),
            )

    def test_generated_cache_policy_rejects_every_unsafe_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()

            def assert_failure(
                fixture: QualifiedBundleFixture,
                destination_name: str,
                code: str,
            ) -> None:
                destination = parent / destination_name
                with self.assertRaises(clone_module.BundleCloneError) as caught:
                    clone_module.clone_qualified_bundle(
                        fixture.root, destination
                    )
                self.assertEqual(caught.exception.code, code)
                self.assertFalse(destination.exists())

            sourceless = self.fixture(parent, "sourceless")
            sourceless.add_python_cache(
                source_relative="jarvis_inference/missing.py"
            )
            assert_failure(
                sourceless,
                "sourceless-clone",
                "bundle_generated_pycache_invalid",
            )

            malformed = self.fixture(parent, "malformed")
            malformed.add_python_cache(
                cache_name="gateway.cpython-311.opt-debug.pyc"
            )
            assert_failure(
                malformed,
                "malformed-clone",
                "bundle_generated_pycache_invalid",
            )

            wrong_tag = self.fixture(parent, "wrong-tag")
            wrong_tag.add_python_cache(
                cache_name="gateway.cpython-312.pyc"
            )
            assert_failure(
                wrong_tag,
                "wrong-tag-clone",
                "bundle_generated_pycache_invalid",
            )

            recorded = self.fixture(parent, "recorded-cache")
            recorded_cache = recorded.add_python_cache()
            recorded.record_file(recorded.gateway_dist, recorded_cache)
            assert_failure(
                recorded,
                "recorded-cache-clone",
                "bundle_generated_pycache_invalid",
            )

            referenced = self.fixture(parent, "referenced-cache")
            referenced_cache = referenced.add_python_cache()
            (referenced.root / "manifest-secondary.json").write_text(
                json.dumps(
                    {"cache_path": referenced_cache.relative_to(referenced.root).as_posix()}
                ),
                encoding="utf-8",
            )
            assert_failure(
                referenced,
                "referenced-cache-clone",
                "bundle_generated_pycache_referenced",
            )

            wheel_mismatch = self.fixture(parent, "wheel-mismatch")
            wheel_mismatch.add_python_cache()
            changed_source = b"GATEWAY = 'changed after wheel install'\n"
            (wheel_mismatch.gateway / "gateway.py").write_bytes(changed_source)
            wheel_mismatch.rewrite_record_row(
                wheel_mismatch.gateway_dist,
                "jarvis_inference/gateway.py",
                encoded_digest=_record_hash(changed_source),
                encoded_size=str(len(changed_source)),
            )
            wheel_mismatch.refresh_gateway_identity()
            assert_failure(
                wheel_mismatch,
                "wheel-mismatch-clone",
                "bundle_generated_pycache_invalid",
            )

            wheel_member_missing = self.fixture(
                parent, "wheel-member-missing"
            )
            added_source = wheel_member_missing._write(
                wheel_member_missing.gateway / "added.py",
                b"ADDED_AFTER_WHEEL_INSTALL = True\n",
            )
            wheel_member_missing.record_file(
                wheel_member_missing.gateway_dist, added_source
            )
            wheel_member_missing.add_python_cache(
                source_relative="jarvis_inference/added.py",
                cache_name="added.cpython-311.pyc",
            )
            wheel_member_missing.refresh_gateway_identity()
            assert_failure(
                wheel_member_missing,
                "wheel-member-missing-clone",
                "bundle_generated_pycache_invalid",
            )

            linked = self.fixture(parent, "linked-cache")
            linked_cache = (
                linked.gateway
                / "__pycache__/gateway.cpython-311.pyc"
            )
            linked_cache.parent.mkdir(parents=True)
            linked_cache.symlink_to(linked.gateway / "gateway.py")
            assert_failure(
                linked,
                "linked-cache-clone",
                "bundle_source_link_forbidden",
            )

    def test_python_convenience_link_policy_rejects_every_unsafe_variant(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()

            absolute = self.fixture(parent, "absolute")
            absolute.add_standard_python_convenience_links()
            absolute_link = absolute.root / "python/bin/python"
            absolute_link.unlink()
            absolute_link.symlink_to(
                (absolute.root / "python/bin/python3.11").resolve()
            )
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    absolute.root, parent / "absolute-clone"
                )
            self.assertEqual(
                caught.exception.code,
                "bundle_python_convenience_link_invalid",
            )

            escaping = self.fixture(parent, "escaping")
            escaping.add_standard_python_convenience_links()
            escaping_link = escaping.root / "python/bin/python"
            escaping_link.unlink()
            escaping_link.symlink_to("../../outside-python")
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    escaping.root, parent / "escaping-clone"
                )
            self.assertEqual(
                caught.exception.code,
                "bundle_python_convenience_link_invalid",
            )

            dangling = self.fixture(parent, "dangling")
            dangling.add_standard_python_convenience_links()
            (dangling.root / "python/bin/pydoc3.11").unlink()
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    dangling.root, parent / "dangling-clone"
                )
            self.assertEqual(
                caught.exception.code,
                "bundle_python_convenience_link_invalid",
            )

            recorded = self.fixture(parent, "recorded-link")
            recorded.add_standard_python_convenience_links()
            recorded.record_convenience_link("python/bin/2to3")
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    recorded.root, parent / "recorded-link-clone"
                )
            self.assertEqual(
                caught.exception.code,
                "bundle_omitted_link_referenced",
            )

            other_profile = self.fixture(parent, "other-profile-link")
            other_profile.add_standard_python_convenience_links()
            (other_profile.root / "manifest-secondary.json").write_text(
                json.dumps({"runtime_alias": "python/bin/python"}),
                encoding="utf-8",
            )
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    other_profile.root,
                    parent / "other-profile-link-clone",
                )
            self.assertEqual(
                caught.exception.code,
                "bundle_omitted_link_referenced",
            )

            identity_bound = self.fixture(parent, "identity-bound-link")
            identity_bound.add_standard_python_convenience_links()
            manifest_path = identity_bound.root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["python"]["executable"] = _artifact(
                identity_bound.root,
                identity_bound.root / "python/bin/python",
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    identity_bound.root,
                    parent / "identity-bound-link-clone",
                )
            self.assertEqual(caught.exception.code, "bundle_manifest_invalid")

            non_whitelisted = self.fixture(parent, "non-whitelisted-link")
            (non_whitelisted.root / "python/bin/pip3").symlink_to("python3.11")
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    non_whitelisted.root,
                    parent / "non-whitelisted-link-clone",
                )
            self.assertEqual(caught.exception.code, "bundle_source_link_forbidden")

    def test_recorded_conflict_is_preserved_and_identity_bound_conflict_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            recorded = self.fixture(parent, "recorded")
            conflict = recorded.add_recorded_conflict()
            destination = parent / "recorded-clone"
            report = clone_module.clone_qualified_bundle(
                recorded.root, destination
            )
            self.assertTrue(report["ready"])
            self.assertEqual(report["excluded"]["conflict_copy_files"], 0)
            self.assertEqual(
                (destination / conflict.relative_to(recorded.root)).read_bytes(),
                conflict.read_bytes(),
            )

            bound = self.fixture(parent, "identity-bound")
            bound._write(bound.gateway / "gateway 2.py", b"BOUND = True\n")
            bound.refresh_gateway_identity()
            bound_destination = parent / "identity-bound-clone"
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    bound.root, bound_destination
                )
            self.assertEqual(
                caught.exception.code, "bundle_gateway_identity_would_change"
            )
            self.assertFalse(bound_destination.exists())

    def test_unknown_site_payload_links_overlap_and_existing_destination_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()

            unknown = self.fixture(parent, "unknown")
            unknown._write(unknown.site / "rogue_secret.py", b"SECRET = True\n")
            unknown_destination = parent / "unknown-clone"
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    unknown.root, unknown_destination
                )
            self.assertEqual(
                caught.exception.code, "bundle_unrecorded_site_packages_file"
            )
            self.assertFalse(unknown_destination.exists())

            linked = self.fixture(parent, "linked")
            outside = parent / "outside.txt"
            outside.write_text("keep\n")
            (linked.root / "linked.txt").symlink_to(outside)
            linked_destination = parent / "linked-clone"
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    linked.root, linked_destination
                )
            self.assertEqual(caught.exception.code, "bundle_source_link_forbidden")
            self.assertEqual(outside.read_text(), "keep\n")
            self.assertFalse(linked_destination.exists())

            overlap = self.fixture(parent, "overlap")
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    overlap.root, overlap.root / "clone"
                )
            self.assertEqual(
                caught.exception.code, "bundle_source_destination_overlap"
            )

            existing = self.fixture(parent, "existing-source")
            existing_destination = parent / "existing-destination"
            existing_destination.mkdir()
            marker = existing_destination / "marker.txt"
            marker.write_text("preserve\n")
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    existing.root, existing_destination
                )
            self.assertEqual(caught.exception.code, "bundle_destination_exists")
            self.assertEqual(marker.read_text(), "preserve\n")

            linked_destination_target = parent / "linked-destination-target"
            linked_destination_target.mkdir()
            linked_destination_marker = linked_destination_target / "marker.txt"
            linked_destination_marker.write_text("preserve\n")
            linked_destination = parent / "linked-destination"
            try:
                linked_destination.symlink_to(
                    linked_destination_target, target_is_directory=True
                )
            except (NotImplementedError, OSError):
                linked_destination = None
            if linked_destination is not None:
                with self.assertRaises(clone_module.BundleCloneError) as caught:
                    clone_module.clone_qualified_bundle(
                        existing.root, linked_destination
                    )
                self.assertEqual(
                    caught.exception.code, "bundle_destination_exists"
                )
                self.assertEqual(linked_destination_marker.read_text(), "preserve\n")

    def test_artifact_record_report_and_copy_failures_rollback_only_new_clone(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()

            artifact = self.fixture(parent, "artifact")
            (artifact.root / "models/qwen/config.json").write_bytes(b"tampered\n")
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    artifact.root, parent / "artifact-clone"
                )
            self.assertEqual(
                caught.exception.code, "bundle_artifact_identity_mismatch"
            )

            record = self.fixture(parent, "record")
            (record.gateway / "gateway.py").write_bytes(b"tampered\n")
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    record.root, parent / "record-clone"
                )
            self.assertEqual(caught.exception.code, "bundle_record_digest_mismatch")

            report_fixture = self.fixture(parent, "report")
            report_path = report_fixture.root / "qualification-report.json"
            value = json.loads(report_path.read_text())
            value["identity"]["manifest_digest"] = "0" * 64
            report_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    report_fixture.root, parent / "report-clone"
                )
            self.assertEqual(caught.exception.code, "bundle_runtime_identity_mismatch")

            provenance_fixture = self.fixture(parent, "provenance")
            provenance_path = provenance_fixture.root / "qualification-report.json"
            value = json.loads(provenance_path.read_text())
            value["provenance"]["identity_digest"] = "0" * 64
            provenance_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    provenance_fixture.root, parent / "provenance-clone"
                )
            self.assertEqual(
                caught.exception.code,
                "bundle_qualification_provenance_invalid",
            )

            rollback = self.fixture(parent, "rollback")
            before = rollback.source_hashes()
            destination = parent / "rollback-clone"
            original = clone_module._copy_regular
            invocations = 0

            def fail_second(source: Path, target: Path, expected: object) -> None:
                nonlocal invocations
                invocations += 1
                if invocations == 2:
                    raise clone_module.BundleCloneError("synthetic_copy_failure")
                original(source, target, expected)

            with patch.object(clone_module, "_copy_regular", side_effect=fail_second):
                with self.assertRaises(clone_module.BundleCloneError) as caught:
                    clone_module.clone_qualified_bundle(
                        rollback.root, destination
                    )
            self.assertEqual(caught.exception.code, "synthetic_copy_failure")
            self.assertFalse(destination.exists())
            self.assertEqual(before, rollback.source_hashes())

    @unittest.skipUnless(sys.platform == "win32", "native Windows reparse test")
    def test_windows_source_junction_is_rejected_without_destination(self) -> None:
        import subprocess

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            fixture = self.fixture(parent)
            target = parent / "junction-target"
            target.mkdir()
            marker = target / "marker.txt"
            marker.write_text("preserve\n")
            junction = fixture.root / "junction"
            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(target)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if completed.returncode != 0:
                self.fail("Windows junction coverage could not be exercised")
            destination = parent / "clone"
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    fixture.root, destination
                )
            self.assertEqual(caught.exception.code, "bundle_source_link_forbidden")
            self.assertFalse(destination.exists())
            self.assertEqual(marker.read_text(), "preserve\n")

    @unittest.skipUnless(sys.platform == "win32", "native Windows reparse test")
    def test_windows_pycache_junction_is_rejected_without_destination(self) -> None:
        import subprocess

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            fixture = self.fixture(parent)
            target = parent / "pycache-target"
            target.mkdir()
            marker = target / "gateway.cpython-311.pyc"
            marker.write_bytes(b"outside bytecode\n")
            junction = fixture.gateway / "__pycache__"
            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(target)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if completed.returncode != 0:
                self.fail("Windows pycache junction coverage could not be exercised")
            destination = parent / "clone"
            with self.assertRaises(clone_module.BundleCloneError) as caught:
                clone_module.clone_qualified_bundle(
                    fixture.root, destination
                )
            self.assertEqual(caught.exception.code, "bundle_source_link_forbidden")
            self.assertFalse(destination.exists())
            self.assertEqual(marker.read_bytes(), b"outside bytecode\n")


if __name__ == "__main__":
    unittest.main()
