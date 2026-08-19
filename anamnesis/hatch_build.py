"""Fail-closed wheel boundary for the Anamnesis runtime distribution."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PACKAGE = "anamnesis"
PROJECT_NAME = "anamnesis_memory_eval"
SOURCE_ROOT = Path(__file__).resolve().parent / "src" / PACKAGE
DIST_INFO_FILES = {"METADATA", "RECORD", "WHEEL", "entry_points.txt"}


def _is_link_or_reparse(path: Path) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(status, "st_file_attributes", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _expected_modules() -> dict[str, tuple[str, int]]:
    source_parent = SOURCE_ROOT.parent
    if (
        _is_link_or_reparse(source_parent)
        or _is_link_or_reparse(SOURCE_ROOT)
        or not SOURCE_ROOT.is_dir()
    ):
        raise RuntimeError("Anamnesis source root is not a real directory")

    expected: dict[str, tuple[str, int]] = {}
    for current_raw, directories, filenames in os.walk(
        SOURCE_ROOT,
        followlinks=False,
    ):
        current = Path(current_raw)
        for name in (*directories, *filenames):
            if _is_link_or_reparse(current / name):
                raise RuntimeError("Anamnesis source tree contains a link")

        if "__init__.py" not in filenames:
            continue
        package_path = current.relative_to(source_parent)
        if any(not component.isidentifier() for component in package_path.parts):
            raise RuntimeError("Anamnesis source contains an invalid package name")
        for filename in filenames:
            candidate = Path(filename)
            if candidate.suffix == ".py" and candidate.stem.isidentifier():
                source = current / candidate
                status = source.lstat()
                if not stat.S_ISREG(status.st_mode):
                    raise RuntimeError(
                        "Anamnesis source contains an unsupported module entry"
                    )
                payload = source.read_bytes()
                expected[(package_path / candidate.name).as_posix()] = (
                    hashlib.sha256(payload).hexdigest(),
                    len(payload),
                )
    if f"{PACKAGE}/__init__.py" not in expected:
        raise RuntimeError("Anamnesis package root is incomplete")
    return expected


def _safe_archive_name(name: str) -> bool:
    if not name or "\\" in name or name.startswith("/"):
        return False
    path = PurePosixPath(name)
    return bool(path.parts) and all(part not in {"", ".", ".."} for part in path.parts)


def _remove_generated_wheel(path: Path) -> None:
    """Remove only the exact regular wheel just produced by Hatchling."""

    if (
        path.name.startswith(f"{PROJECT_NAME}-")
        and path.suffix == ".whl"
        and not _is_link_or_reparse(path)
        and path.is_file()
    ):
        path.unlink()


def _verify_record(
    payloads: dict[str, bytes],
    record_name: str,
) -> None:
    try:
        rows = list(
            csv.reader(
                io.StringIO(payloads[record_name].decode("utf-8"), newline="")
            )
        )
    except (UnicodeDecodeError, csv.Error) as exc:
        raise RuntimeError("Anamnesis wheel RECORD cannot be parsed") from exc
    if any(len(row) != 3 for row in rows) or len(rows) != len(payloads):
        raise RuntimeError("Anamnesis wheel RECORD shape is invalid")
    records = {row[0]: (row[1], row[2]) for row in rows}
    if len(records) != len(rows) or set(records) != set(payloads):
        raise RuntimeError("Anamnesis wheel RECORD paths are not exact")

    for name, payload in payloads.items():
        digest, size = records[name]
        if name == record_name:
            if digest or size:
                raise RuntimeError("Anamnesis wheel RECORD self-row is invalid")
            continue
        encoded_digest = base64.urlsafe_b64encode(
            hashlib.sha256(payload).digest()
        ).rstrip(b"=").decode("ascii")
        if digest != f"sha256={encoded_digest}" or size != str(len(payload)):
            raise RuntimeError("Anamnesis wheel RECORD content is invalid")


def _verify_wheel(
    path: Path,
    expected_modules: dict[str, tuple[str, int]],
) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)) or any(
                not _safe_archive_name(name) for name in names
            ):
                raise RuntimeError("Anamnesis wheel has unsafe member names")
            for entry in entries:
                mode = (entry.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise RuntimeError("Anamnesis wheel contains a symlink")

            roots = {
                PurePosixPath(name).parts[0]
                for name in names
                if PurePosixPath(name).parts[0].endswith(".dist-info")
            }
            if len(roots) != 1:
                raise RuntimeError("Anamnesis wheel dist-info root is not exact")
            dist_info = next(iter(roots))
            if not dist_info.startswith(f"{PROJECT_NAME}-"):
                raise RuntimeError("Anamnesis wheel identity is invalid")
            expected = set(expected_modules) | {
                f"{dist_info}/{filename}" for filename in DIST_INFO_FILES
            }
            if set(names) != expected:
                raise RuntimeError(
                    "Anamnesis wheel payload is not exact "
                    f"(unexpected={len(set(names) - expected)}, "
                    f"missing={len(expected - set(names))})"
                )
            bad_test = archive.testzip()
            if bad_test is not None:
                raise RuntimeError("Anamnesis wheel failed its integrity check")
            payloads = {name: archive.read(name) for name in names}

            for name, (source_digest, source_size) in expected_modules.items():
                payload = payloads[name]
                if (
                    len(payload) != source_size
                    or hashlib.sha256(payload).hexdigest() != source_digest
                ):
                    raise RuntimeError(
                        "Anamnesis wheel module differs from its source"
                    )
            _verify_record(payloads, f"{dist_info}/RECORD")
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError("Anamnesis wheel cannot be verified") from exc


class CustomBuildHook(BuildHookInterface):
    """Verify source identity before build and the whole wheel after build."""

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        if self.target_name == "wheel":
            _expected_modules()

    def finalize(
        self,
        version: str,
        build_data: dict[str, object],
        artifact_path: str,
    ) -> None:
        if self.target_name != "wheel":
            return
        artifact = Path(artifact_path)
        try:
            _verify_wheel(artifact, _expected_modules())
        except Exception:
            _remove_generated_wheel(artifact)
            raise
