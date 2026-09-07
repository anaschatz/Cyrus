"""Content identity for the sealed Python installation, independent of its location."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Iterable


PYTHON_CONVENIENCE_LINKS = {
    "bin/2to3": "2to3-3.11",
    "bin/idle3": "idle3.11",
    "bin/pydoc3": "pydoc3.11",
    "bin/python": "python3.11",
    "bin/python3": "python3.11",
    "bin/python3-config": "python3.11-config",
    "lib/pkgconfig/python3-embed.pc": "python-3.11-embed.pc",
    "lib/pkgconfig/python3.pc": "python-3.11.pc",
    "share/man/man1/python3.1": "python3.11.1",
}


def _identity(info: os.stat_result) -> tuple[int, ...]:
    # Windows stat infers execute bits from the suffix; fstat cannot do so.
    mode = (
        info.st_mode & ~0o111 if hasattr(info, "st_file_attributes") else info.st_mode
    )
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        mode,
    )


def _reparse(info: os.stat_result) -> bool:
    return bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def hash_file(path: Path) -> tuple[int, str]:
    """Hash one stable regular file without following a substituted FIFO or link."""
    initial = path.lstat()
    if not stat.S_ISREG(initial.st_mode) or _reparse(initial):
        raise OSError("unsafe artifact type")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _identity(initial) != _identity(opened) or _reparse(opened):
            raise OSError("artifact changed before hashing")
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(min(1024 * 1024, opened.st_size - size + 1)):
                size += len(chunk)
                if size > opened.st_size:
                    raise OSError("artifact grew while hashing")
                digest.update(chunk)
        if (
            size != opened.st_size
            or _identity(os.fstat(descriptor)) != _identity(opened)
            or _identity(path.lstat()) != _identity(opened)
        ):
            raise OSError("artifact changed while hashing")
        return size, digest.hexdigest()
    finally:
        os.close(descriptor)


def environment_digest(files: Iterable[tuple[str, int, str, bool]]) -> str:
    """Hash sorted relative path, size, content hash and executable-bit entries."""
    digest = hashlib.sha256(b"cyrus-python-environment-v1\n")
    for entry in sorted(files):
        digest.update(
            json.dumps(entry, ensure_ascii=True, separators=(",", ":")).encode("ascii")
            + b"\n"
        )
    return digest.hexdigest()


def hash_environment(root: Path) -> str:
    files = []

    def visit(directory: Path) -> None:
        before = directory.lstat()
        if not stat.S_ISDIR(before.st_mode) or _reparse(before):
            raise OSError("unsafe Python environment directory")
        with os.scandir(directory) as iterator:
            entries = list(iterator)
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                target = PYTHON_CONVENIENCE_LINKS.get(relative)
                if target is None or os.readlink(path) != target:
                    raise OSError("unsupported Python environment link")
                target_info = (path.parent / target).lstat()
                if not stat.S_ISREG(target_info.st_mode) or _reparse(target_info):
                    raise OSError("unsafe Python convenience link target")
            elif _reparse(info):
                raise OSError("unsafe Python environment entry")
            elif stat.S_ISDIR(info.st_mode):
                visit(path)
            else:
                size, digest = hash_file(path)
                if _identity(path.lstat()) != _identity(info):
                    raise OSError("Python environment changed while hashing")
                files.append((relative, size, digest, bool(info.st_mode & 0o111)))
        if _identity(directory.lstat()) != _identity(before):
            raise OSError("Python environment directory changed while hashing")

    visit(root)
    return environment_digest(files)
