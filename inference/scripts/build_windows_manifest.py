#!/usr/bin/env python3
"""Build the frozen Windows Gemma inference manifest from sealed artifacts.

The builder is deliberately offline.  It hashes an already materialized
llama-server bundle and accepts only the exact primary GGUF selected by the
Windows handoff contract.  It performs no download, install, model load, or
qualification and never writes outside the supplied runtime root.
Run with the gateway package installed or with PYTHONPATH=inference/src.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any

from jarvis_inference.python_environment import hash_file as _hash


GEMMA_REPOSITORY = "google/gemma-4-12B-it-qat-q4_0-gguf"
GEMMA_REVISION = "29d097773436b69ff9feafd636ab4cf873786537"
GEMMA_FILENAME = "gemma-4-12b-it-qat-q4_0.gguf"
GEMMA_SIZE_BYTES = 6_975_879_296
GEMMA_SHA256 = "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b"
GEMMA_PROMPT_PROFILE = {
    "id": "google-gemma4-12b-it-qat-q4_0-v1",
    "model_repository": GEMMA_REPOSITORY,
    "model_revision": GEMMA_REVISION,
    "model_artifact": {
        "filename": GEMMA_FILENAME,
        "size_bytes": GEMMA_SIZE_BYTES,
        "sha256": GEMMA_SHA256,
    },
    "embedded_chat_template": {
        "metadata_key": "tokenizer.chat_template",
        "size_bytes": 18_683,
        "sha256": "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
    },
    "renderer": {
        "id": "cyrus.gemma4-appliance-renderer/1",
        "sha256": "3ce59b2dadfeeff369b337da785deb096ca50d8a2182570cc9db36f7f63a1951",
    },
    "tool_parser": {
        "id": "cyrus.gemma4-single-tool-parser/1",
        "sha256": "e118a2ab2d39113fd2a18964f8f6ade87692afeed26c87d95d87a0dcb5c6afd4",
    },
    "thinking": False,
}

_COMMIT = re.compile(r"^[0-9a-f]{40}$", re.ASCII)
_BUILD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$", re.ASCII)


def _is_reparse(path: Path) -> bool:
    try:
        attributes = path.lstat().st_file_attributes
    except AttributeError:
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _native_windows() -> bool:
    return os.name == "nt"


def _windows_local_drive_path(path: Path | PureWindowsPath) -> bool:
    candidate = PureWindowsPath(str(path))
    return bool(
        re.fullmatch(r"[A-Za-z]:", candidate.drive)
        and candidate.root == "\\"
    )


def _native_windows_drive_type(root: str) -> int:  # pragma: no cover - native Windows only
    try:
        import ctypes
        from ctypes import wintypes

        function = ctypes.windll.kernel32.GetDriveTypeW
        function.argtypes = [wintypes.LPCWSTR]
        function.restype = wintypes.UINT
        return int(function(root))
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


def _windows_drive_is_local(
    path: Path | PureWindowsPath,
    drive_type_reader: Any = None,
) -> bool:
    candidate = PureWindowsPath(str(path))
    if not _windows_local_drive_path(candidate):
        return False
    reader = drive_type_reader or _native_windows_drive_type
    try:
        return reader(candidate.anchor) in {2, 3, 5, 6}
    except (OSError, TypeError, ValueError):
        return False


def _require_windows_local_drive_paths(*paths: Path) -> None:
    if _native_windows() and any(
        not _windows_drive_is_local(path) for path in paths
    ):
        raise ValueError("Windows runtime and output require local-drive paths")


def _reject_reparse_ancestor_chain(path: Path) -> None:
    """Reject every existing link/reparse component before path resolution."""

    for candidate in (path, *path.parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("Windows path ancestry cannot be inspected") from exc
        attributes = getattr(info, "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if stat.S_ISLNK(info.st_mode) or bool(attributes & reparse_flag):
            raise ValueError("Windows path ancestry contains a reparse point")


def _reject_windows_reparse_ancestors(*paths: Path) -> None:
    if _native_windows():
        for path in paths:
            _reject_reparse_ancestor_chain(path)


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and not _is_reparse(path)


def _artifact(root: Path, path: Path) -> dict[str, Any]:
    if not _regular_file(path):
        raise ValueError("artifact must be a regular non-reparse file")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("artifact escapes runtime root") from exc
    size, digest = _hash(path)
    return {"path": relative, "size_bytes": size, "sha256": digest}


def _safe_report_name(value: str) -> str:
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name != value
        or not value.startswith("qualification-report")
        or candidate.suffix != ".json"
    ):
        raise ValueError("qualification report path is invalid")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    """Publish one frozen manifest atomically and without overwriting a peer."""

    raw = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if os.name != "nt":
                os.fchmod(stream.fileno(), 0o600)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ValueError("output manifest already exists") from exc
        if os.name != "nt" and hasattr(os, "O_DIRECTORY"):
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def build(
    root: Path,
    output: Path,
    llama_cpp_commit: str,
    build_id: str,
    *,
    context_window: int = 32_768,
    gpu_layers: int = 99,
    batch_size: int = 2_048,
    ubatch_size: int = 512,
    threads: int = 8,
    enable_tools: bool = False,
    enable_structured: bool = False,
    qualification_report: str = "qualification-report.json",
) -> dict[str, Any]:
    """Hash a sealed Windows bundle and write one strict manifest."""

    # Validate the original Windows syntax before ``absolute()`` can normalize
    # a drive-relative operand into an apparently drive-rooted path.
    _require_windows_local_drive_paths(root, output)
    root = root.absolute()
    output = output.absolute()
    _reject_windows_reparse_ancestors(root, output)
    if (
        not root.is_dir()
        or root.is_symlink()
        or _is_reparse(root)
        or output.parent != root
    ):
        raise ValueError("runtime root or output path is invalid")
    if _COMMIT.fullmatch(llama_cpp_commit) is None:
        raise ValueError("llama.cpp commit is invalid")
    if _BUILD_ID.fullmatch(build_id) is None:
        raise ValueError("llama-server build identifier is invalid")
    for name, value, minimum, maximum in (
        ("context_window", context_window, 1_024, 262_144),
        ("gpu_layers", gpu_layers, 1, 1_024),
        ("batch_size", batch_size, 1, 65_536),
        ("ubatch_size", ubatch_size, 1, 65_536),
        ("threads", threads, 1, 1_024),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{name} is invalid")
    if ubatch_size > batch_size:
        raise ValueError("ubatch_size exceeds batch_size")
    if enable_tools and not enable_structured:
        raise ValueError("tool capability requires structured output")
    report_name = _safe_report_name(qualification_report)

    runtime_dir = root / "runtime" / "llama"
    model_dir = root / "models"
    executable = runtime_dir / "llama-server.exe"
    model = model_dir / GEMMA_FILENAME
    _reject_windows_reparse_ancestors(
        runtime_dir, model_dir, executable, model
    )
    if (
        not runtime_dir.is_dir()
        or runtime_dir.is_symlink()
        or _is_reparse(runtime_dir)
        or not model_dir.is_dir()
        or model_dir.is_symlink()
        or _is_reparse(model_dir)
        or not _regular_file(executable)
        or not _regular_file(model)
    ):
        raise ValueError("sealed Windows runtime layout is incomplete")

    model_size, model_digest = _hash(model)
    if model_size != GEMMA_SIZE_BYTES or model_digest != GEMMA_SHA256:
        raise ValueError("primary Gemma artifact identity mismatch")

    dll_paths = sorted(
        (path for path in runtime_dir.iterdir() if path.suffix.lower() == ".dll"),
        key=lambda path: path.name.lower(),
    )
    if not dll_paths or any(not _regular_file(path) for path in dll_paths):
        raise ValueError("llama-server DLL allowlist is empty or unsafe")
    _reject_windows_reparse_ancestors(*dll_paths)
    if len({path.name.lower() for path in dll_paths}) != len(dll_paths):
        raise ValueError("llama-server DLL names are not unique")

    unexpected = [
        path
        for path in runtime_dir.iterdir()
        if path.is_file()
        and path.name.lower() != "llama-server.exe"
        and path.suffix.lower() != ".dll"
    ]
    if unexpected:
        raise ValueError("llama runtime contains an undeclared regular file")

    manifest = {
        "manifest_schema_version": 1,
        "alias": "jarvis-local",
        "platform": "windows-x86_64",
        "backend": {
            "type": "llama-server",
            "llama_cpp_commit": llama_cpp_commit,
            "build_id": build_id,
            "executable": _artifact(root, executable),
            "dlls": [_artifact(root, path) for path in dll_paths],
            "parameters": {
                "context_window": context_window,
                "gpu_layers": gpu_layers,
                "batch_size": batch_size,
                "ubatch_size": ubatch_size,
                "threads": threads,
                "flash_attention": True,
            },
        },
        "model": {
            "format": "gguf",
            "context_window": context_window,
            "artifacts": [
                {
                    "path": model.relative_to(root).as_posix(),
                    "size_bytes": model_size,
                    "sha256": model_digest,
                }
            ],
        },
        # Serving never evaluates model Jinja. This exact profile selects the
        # appliance-owned renderer/parser and binds them to the embedded
        # template identity in the already hash-locked GGUF.
        "prompt_profile": GEMMA_PROMPT_PROFILE,
        "python": {},
        "capabilities": {
            "text": True,
            "stream": True,
            "tools": enable_tools,
            "json_object": enable_structured,
            "json_schema": enable_structured,
        },
        "qualification_report": report_name,
        "profile_qualification_report": "qualification-profile-local-primary.json",
    }
    _write(output, manifest)
    return {
        "ready": True,
        "code": "materialization.windows_inference_manifest_written",
        "model_repository": GEMMA_REPOSITORY,
        "model_revision": GEMMA_REVISION,
        "dlls": len(dll_paths),
        "network_used": False,
        "model_loaded": False,
        "native_qualification_complete": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="build-windows-inference-manifest")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-cpp-commit", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--context-window", type=int, default=32_768)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--enable-tools", action="store_true")
    parser.add_argument("--enable-structured", action="store_true")
    parser.add_argument("--qualification-report", default="qualification-report.json")
    args = parser.parse_args()
    try:
        result = build(
            args.root,
            args.output,
            args.llama_cpp_commit,
            args.build_id,
            context_window=args.context_window,
            gpu_layers=args.gpu_layers,
            batch_size=args.batch_size,
            ubatch_size=args.ubatch_size,
            threads=args.threads,
            enable_tools=args.enable_tools,
            enable_structured=args.enable_structured,
            qualification_report=args.qualification_report,
        )
    except (OSError, ValueError):
        print(
            json.dumps(
                {
                    "ready": False,
                    "code": "materialization.windows_inference_manifest_failed",
                },
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
