#!/usr/bin/env python3
"""Build the pinned OPUS Greek-to-English CTranslate2 runtime pack offline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from typing import Any

from jarvis_orchestrator.config import _parse_strict_json
from jarvis_orchestrator.domain import canonical_json
from jarvis_orchestrator.translation import (
    TRANSLATION_BACKEND,
    TRANSLATION_COMPUTE_TYPE,
    TRANSLATION_CONVERSION_POLICY_SHA256,
    TRANSLATION_CTRANSLATE2_VERSION,
    TRANSLATION_LICENSE,
    TRANSLATION_MANIFEST_VERSION,
    TRANSLATION_MODEL_ID,
    TRANSLATION_MODEL_REVISION,
    TRANSLATION_SENTENCEPIECE_VERSION,
    _read_bounded,
    load_translation_manifest,
)


BUILD_LOCK_VERSION = "jarvis.translation-build-lock/1"
_SHA256 = frozenset("0123456789abcdef")


def _read_json(path: Path, maximum: int = 256 * 1024) -> Any:
    if not path.is_absolute():
        raise ValueError("build-lock path is invalid")
    return _parse_strict_json(_read_bounded(path, maximum))


def canonical_build_lock_sha256(document: Any) -> str:
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def _artifact_list(value: Any, *, field: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} is invalid")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise ValueError(f"{field} is invalid")
        path = item.get("path")
        size = item.get("size")
        digest = item.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or "/" in path
            or "\\" in path
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
            or not isinstance(digest, str)
            or len(digest) != 64
            or not set(digest) <= _SHA256
        ):
            raise ValueError(f"{field} is invalid")
        result.append(dict(item))
    paths = [item["path"] for item in result]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError(f"{field} is not a sorted closed set")
    return tuple(result)


def load_build_lock(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    required = {
        "version",
        "model_id",
        "revision",
        "license",
        "source_language",
        "target_language",
        "source_files",
        "builder",
        "conversion",
        "runtime_files",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError("build-lock schema is invalid")
    if {
        "version": document.get("version"),
        "model_id": document.get("model_id"),
        "revision": document.get("revision"),
        "license": document.get("license"),
        "source_language": document.get("source_language"),
        "target_language": document.get("target_language"),
    } != {
        "version": BUILD_LOCK_VERSION,
        "model_id": TRANSLATION_MODEL_ID,
        "revision": TRANSLATION_MODEL_REVISION,
        "license": TRANSLATION_LICENSE,
        "source_language": "el",
        "target_language": "en",
    }:
        raise ValueError("build-lock identity is invalid")
    _artifact_list(document["source_files"], field="source_files")
    _artifact_list(document["runtime_files"], field="runtime_files")
    builder = document.get("builder")
    if (
        not isinstance(builder, dict)
        or set(builder) != {"python_implementation", "python_version", "packages"}
        or builder.get("python_implementation") != "CPython"
        or builder.get("python_version") != "3.11.14"
        or not isinstance(builder.get("packages"), dict)
        or not builder["packages"]
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in builder["packages"].items()
        )
    ):
        raise ValueError("build-lock builder is invalid")
    conversion = document.get("conversion")
    if conversion != {
        "api": "ctranslate2.converters.TransformersConverter",
        "activation_scales": None,
        "copy_files": ["source.spm", "target.spm"],
        "load_as_float16": True,
        "revision": None,
        "low_cpu_mem_usage": False,
        "trust_remote_code": False,
        "vocab_mapping": None,
        "quantization": "int8",
        "force": False,
        "network_allowed": False,
    }:
        raise ValueError("build-lock conversion policy is invalid")
    if canonical_build_lock_sha256(document) != TRANSLATION_CONVERSION_POLICY_SHA256:
        raise ValueError("build-lock digest is invalid")
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_closed_artifacts(
    root: Path,
    artifacts: tuple[dict[str, Any], ...],
) -> None:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("artifact root is invalid")
    entries = tuple(root.iterdir())
    expected = {item["path"] for item in artifacts}
    if (
        len(entries) != len(expected)
        or {entry.name for entry in entries} != expected
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise ValueError("artifact root is not closed")
    for item in artifacts:
        selected = root / item["path"]
        if selected.stat().st_size != item["size"]:
            raise ValueError("artifact size mismatch")
        if _sha256_file(selected) != item["sha256"]:
            raise ValueError("artifact digest mismatch")


def verify_builder(document: dict[str, Any]) -> None:
    builder = document["builder"]
    if (
        platform.python_implementation() != builder["python_implementation"]
        or platform.python_version() != builder["python_version"]
    ):
        raise ValueError("builder Python identity mismatch")
    observed = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
    }
    allowed_local = {"jarvis-orchestrator"}
    if (
        {name: observed.get(name) for name in builder["packages"]}
        != builder["packages"]
        or set(observed) - set(builder["packages"]) - allowed_local
    ):
        raise ValueError("builder package identity mismatch")


def _deny_network(event: str, args: tuple[Any, ...]) -> None:
    del args
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname"}:
        raise RuntimeError("network is disabled for translation conversion")


def build_pack(
    *,
    source_root: Path,
    output_root: Path,
    build_lock_path: Path,
    manifest_output: Path,
) -> dict[str, str]:
    document = load_build_lock(build_lock_path)
    verify_builder(document)
    verify_closed_artifacts(
        source_root,
        _artifact_list(document["source_files"], field="source_files"),
    )
    if not output_root.is_absolute() or output_root.exists():
        raise ValueError("output root must be a new absolute path")
    if (
        not manifest_output.is_absolute()
        or manifest_output.exists()
        or output_root == manifest_output.parent
    ):
        raise ValueError("manifest output path is invalid")
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
    )
    sys.addaudithook(_deny_network)
    from ctranslate2.converters import TransformersConverter

    converter = TransformersConverter(
        str(source_root),
        activation_scales=None,
        copy_files=["source.spm", "target.spm"],
        load_as_float16=True,
        revision=None,
        low_cpu_mem_usage=False,
        trust_remote_code=False,
    )
    converter.convert(
        str(output_root),
        vmap=None,
        quantization="int8",
        force=False,
    )
    runtime_artifacts = _artifact_list(
        document["runtime_files"],
        field="runtime_files",
    )
    verify_closed_artifacts(output_root, runtime_artifacts)
    manifest = {
        "version": TRANSLATION_MANIFEST_VERSION,
        "backend": TRANSLATION_BACKEND,
        "model_id": TRANSLATION_MODEL_ID,
        "revision": TRANSLATION_MODEL_REVISION,
        "license": TRANSLATION_LICENSE,
        "source_language": "el",
        "target_language": "en",
        "compute_type": TRANSLATION_COMPUTE_TYPE,
        "ctranslate2_version": TRANSLATION_CTRANSLATE2_VERSION,
        "sentencepiece_version": TRANSLATION_SENTENCEPIECE_VERSION,
        "conversion_policy_sha256": TRANSLATION_CONVERSION_POLICY_SHA256,
        "beam_size": 4,
        "inter_threads": 1,
        "intra_threads": 1,
        "max_queued_batches": 1,
        "files": list(runtime_artifacts),
        "total_size": sum(item["size"] for item in runtime_artifacts),
    }
    raw = canonical_json(manifest).encode("utf-8")
    with open(manifest_output, "xb", opener=lambda path, flags: os.open(path, flags, 0o600)) as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    load_translation_manifest(output_root, manifest_output, manifest_sha256)
    return {
        "build_lock_sha256": TRANSLATION_CONVERSION_POLICY_SHA256,
        "manifest_sha256": manifest_sha256,
        "model_total_size": str(manifest["total_size"]),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-root", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--build-lock", type=Path, required=True)
    result.add_argument("--manifest-output", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parsed = parser().parse_args(argv)
    try:
        result = build_pack(
            source_root=parsed.source_root,
            output_root=parsed.output_root,
            build_lock_path=parsed.build_lock,
            manifest_output=parsed.manifest_output,
        )
    except BaseException:
        print(canonical_json({"status": "failed", "code": "translation_build_failed"}))
        return 2
    print(canonical_json({"status": "passed", **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
