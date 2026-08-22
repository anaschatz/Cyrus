"""Offline Greek-to-English projection with a killable CTranslate2 worker."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import unicodedata
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .context_assembler import ContextAssemblyError
from .domain import canonical_json


TRANSLATION_MANIFEST_VERSION = "jarvis.translation-model-manifest/1"
TRANSLATION_REQUEST_VERSION = "jarvis.input-translation-request/1"
TRANSLATION_RESPONSE_VERSION = "jarvis.input-translation-response/1"
TRANSLATION_BACKEND = "opus-mt-ct2"
TRANSLATION_MODEL_ID = "Helsinki-NLP/opus-mt-tc-big-el-en"
TRANSLATION_MODEL_REVISION = "a69108562775c6727edf3540346f9d8900b73634"
TRANSLATION_LICENSE = "CC-BY-4.0"
TRANSLATION_COMPUTE_TYPE = "int8"
TRANSLATION_CTRANSLATE2_VERSION = "4.8.1"
TRANSLATION_SENTENCEPIECE_VERSION = "0.2.2"
TRANSLATION_MAXIMUM_COMPONENTS = 16
TRANSLATION_MAXIMUM_TEXT_BYTES = 8_192
TRANSLATION_MAXIMUM_REQUEST_BYTES = 65_536
TRANSLATION_MAXIMUM_RESPONSE_BYTES = 65_536
TRANSLATION_MAXIMUM_MODEL_BYTES = 512 * 1024 * 1024
TRANSLATION_REQUIRED_FILES = frozenset(
    {
        "config.json",
        "model.bin",
        "shared_vocabulary.json",
        "source.spm",
        "target.spm",
    }
)
_CONVERSION_POLICY = {
    "converter": "ct2-transformers-converter",
    "ctranslate2_version": TRANSLATION_CTRANSLATE2_VERSION,
    "transformers_version": "4.57.6",
    "torch_version": "2.13.0",
    "sentencepiece_version": TRANSLATION_SENTENCEPIECE_VERSION,
    "model_id": TRANSLATION_MODEL_ID,
    "revision": TRANSLATION_MODEL_REVISION,
    "quantization": TRANSLATION_COMPUTE_TYPE,
    "load_as_float16": True,
    "copied_files": ["source.spm", "target.spm"],
    "runtime_files": sorted(TRANSLATION_REQUIRED_FILES),
}
TRANSLATION_CONVERSION_POLICY_SHA256 = hashlib.sha256(
    canonical_json(_CONVERSION_POLICY).encode("utf-8")
).hexdigest()
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_COMPONENT_ID = re.compile(r"[a-z][a-z0-9_.:-]{0,63}")
_GREEK = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")


def contains_greek(value: str) -> bool:
    return bool(_GREEK.search(value))


def _bounded_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContextAssemblyError("input_translation_invalid", status=502)
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value or any(
        unicodedata.category(char).startswith("C") for char in value
    ):
        raise ContextAssemblyError("input_translation_invalid", status=502)
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError:
        raise ContextAssemblyError("input_translation_invalid", status=502) from None
    if size > TRANSLATION_MAXIMUM_TEXT_BYTES:
        raise ContextAssemblyError("input_translation_too_large", status=502)
    return value


def validate_translation_output(value: Any) -> str:
    text = _bounded_text(value)
    if contains_greek(text) or "<unk>" in text or "\ufffd" in text or "\u2047" in text:
        raise ContextAssemblyError("input_translation_invalid", status=502)
    return text


@dataclass(frozen=True, slots=True)
class TranslationArtifact:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class TranslationModelManifest:
    model_id: str
    revision: str
    files: tuple[TranslationArtifact, ...]
    total_size: int


def _read_bounded(path: Path, maximum: int) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("translation manifest must be a regular file")
        with path.open("rb") as stream:
            raw = stream.read(maximum + 1)
    except OSError as exc:
        raise ValueError("translation manifest is unreadable") from exc
    if len(raw) > maximum:
        raise ValueError("translation manifest exceeds its maximum size")
    return raw


def _closed_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("translation manifest contains duplicate keys")
        result[key] = value
    return result


def load_translation_manifest(
    model_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> TranslationModelManifest:
    """Verify the closed converted-model pack before any worker is spawned."""

    if not model_root.is_absolute() or not manifest_path.is_absolute():
        raise ValueError("translation artifact paths must be absolute")
    if not _SHA256.fullmatch(expected_manifest_sha256):
        raise ValueError("translation manifest sha256 is invalid")
    raw = _read_bounded(manifest_path, 256 * 1024)
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("translation manifest hash mismatch")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_closed_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("translation manifest is invalid JSON") from exc
    required_keys = {
        "version",
        "backend",
        "model_id",
        "revision",
        "license",
        "source_language",
        "target_language",
        "compute_type",
        "ctranslate2_version",
        "sentencepiece_version",
        "conversion_policy_sha256",
        "beam_size",
        "inter_threads",
        "intra_threads",
        "max_queued_batches",
        "files",
        "total_size",
    }
    if not isinstance(document, dict) or set(document) != required_keys:
        raise ValueError("translation manifest schema is invalid")
    frozen = {
        "version": TRANSLATION_MANIFEST_VERSION,
        "backend": TRANSLATION_BACKEND,
        "model_id": TRANSLATION_MODEL_ID,
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
    }
    if any(
        type(document.get(key)) is not type(value) or document.get(key) != value
        for key, value in frozen.items()
    ):
        raise ValueError("translation manifest policy is invalid")
    revision = document.get("revision")
    if revision != TRANSLATION_MODEL_REVISION or not _REVISION.fullmatch(revision):
        raise ValueError("translation model revision is invalid")
    files = document.get("files")
    total_size = document.get("total_size")
    if (
        not isinstance(files, list)
        or not files
        or not isinstance(total_size, int)
        or isinstance(total_size, bool)
        or not 1 <= total_size <= TRANSLATION_MAXIMUM_MODEL_BYTES
    ):
        raise ValueError("translation model file manifest is invalid")
    artifacts: list[TranslationArtifact] = []
    names: set[str] = set()
    computed_total = 0
    if model_root.is_symlink():
        raise ValueError("translation model root cannot be a symlink")
    try:
        root = model_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("translation model root is unavailable") from exc
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise ValueError("translation model artifact schema is invalid")
        relative = item.get("path")
        size = item.get("size")
        digest = item.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or "\\" in relative
            or relative in names
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            raise ValueError("translation model artifact entry is invalid")
        names.add(relative)
        selected = model_root / relative
        try:
            if selected.is_symlink() or not selected.is_file():
                raise ValueError("translation model artifact is missing")
            resolved = selected.resolve(strict=True)
            if resolved.parent != root and root not in resolved.parents:
                raise ValueError("translation model artifact escapes its root")
            if selected.stat().st_size != size:
                raise ValueError("translation model artifact size mismatch")
            hasher = hashlib.sha256()
            with selected.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != digest:
                raise ValueError("translation model artifact hash mismatch")
        except OSError as exc:
            raise ValueError("translation model artifact is unreadable") from exc
        computed_total += size
        artifacts.append(TranslationArtifact(relative, size, digest))
    actual_names = {
        str(path.relative_to(root)).replace(os.sep, "/")
        for path in root.rglob("*")
        if path.is_file()
    }
    if names != TRANSLATION_REQUIRED_FILES or actual_names != names:
        raise ValueError("translation model artifact set is invalid")
    if computed_total != total_size:
        raise ValueError("translation model total size mismatch")
    return TranslationModelManifest(
        model_id=TRANSLATION_MODEL_ID,
        revision=revision,
        files=tuple(artifacts),
        total_size=total_size,
    )


class TranslationPort(Protocol):
    async def translate(
        self,
        *,
        task_id: str,
        epoch: int,
        items: Mapping[str, str],
    ) -> Mapping[str, str]: ...


class DisabledTranslationPort:
    async def translate(
        self,
        *,
        task_id: str,
        epoch: int,
        items: Mapping[str, str],
    ) -> Mapping[str, str]:
        del task_id, epoch, items
        raise ContextAssemblyError("input_translation_unavailable", status=502)


class OpusCTranslate2TranslationPort:
    """Spawn one owned process per batch so cancellation releases all model RAM."""

    def __init__(
        self,
        *,
        model_root: Path,
        manifest_path: Path,
        manifest_sha256: str,
        timeout_seconds: float,
        worker_python: Path | None = None,
        process_factory: Any = asyncio.create_subprocess_exec,
    ) -> None:
        self.model_root = model_root
        self.manifest_path = manifest_path
        self.manifest_sha256 = manifest_sha256
        self.timeout_seconds = timeout_seconds
        self.worker_python = Path(sys.executable) if worker_python is None else worker_python
        self.process_factory = process_factory
        self._worker_lock = asyncio.Lock()
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0.1 <= float(timeout_seconds) <= 30.0
        ):
            raise ValueError("translation timeout is invalid")
        if not self.worker_python.is_absolute() or not self.worker_python.is_file():
            raise ValueError("translation worker Python is unavailable")
        self.manifest = load_translation_manifest(
            model_root,
            manifest_path,
            manifest_sha256,
        )

    @staticmethod
    async def _stop(process: Any) -> None:
        if process.returncode is not None:
            return
        try:
            pid = getattr(process, "pid", None)
            if os.name == "posix" and isinstance(pid, int) and pid > 0:
                os.killpg(pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            try:
                pid = getattr(process, "pid", None)
                if os.name == "posix" and isinstance(pid, int) and pid > 0:
                    os.killpg(pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                return
            await process.wait()

    async def translate(
        self,
        *,
        task_id: str,
        epoch: int,
        items: Mapping[str, str],
    ) -> Mapping[str, str]:
        del task_id, epoch
        if not isinstance(items, Mapping) or not 1 <= len(items) <= TRANSLATION_MAXIMUM_COMPONENTS:
            raise ContextAssemblyError("input_translation_invalid", status=502)
        normalized: dict[str, str] = {}
        for component_id, text in items.items():
            if not isinstance(component_id, str) or not _COMPONENT_ID.fullmatch(component_id):
                raise ContextAssemblyError("input_translation_invalid", status=502)
            normalized[component_id] = _bounded_text(text)
        request_id = str(uuid.uuid4())
        payload = {
            "version": TRANSLATION_REQUEST_VERSION,
            "request_id": request_id,
            "items": [
                {"id": component_id, "text": text}
                for component_id, text in normalized.items()
            ],
        }
        raw = (canonical_json(payload) + "\n").encode("utf-8")
        if len(raw) > TRANSLATION_MAXIMUM_REQUEST_BYTES:
            raise ContextAssemblyError("input_translation_too_large", status=502)
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
        process_options: dict[str, Any] = {
            "stdin": asyncio.subprocess.PIPE,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.DEVNULL,
            "env": environment,
            "limit": TRANSLATION_MAXIMUM_RESPONSE_BYTES + 1,
            "close_fds": True,
        }
        if os.name == "posix":
            process_options["start_new_session"] = True
        elif os.name == "nt":  # pragma: no cover - exercised in native Windows CI
            process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        process = None
        try:
            async with asyncio.timeout(float(self.timeout_seconds)):
                async with self._worker_lock:
                    try:
                        process = await self.process_factory(
                            str(self.worker_python),
                            "-I",
                            "-m",
                            "jarvis_orchestrator.translation_worker",
                            "--model-root",
                            str(self.model_root),
                            "--manifest-path",
                            str(self.manifest_path),
                            "--manifest-sha256",
                            self.manifest_sha256,
                            **process_options,
                        )
                        stdout, _stderr = await process.communicate(raw)
                    except asyncio.CancelledError:
                        if process is not None:
                            await self._stop(process)
                        raise
                    except (OSError, ValueError) as exc:
                        if process is not None:
                            await self._stop(process)
                        code = (
                            "input_translation_unavailable"
                            if process is None
                            else "input_translation_failed"
                        )
                        raise ContextAssemblyError(code, status=502) from exc
        except asyncio.CancelledError:
            raise
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise ContextAssemblyError("input_translation_timeout", status=502) from exc
        assert process is not None
        if process.returncode != 0:
            raise ContextAssemblyError("input_translation_failed", status=502)
        if not stdout or len(stdout) > TRANSLATION_MAXIMUM_RESPONSE_BYTES:
            raise ContextAssemblyError("input_translation_invalid", status=502)
        try:
            response = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContextAssemblyError("input_translation_invalid", status=502) from exc
        if (
            not isinstance(response, dict)
            or set(response) != {"version", "request_id", "items"}
            or response.get("version") != TRANSLATION_RESPONSE_VERSION
            or response.get("request_id") != request_id
            or not isinstance(response.get("items"), list)
        ):
            raise ContextAssemblyError("input_translation_invalid", status=502)
        translated: dict[str, str] = {}
        for item in response["items"]:
            if not isinstance(item, dict) or set(item) != {"id", "text"}:
                raise ContextAssemblyError("input_translation_invalid", status=502)
            component_id = item.get("id")
            if component_id not in normalized or component_id in translated:
                raise ContextAssemblyError("input_translation_invalid", status=502)
            translated[component_id] = validate_translation_output(item.get("text"))
        if tuple(translated) != tuple(normalized):
            raise ContextAssemblyError("input_translation_invalid", status=502)
        return translated


__all__ = [
    "DisabledTranslationPort",
    "OpusCTranslate2TranslationPort",
    "TRANSLATION_BACKEND",
    "TRANSLATION_COMPUTE_TYPE",
    "TRANSLATION_CONVERSION_POLICY_SHA256",
    "TRANSLATION_CTRANSLATE2_VERSION",
    "TRANSLATION_LICENSE",
    "TRANSLATION_MANIFEST_VERSION",
    "TRANSLATION_MODEL_ID",
    "TRANSLATION_MODEL_REVISION",
    "TRANSLATION_SENTENCEPIECE_VERSION",
    "TranslationModelManifest",
    "TranslationPort",
    "contains_greek",
    "load_translation_manifest",
    "validate_translation_output",
]
