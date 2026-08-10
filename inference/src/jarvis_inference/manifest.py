from __future__ import annotations

import hashlib
import base64
import csv
import importlib.metadata
import os
import platform
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .contracts import ALIAS_RE, Capabilities
from .errors import ApplianceError, invalid, unsupported
from .limits import MANIFEST_BYTES
from .strict_json import canonical_bytes, parse_body

SHA_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.ASCII)
PLATFORMS = {"windows-x86_64", "macos-arm64"}
LLAMA_PARAMETER_TYPES: dict[str, type] = {
    "context_window": int,
    "gpu_layers": int,
    "batch_size": int,
    "ubatch_size": int,
    "threads": int,
    "flash_attention": bool,
}


@dataclass(frozen=True, slots=True)
class Artifact:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Manifest:
    path: Path
    root: Path
    raw: dict[str, Any]
    digest: str
    alias: str
    platform: str
    backend: dict[str, Any]
    model: dict[str, Any]
    python: dict[str, Any]
    capabilities: Capabilities
    qualification_report: Path
    artifacts: tuple[Artifact, ...]
    context_window: int


def _exact(value: Any, required: set[str], optional: set[str], parameter: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise invalid("invalid_manifest", "A manifest object has an invalid shape.", parameter)
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise invalid("invalid_manifest", "A required manifest field is missing.", parameter)
    if unknown:
        raise unsupported(parameter + "." + sorted(unknown)[0])
    return value


def _artifact(value: Any, parameter: str) -> Artifact:
    obj = _exact(value, {"path", "size_bytes", "sha256"}, set(), parameter)
    if not isinstance(obj["path"], str) or not obj["path"]:
        raise invalid("invalid_manifest", "An artifact path is invalid.", parameter + ".path")
    if not isinstance(obj["size_bytes"], int) or isinstance(obj["size_bytes"], bool) or obj["size_bytes"] < 0:
        raise invalid("invalid_manifest", "An artifact size is invalid.", parameter + ".size_bytes")
    if not isinstance(obj["sha256"], str) or not SHA_RE.fullmatch(obj["sha256"]):
        raise invalid("invalid_manifest", "An artifact hash is invalid.", parameter + ".sha256")
    return Artifact(obj["path"], obj["size_bytes"], obj["sha256"])


def _relative_path(text: str, parameter: str) -> PurePosixPath:
    if not isinstance(text, str) or not text or "\\" in text or text.startswith(("~", "/")) or "$" in text or "%" in text:
        raise invalid("invalid_manifest_path", "A manifest path is not a safe relative path.", parameter)
    win = PureWindowsPath(text)
    path = PurePosixPath(text)
    if win.is_absolute() or win.drive or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise invalid("invalid_manifest_path", "A manifest path is not a safe relative path.", parameter)
    return path


def resolve_local(root: Path, text: str, parameter: str, *, must_exist: bool = True) -> Path:
    relative = _relative_path(text, parameter)
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            try:
                info = current.lstat()
            except OSError as exc:
                raise invalid("invalid_manifest_path", "An artifact path cannot be inspected.", parameter) from exc
            if stat.S_ISLNK(info.st_mode):
                raise invalid("invalid_manifest_path", "Symlink artifact paths are forbidden.", parameter)
            if os.name == "nt" and info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise invalid("invalid_manifest_path", "Windows reparse points are forbidden.", parameter)
    if must_exist and (not current.exists() or not current.is_file()):
        raise invalid("artifact_missing", "A declared artifact is missing.", parameter)
    try:
        current.resolve(strict=must_exist).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise invalid("invalid_manifest_path", "An artifact escapes the manifest root.", parameter) from exc
    return current


def _model(value: Any) -> tuple[dict[str, Any], list[Artifact], int]:
    obj = _exact(value, {"format", "context_window", "artifacts"}, set(), "model")
    if obj["format"] not in {"gguf", "mlx"}:
        raise invalid("invalid_manifest", "The model format is unsupported.", "model.format")
    window = obj["context_window"]
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise invalid("invalid_manifest", "The context window is invalid.", "model.context_window")
    values = obj["artifacts"]
    if not isinstance(values, list) or not values:
        raise invalid("invalid_manifest", "Model artifacts must be a non-empty array.", "model.artifacts")
    artifacts = [_artifact(item, f"model.artifacts.{i}") for i, item in enumerate(values)]
    if [item.path for item in artifacts] != sorted(item.path for item in artifacts):
        raise invalid("invalid_manifest", "Model artifacts must be sorted by path.", "model.artifacts")
    return obj, artifacts, window


def _backend(value: Any, target: str) -> tuple[dict[str, Any], list[Artifact]]:
    if target == "windows-x86_64":
        obj = _exact(value, {"type", "llama_cpp_commit", "build_id", "executable", "dlls", "parameters"}, set(), "backend")
        if obj["type"] != "llama-server" or not isinstance(obj["llama_cpp_commit"], str) or not COMMIT_RE.fullmatch(obj["llama_cpp_commit"]):
            raise invalid("invalid_manifest", "The llama.cpp backend identity is invalid.", "backend")
        if not isinstance(obj["build_id"], str) or not obj["build_id"] or len(obj["build_id"].encode()) > 256:
            raise invalid("invalid_manifest", "The llama.cpp build identifier is invalid.", "backend.build_id")
        executable = _artifact(obj["executable"], "backend.executable")
        dll_values = obj["dlls"]
        if not isinstance(dll_values, list):
            raise invalid("invalid_manifest", "backend.dlls must be an array.", "backend.dlls")
        dlls = [_artifact(item, f"backend.dlls.{i}") for i, item in enumerate(dll_values)]
        names = [PurePosixPath(item.path).name.lower() for item in dlls]
        if names != sorted(names) or len(names) != len(set(names)) or any(not name.endswith(".dll") for name in names):
            raise invalid("invalid_manifest", "Bundled DLLs must be a sorted unique allowlist.", "backend.dlls")
        params = obj["parameters"]
        if not isinstance(params, dict) or set(params) - set(LLAMA_PARAMETER_TYPES):
            raise invalid("invalid_manifest", "The llama-server parameter set is not allowlisted.", "backend.parameters")
        for key, value in params.items():
            expected = LLAMA_PARAMETER_TYPES[key]
            if type(value) is not expected or (expected is int and value < 0):
                raise invalid("invalid_manifest", "A llama-server parameter has an invalid type or value.", f"backend.parameters.{key}")
        return obj, [executable, *dlls]
    obj = _exact(value, {"type", "worker"}, set(), "backend")
    if obj["type"] != "mlx-worker":
        raise invalid("invalid_manifest", "The MLX backend identity is invalid.", "backend.type")
    return obj, [_artifact(obj["worker"], "backend.worker")]


def _python(value: Any, target: str) -> tuple[dict[str, Any], list[Artifact]]:
    if target == "windows-x86_64":
        return _exact(value, set(), set(), "python"), []
    obj = _exact(value, {"implementation", "version", "executable", "site_packages", "packages"}, set(), "python")
    if obj["implementation"] != "CPython" or not isinstance(obj["version"], str):
        raise invalid("invalid_manifest", "The Python runtime identity is invalid.", "python")
    _relative_path(obj["site_packages"], "python.site_packages")
    artifacts = [_artifact(obj["executable"], "python.executable")]
    packages = obj["packages"]
    if not isinstance(packages, list) or not packages:
        raise invalid("invalid_manifest", "Python packages must be a non-empty array.", "python.packages")
    names: list[str] = []
    for i, package in enumerate(packages):
        p = _exact(package, {"name", "version", "wheel_filename", "wheel"}, set(), f"python.packages.{i}")
        if not all(isinstance(p[key], str) and p[key] for key in ("name", "version", "wheel_filename")):
            raise invalid("invalid_manifest", "A Python package identity is invalid.", f"python.packages.{i}")
        canonical_name = re.sub(r"[-_.]+", "-", p["name"]).lower()
        if canonical_name != p["name"]:
            raise invalid("invalid_manifest", "Python package names must be canonical.", f"python.packages.{i}.name")
        names.append(canonical_name)
        wheel = _artifact(p["wheel"], f"python.packages.{i}.wheel")
        if PurePosixPath(wheel.path).name != p["wheel_filename"] or not p["wheel_filename"].endswith(".whl"):
            raise invalid("invalid_manifest", "The declared wheel filename does not match its artifact.", f"python.packages.{i}.wheel_filename")
        artifacts.append(wheel)
    if names != sorted(names) or len(names) != len(set(names)):
        raise invalid("invalid_manifest", "Python packages must be sorted and unique.", "python.packages")
    return obj, artifacts


def load_manifest(path: Path) -> Manifest:
    path = path.absolute()
    raw_bytes = path.read_bytes()
    if len(raw_bytes) > MANIFEST_BYTES:
        raise invalid("manifest_too_large", "The manifest exceeds its size limit.")
    raw = parse_body(raw_bytes)
    top = _exact(
        raw,
        {"manifest_schema_version", "alias", "platform", "backend", "model", "python", "capabilities", "qualification_report"},
        set(),
        "manifest",
    )
    if top["manifest_schema_version"] != 1:
        raise invalid("invalid_manifest", "The manifest schema version is unsupported.", "manifest_schema_version")
    if not isinstance(top["alias"], str) or not ALIAS_RE.fullmatch(top["alias"]):
        raise invalid("invalid_manifest", "The logical alias is invalid.", "alias")
    target = top["platform"]
    if target not in PLATFORMS:
        raise invalid("invalid_manifest", "The target platform is unsupported.", "platform")
    capabilities_raw = _exact(top["capabilities"], {"text", "stream", "tools", "json_object", "json_schema"}, set(), "capabilities")
    if any(type(value) is not bool for value in capabilities_raw.values()) or capabilities_raw["text"] is not True:
        raise invalid("invalid_manifest", "Capability values must be booleans and text must be enabled.", "capabilities")
    backend, backend_artifacts = _backend(top["backend"], target)
    model, model_artifacts, window = _model(top["model"])
    python, python_artifacts = _python(top["python"], target)
    if (target == "windows-x86_64") != (model["format"] == "gguf"):
        raise invalid("invalid_manifest", "The model format does not match the platform.", "model.format")
    root = path.parent.resolve(strict=True)
    report_relative = _relative_path(top["qualification_report"], "qualification_report")
    report = resolve_local(root, top["qualification_report"], "qualification_report", must_exist=False)
    if report.exists() and not report.is_file():
        raise invalid("invalid_manifest_path", "The qualification report path is not a regular file.", "qualification_report")
    artifacts = tuple(backend_artifacts + model_artifacts + python_artifacts)
    paths = [artifact.path for artifact in artifacts]
    if len(paths) != len(set(paths)):
        raise invalid("invalid_manifest", "Artifact paths must be unique.", "manifest")
    for artifact in artifacts:
        resolve_local(root, artifact.path, "artifact")
    return Manifest(
        path, root, top, hashlib.sha256(canonical_bytes(top)).hexdigest(), top["alias"], target,
        backend, model, python, Capabilities(**capabilities_raw), report, artifacts, window,
    )


def hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def verify_artifacts(manifest: Manifest) -> None:
    declared_dlls: set[str] = set()
    executable_dir: Path | None = None
    for artifact in manifest.artifacts:
        local = resolve_local(manifest.root, artifact.path, "artifact")
        size, digest = hash_file(local)
        if size != artifact.size_bytes or digest != artifact.sha256:
            raise ApplianceError(503, "artifact_identity_mismatch", "A declared artifact failed identity verification.")
        if local.suffix.lower() == ".dll":
            declared_dlls.add(local.name.lower())
        if manifest.platform == "windows-x86_64" and artifact.path == manifest.backend["executable"]["path"]:
            executable_dir = local.parent
    if executable_dir is not None:
        actual = {child.name.lower() for child in executable_dir.iterdir() if child.is_file() and child.suffix.lower() == ".dll"}
        if actual != declared_dlls:
            raise ApplianceError(503, "artifact_identity_mismatch", "The bundled DLL allowlist does not match the runtime directory.")
    if manifest.platform == "macos-arm64":
        executable = resolve_local(manifest.root, manifest.python["executable"]["path"], "python.executable")
        if platform.python_implementation() != manifest.python["implementation"] or platform.python_version() != manifest.python["version"] or Path(sys.executable).resolve() != executable.resolve():
            raise ApplianceError(503, "artifact_identity_mismatch", "The active Python runtime does not match the manifest.")
        _verify_python_record_files(manifest)
        _verify_complete_model_tree(manifest)


def _verify_complete_model_tree(manifest: Manifest) -> None:
    declared = {item["path"] for item in manifest.model["artifacts"]}
    parts = [PurePosixPath(path).parts for path in declared]
    if not parts or any(len(item) < 3 or item[:2] != parts[0][:2] for item in parts):
        raise ApplianceError(503, "artifact_identity_mismatch", "MLX model artifacts must share one dedicated model directory.")
    model_root = resolve_local(manifest.root, PurePosixPath(*parts[0][:2]).as_posix(), "model.artifacts", must_exist=False)
    if not model_root.is_dir():
        raise ApplianceError(503, "artifact_identity_mismatch", "The MLX model root is invalid.")
    actual: set[str] = set()
    for path in model_root.rglob("*"):
        if path.is_symlink() or (os.name == "nt" and path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ApplianceError(503, "artifact_identity_mismatch", "The MLX model tree contains an unsafe link.")
        if path.is_file():
            actual.add(path.relative_to(manifest.root).as_posix())
    if actual != declared:
        raise ApplianceError(503, "artifact_identity_mismatch", "The MLX model artifact allowlist is incomplete.")


def assert_runtime_platform(manifest: Manifest) -> None:
    actual = "windows-x86_64" if sys.platform == "win32" and platform.machine().lower() in {"amd64", "x86_64"} else "macos-arm64" if sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"} else "unsupported"
    if actual != manifest.platform:
        raise ApplianceError(503, "platform_identity_mismatch", "The manifest target does not match this native runtime.")


def _verify_python_record_files(manifest: Manifest) -> None:
    site = resolve_local(manifest.root, manifest.python["site_packages"], "python.site_packages", must_exist=False)
    if not site.is_dir():
        raise ApplianceError(503, "artifact_identity_mismatch", "The pinned Python site-packages directory is missing.")
    expected = {item["name"]: item for item in manifest.python["packages"]}
    actual: dict[str, importlib.metadata.Distribution] = {}
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        if name in actual:
            raise ApplianceError(503, "artifact_identity_mismatch", "A Python distribution is duplicated.")
        actual[name] = distribution
    if set(actual) != set(expected):
        raise ApplianceError(503, "artifact_identity_mismatch", "Installed Python distributions do not match the manifest.")
    executable_parts = PurePosixPath(manifest.python["executable"]["path"]).parts
    environment_root = resolve_local(manifest.root, executable_parts[0], "python.environment", must_exist=False)
    if not environment_root.is_dir():
        raise ApplianceError(503, "artifact_identity_mismatch", "The pinned Python environment root is missing.")
    for name, distribution in actual.items():
        if distribution.version != expected[name]["version"]:
            raise ApplianceError(503, "artifact_identity_mismatch", "A Python distribution version does not match the manifest.")
        record = distribution.read_text("RECORD")
        if record is None:
            raise ApplianceError(503, "artifact_identity_mismatch", "A Python distribution has no RECORD.")
        for row in csv.reader(record.splitlines()):
            if len(row) != 3:
                raise ApplianceError(503, "artifact_identity_mismatch", "A RECORD row has an invalid shape.")
            if not row[1]:
                if PurePosixPath(row[0]).name != "RECORD" or row[2]:
                    raise ApplianceError(503, "artifact_identity_mismatch", "A RECORD entry lacks a SHA-256 hash.")
                continue
            algorithm, separator, encoded = row[1].partition("=")
            if separator != "=" or algorithm != "sha256":
                raise ApplianceError(503, "artifact_identity_mismatch", "A RECORD entry lacks a SHA-256 hash.")
            installed = Path(os.path.abspath(distribution.locate_file(row[0])))
            try:
                relative = installed.relative_to(environment_root.absolute())
            except ValueError as exc:
                raise ApplianceError(503, "artifact_identity_mismatch", "A RECORD path escapes the Python environment.") from exc
            current = environment_root
            for part in relative.parts:
                current = current / part
                try:
                    info = current.lstat()
                except OSError as exc:
                    raise ApplianceError(503, "artifact_identity_mismatch", "A RECORD path is missing.") from exc
                if stat.S_ISLNK(info.st_mode) or (os.name == "nt" and info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                    raise ApplianceError(503, "artifact_identity_mismatch", "A RECORD path contains an unsafe link.")
            if not installed.is_file():
                raise ApplianceError(503, "artifact_identity_mismatch", "A RECORD path is not a regular file.")
            size, digest = hash_file(installed)
            expected_digest = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex()
            if digest != expected_digest or (row[2] and size != int(row[2])):
                raise ApplianceError(503, "artifact_identity_mismatch", "An installed Python file failed RECORD verification.")


def runtime_identity(manifest: Manifest, gateway_digest: str, effective_configuration: dict[str, Any]) -> dict[str, Any]:
    uname = platform.uname()
    try:
        gateway_version = importlib.metadata.version("jarvis-inference")
    except importlib.metadata.PackageNotFoundError:
        gateway_version = "source-tree"
    try:
        sdk_version = importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        sdk_version = "missing"
    return {
        "manifest_digest": manifest.digest,
        "gateway_digest": gateway_digest,
        "gateway_runtime": {
            "version": gateway_version,
            "openai_sdk_version": sdk_version,
            "wheel": _gateway_wheel_identity(manifest),
        },
        "effective_configuration": effective_configuration,
        "os": {"system": uname.system, "release": uname.release, "version": uname.version},
        "architecture": platform.machine(),
        "cpu": platform.processor() or uname.processor,
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version()},
        "hardware_runtime": {
            "ram_bytes": _ram_bytes(),
            "gpu_device": os.environ.get("JARVIS_QUAL_GPU_DEVICE", "unknown"),
            "gpu_memory_bytes": _env_int("JARVIS_QUAL_GPU_MEMORY_BYTES"),
            "driver": os.environ.get("JARVIS_QUAL_GPU_DRIVER", "unknown"),
            "compute_runtime": os.environ.get("JARVIS_QUAL_COMPUTE_RUNTIME", "unknown"),
            "compute_capability": os.environ.get("JARVIS_QUAL_COMPUTE_CAPABILITY", "unknown"),
        },
    }


def _gateway_wheel_identity(manifest: Manifest) -> dict[str, Any] | None:
    relative = os.environ.get("JARVIS_GATEWAY_WHEEL", "")
    if not relative:
        return None
    try:
        path = resolve_local(manifest.root, relative, "JARVIS_GATEWAY_WHEEL")
        if path.suffix != ".whl":
            return None
        size, digest = hash_file(path)
        return {"path": relative, "size_bytes": size, "sha256": digest}
    except (ApplianceError, OSError):
        return None


def _ram_bytes() -> int | None:
    try:
        if os.name == "posix":
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    return None


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None
