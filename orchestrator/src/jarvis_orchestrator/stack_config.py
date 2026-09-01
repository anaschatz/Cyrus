"""Strict private composition manifest for the real local CYRUS stack."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .config import OrchestratorConfig
from .windows_private_file import (
    WindowsPrivateFileError,
    harden_windows_private_path,
    validate_windows_private_descriptor,
    validate_windows_private_path,
)


STACK_SCHEMA_VERSION = "jarvis-private-stack/2"


class StackConfigError(ValueError):
    """Content-free private stack configuration failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _object(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StackConfigError(code)
    return value


def _absolute_file(value: Any, code: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StackConfigError(code)
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise StackConfigError(code)
    # Do not use resolve()/is_file() here: both follow a configured symlink and
    # would hide the fact that the operator selected a mutable indirection.
    path = Path(os.path.abspath(path))
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise StackConfigError(code) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise StackConfigError(code)
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise StackConfigError(code) from exc


def _private_mode(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise StackConfigError("private_file_permissions") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise StackConfigError("private_file_permissions")
    platform_name = _platform_name()
    if platform_name == "posix":
        if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise StackConfigError("private_file_permissions")
        return
    if platform_name == "nt":
        try:
            validate_windows_private_path(path)
        except WindowsPrivateFileError:
            raise StackConfigError("private_file_permissions") from None
        return
    raise StackConfigError("private_file_permissions")


def _platform_name() -> str:
    """Return the filesystem security family; isolated for cross-OS tests."""

    return os.name


def _validate_windows_stack_parent(path: Path) -> None:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise WindowsPrivateFileError
        validate_windows_private_path(path)
    except (OSError, WindowsPrivateFileError):
        raise StackConfigError("stack_output_permissions") from None


def _secure_new_stack_descriptor(descriptor: int, temporary: Path) -> None:
    platform_name = _platform_name()
    if platform_name == "posix":
        os.fchmod(descriptor, 0o600)
        return
    if platform_name == "nt":
        try:
            harden_windows_private_path(temporary)
            validate_windows_private_descriptor(descriptor)
        except WindowsPrivateFileError:
            raise StackConfigError("stack_output_permissions") from None
        return
    raise StackConfigError("stack_output_permissions")


def _revalidate_stack_descriptor(descriptor: int) -> None:
    if _platform_name() != "nt":
        return
    try:
        validate_windows_private_descriptor(descriptor)
    except WindowsPrivateFileError:
        raise StackConfigError("stack_output_permissions") from None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise StackConfigError("component_identity_unavailable") from exc
    return digest.hexdigest()


def _canonical_document(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class InferenceStackConfig:
    runner_path: Path
    manifest_path: Path
    qualification_report_path: Path


@dataclass(frozen=True, slots=True)
class PrivateStackConfig:
    source_path: Path
    voice_config_path: Path
    orchestrator_config_path: Path
    inference: InferenceStackConfig

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, source_path: str | Path
    ) -> "PrivateStackConfig":
        expected = {
            "schema_version",
            "voice_config_path",
            "orchestrator_config_path",
            "inference",
        }
        if set(value) != expected or value.get("schema_version") != STACK_SCHEMA_VERSION:
            raise StackConfigError("stack_schema_invalid")
        inference = _object(value["inference"], "inference_config_invalid")
        if set(inference) != {
            "runner_path",
            "manifest_path",
            "qualification_report_path",
        }:
            raise StackConfigError("inference_config_invalid")
        source = _absolute_file(str(Path(source_path).expanduser()), "stack_config_unavailable")
        result = cls(
            source_path=source,
            voice_config_path=_absolute_file(
                value["voice_config_path"], "voice_config_unavailable"
            ),
            orchestrator_config_path=_absolute_file(
                value["orchestrator_config_path"],
                "orchestrator_config_unavailable",
            ),
            inference=InferenceStackConfig(
                runner_path=_absolute_file(
                    inference["runner_path"], "inference_runner_unavailable"
                ),
                manifest_path=_absolute_file(
                    inference["manifest_path"], "inference_manifest_unavailable"
                ),
                qualification_report_path=_absolute_file(
                    inference["qualification_report_path"],
                    "inference_qualification_unavailable",
                ),
            ),
        )
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path) -> "PrivateStackConfig":
        selected = Path(path).expanduser()
        try:
            raw = selected.read_text(encoding="utf-8")
            if raw.startswith("\ufeff"):
                raise StackConfigError("stack_json_invalid")
            def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for key, item in items:
                    if key in result:
                        raise ValueError("duplicate key")
                    result[key] = item
                return result

            parsed = json.loads(
                raw,
                object_pairs_hook=pairs,
                parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
            )
        except StackConfigError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise StackConfigError("stack_json_invalid") from exc
        return cls.from_mapping(_object(parsed, "stack_schema_invalid"), source_path=selected)

    def validate(self) -> None:
        for path in (
            self.source_path,
            self.voice_config_path,
            self.orchestrator_config_path,
            self.inference.runner_path,
            self.inference.manifest_path,
            self.inference.qualification_report_path,
        ):
            _private_mode(path)
        try:
            from jarvis_voice.config import load_voice_config

            voice = load_voice_config(self.voice_config_path)
            orchestrator = OrchestratorConfig.load(self.orchestrator_config_path)
            manifest = json.loads(
                self.inference.manifest_path.read_text(encoding="utf-8")
            )
            report = json.loads(
                self.inference.qualification_report_path.read_text(encoding="utf-8")
            )
        except StackConfigError:
            raise
        except Exception as exc:
            raise StackConfigError("component_config_invalid") from exc
        expected_ws = f"ws://{orchestrator.voice_host}:{orchestrator.voice_port}"
        if voice.orchestrator_url != expected_ws:
            raise StackConfigError("voice_orchestrator_mismatch")
        if voice.voice_auth_key_file is None or orchestrator.voice_auth_key_file is None:
            raise StackConfigError("voice_auth_key_required")
        try:
            from jarvis_voice.voice_auth import load_voice_auth_key

            voice_key_path = Path(voice.voice_auth_key_file).absolute()
            orchestrator_key_path = orchestrator.voice_auth_key_file.absolute()
            voice_key = load_voice_auth_key(voice_key_path)
            orchestrator_key = load_voice_auth_key(orchestrator_key_path)
        except StackConfigError:
            raise
        except Exception as exc:
            raise StackConfigError("voice_auth_key_invalid") from exc
        if (
            voice_key_path != orchestrator_key_path
            or voice_key.key_version != orchestrator_key.key_version
            or not hmac.compare_digest(voice_key.key, orchestrator_key.key)
        ):
            raise StackConfigError("voice_auth_key_mismatch")
        inference_url = urlparse(orchestrator.inference_base_url)
        if (
            inference_url.scheme != "http"
            or inference_url.hostname != "127.0.0.1"
            or (inference_url.port or 80) != 8766
            or inference_url.path.rstrip("/") != "/v1"
        ):
            raise StackConfigError("inference_endpoint_mismatch")
        if not isinstance(manifest, Mapping) or manifest.get("alias") != orchestrator.model_alias:
            raise StackConfigError("inference_alias_mismatch")
        identity = report.get("identity") if isinstance(report, Mapping) else None
        digest = hashlib.sha256(_canonical_document(manifest)).hexdigest()
        if (
            report.get("status") != "pass"
            or not isinstance(identity, Mapping)
            or identity.get("manifest_digest") != digest
        ):
            raise StackConfigError("inference_qualification_mismatch")

    def canonical(self) -> dict[str, Any]:
        return {
            "schema_version": STACK_SCHEMA_VERSION,
            "voice_config_path": str(self.voice_config_path),
            "orchestrator_config_path": str(self.orchestrator_config_path),
            "inference": {
                "runner_path": str(self.inference.runner_path),
                "manifest_path": str(self.inference.manifest_path),
                "qualification_report_path": str(
                    self.inference.qualification_report_path
                ),
            },
        }

    @property
    def sha256(self) -> str:
        """Return the deployment identity, not merely a hash of path strings."""

        return self.deployment_identity()["deployment_sha256"]

    def deployment_identity(self) -> dict[str, Any]:
        """Build a content-bound, secret-free identity for this composition.

        The authentication-key commitment is an HMAC over public component
        digests. It changes after key rotation without publishing a reusable
        hash of the key itself.
        """

        try:
            from jarvis_voice.voice_auth import load_voice_auth_key

            orchestrator = OrchestratorConfig.load(self.orchestrator_config_path)
            if orchestrator.voice_auth_key_file is None:
                raise StackConfigError("voice_auth_key_required")
            auth_key = load_voice_auth_key(orchestrator.voice_auth_key_file)
        except StackConfigError:
            raise
        except Exception as exc:
            raise StackConfigError("component_identity_unavailable") from exc
        components = {
            "stack_manifest_sha256": _sha256_file(self.source_path),
            "voice_config_sha256": _sha256_file(self.voice_config_path),
            "orchestrator_config_sha256": _sha256_file(
                self.orchestrator_config_path
            ),
            "inference_runner_sha256": _sha256_file(self.inference.runner_path),
            "inference_manifest_sha256": _sha256_file(
                self.inference.manifest_path
            ),
            "inference_qualification_sha256": _sha256_file(
                self.inference.qualification_report_path
            ),
        }
        commitment_payload = _canonical_document(
            {
                "components": components,
                "domain": "jarvis-private-stack-auth-binding/1",
                "key_version": auth_key.key_version,
            }
        )
        auth_binding = hmac.new(
            auth_key.key,
            commitment_payload,
            hashlib.sha256,
        ).hexdigest()
        core = {
            "auth_binding": auth_binding,
            "auth_key_version": auth_key.key_version,
            "components": components,
            "schema_version": "jarvis-private-stack-deployment/1",
        }
        return {
            **core,
            "deployment_sha256": hashlib.sha256(
                _canonical_document(core)
            ).hexdigest(),
        }

    def expected_inference_runtime_identity(self) -> tuple[str, str]:
        """Return the manifest and qualified runtime-identity digests."""

        try:
            report = json.loads(
                self.inference.qualification_report_path.read_text(
                    encoding="utf-8"
                )
            )
            identity = report.get("identity")
            if not isinstance(identity, Mapping):
                raise ValueError
            manifest_digest = identity.get("manifest_digest")
            if (
                not isinstance(manifest_digest, str)
                or len(manifest_digest) != 64
            ):
                raise ValueError
            runtime_digest = hashlib.sha256(
                _canonical_document(identity)
            ).hexdigest()
            return manifest_digest, runtime_digest
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise StackConfigError("component_identity_unavailable") from exc


def write_private_stack_config(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically create an owner-private manifest after strict validation."""

    selected = Path(path).expanduser()
    if not selected.is_absolute():
        raise StackConfigError("stack_output_invalid")
    encoded = _canonical_document(value) + b"\n"
    descriptor: int | None = None
    temporary: Path | None = None
    published = False
    complete = False
    try:
        selected.parent.mkdir(parents=True, exist_ok=True)
        if _platform_name() == "nt":
            _validate_windows_stack_parent(selected.parent)
        try:
            selected.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise StackConfigError("stack_output_invalid") from exc
        else:
            raise StackConfigError("stack_output_invalid")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{selected.name}.", dir=selected.parent
        )
        temporary = Path(temporary_name)
        _secure_new_stack_descriptor(descriptor, temporary)
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = None
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
            _revalidate_stack_descriptor(destination.fileno())
        PrivateStackConfig.load(temporary)
        try:
            os.link(temporary, selected, follow_symlinks=False)
        except FileExistsError as exc:
            raise StackConfigError("stack_output_invalid") from exc
        published = True
        temporary.unlink()
        temporary = None
        if _platform_name() == "nt":
            try:
                validate_windows_private_path(selected)
            except WindowsPrivateFileError:
                raise StackConfigError("stack_output_permissions") from None
        PrivateStackConfig.load(selected)
        complete = True
    except StackConfigError:
        raise
    except OSError as exc:
        raise StackConfigError("stack_output_write_failed") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if published and not complete:
            try:
                selected.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = [
    "InferenceStackConfig",
    "PrivateStackConfig",
    "STACK_SCHEMA_VERSION",
    "StackConfigError",
    "write_private_stack_config",
]
