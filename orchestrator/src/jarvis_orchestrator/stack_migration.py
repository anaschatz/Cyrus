"""One-way preparation of a legacy private stack for Voice authentication.

The migration is intentionally not an in-place upgrade.  It reads and validates
one ``jarvis-private-stack/1`` composition, creates a fresh private directory,
and writes an independently valid ``jarvis-private-stack/2`` composition there.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis_voice.config import voice_config_from_mapping
from jarvis_voice.voice_auth import create_voice_auth_key_file

from .config import BearerTokenFormatError, OrchestratorConfig
from .stack_config import (
    STACK_SCHEMA_VERSION,
    PrivateStackConfig,
    StackConfigError,
)
from .windows_private_file import (
    WindowsPrivateFileError,
    harden_windows_private_path,
    validate_windows_private_descriptor,
    validate_windows_private_path,
)

LEGACY_STACK_SCHEMA_VERSION = "jarvis-private-stack/1"
MIGRATION_RESULT_VERSION = "jarvis-stack-auth-migration/1"
MAX_JSON_BYTES = 4 * 1024 * 1024

VOICE_CONFIG_NAME = "voice.json"
ORCHESTRATOR_CONFIG_NAME = "orchestrator.json"
VOICE_AUTH_KEY_NAME = "voice-auth.json"
STACK_CONFIG_NAME = "jarvis.private.json"


class StackMigrationError(StackConfigError):
    """Content-free failure while preparing an auth-enabled private stack."""


@dataclass(frozen=True, slots=True)
class StackMigrationResult:
    """Path- and secret-free success result for logs and machine output."""

    version: str = MIGRATION_RESULT_VERSION
    status: str = "prepared"
    stack_schema_version: str = STACK_SCHEMA_VERSION

    def canonical(self) -> dict[str, str]:
        return {
            "version": self.version,
            "status": self.status,
            "stack_schema_version": self.stack_schema_version,
        }


@dataclass(frozen=True, slots=True)
class _LegacyStack:
    voice_config_path: Path
    orchestrator_config_path: Path
    runner_path: Path
    manifest_path: Path
    qualification_report_path: Path
    voice_document: dict[str, Any]
    orchestrator_document: dict[str, Any]


class _DuplicateKey(ValueError):
    pass


def _unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise _DuplicateKey
        result[key] = value
    return result


def _selected_absolute_path(value: str | Path, code: str) -> Path:
    if isinstance(value, Path):
        selected = value
    elif isinstance(value, str) and value:
        selected = Path(value)
    else:
        raise StackMigrationError(code)
    if not selected.is_absolute():
        raise StackMigrationError(code)
    return selected


def _private_regular_file(value: str | Path, code: str) -> tuple[Path, os.stat_result]:
    selected = _selected_absolute_path(value, code)
    try:
        metadata = selected.lstat()
    except OSError:
        raise StackMigrationError(code) from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise StackMigrationError(code)
    if os.name == "nt":  # pragma: no cover - native Windows CI
        try:
            validate_windows_private_path(selected)
        except WindowsPrivateFileError:
            raise StackMigrationError("legacy_private_file_permissions") from None
    elif metadata.st_mode & 0o077:
        raise StackMigrationError("legacy_private_file_permissions")
    return selected, metadata


def _read_private_bytes(value: str | Path, code: str) -> tuple[Path, bytes]:
    selected, before = _private_regular_file(value, code)
    if before.st_size < 1 or before.st_size > MAX_JSON_BYTES:
        raise StackMigrationError(code)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(selected, flags)
    except OSError:
        raise StackMigrationError(code) from None
    try:
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(opened.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or (os.name != "nt" and opened.st_mode & 0o077)
        ):
            raise StackMigrationError(code)
        if os.name == "nt":  # pragma: no cover - native Windows CI
            try:
                validate_windows_private_descriptor(descriptor)
            except WindowsPrivateFileError:
                raise StackMigrationError("legacy_private_file_permissions") from None
        chunks: list[bytes] = []
        remaining = MAX_JSON_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
    except StackMigrationError:
        raise
    except OSError:
        raise StackMigrationError(code) from None
    finally:
        os.close(descriptor)
    if not encoded or len(encoded) > MAX_JSON_BYTES:
        raise StackMigrationError(code)
    return selected, encoded


def _strict_private_json(value: str | Path, unavailable_code: str) -> tuple[Path, Any]:
    selected, encoded = _read_private_bytes(value, unavailable_code)
    try:
        text = encoded.decode("utf-8", errors="strict")
        if text.startswith("\ufeff"):
            raise ValueError
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, _DuplicateKey):
        raise StackMigrationError("legacy_json_invalid") from None
    return selected, parsed


def _object(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StackMigrationError(code)
    return value


def _input_path(value: Any, code: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StackMigrationError(code)
    selected, _metadata = _private_regular_file(value, code)
    return selected


def _load_legacy_stack(
    path: str | Path, *, rotate_weak_bearer_tokens: bool
) -> _LegacyStack:
    _source, parsed = _strict_private_json(path, "legacy_stack_unavailable")
    document = _object(parsed, "legacy_stack_schema_invalid")
    if (
        set(document)
        != {
            "schema_version",
            "voice_config_path",
            "orchestrator_config_path",
            "inference",
        }
        or document.get("schema_version") != LEGACY_STACK_SCHEMA_VERSION
    ):
        raise StackMigrationError("legacy_stack_schema_invalid")
    inference = _object(document.get("inference"), "legacy_inference_config_invalid")
    if set(inference) != {
        "runner_path",
        "manifest_path",
        "qualification_report_path",
    }:
        raise StackMigrationError("legacy_inference_config_invalid")

    voice_path = _input_path(
        document.get("voice_config_path"), "legacy_voice_config_unavailable"
    )
    orchestrator_path = _input_path(
        document.get("orchestrator_config_path"),
        "legacy_orchestrator_config_unavailable",
    )
    runner_path = _input_path(
        inference.get("runner_path"), "legacy_inference_runner_unavailable"
    )
    manifest_path = _input_path(
        inference.get("manifest_path"), "legacy_inference_manifest_unavailable"
    )
    qualification_path = _input_path(
        inference.get("qualification_report_path"),
        "legacy_inference_qualification_unavailable",
    )

    _voice_source, voice_value = _strict_private_json(
        voice_path, "legacy_voice_config_unavailable"
    )
    _orchestrator_source, orchestrator_value = _strict_private_json(
        orchestrator_path, "legacy_orchestrator_config_unavailable"
    )
    voice_document = dict(_object(voice_value, "legacy_voice_config_invalid"))
    orchestrator_document = dict(
        _object(orchestrator_value, "legacy_orchestrator_config_invalid")
    )
    if (
        "voice_auth_key_file" in voice_document
        or "voice_auth_key_file" in orchestrator_document
    ):
        raise StackMigrationError("legacy_auth_already_configured")
    try:
        voice_config_from_mapping(voice_document)
    except Exception:  # noqa: BLE001 - normalize untrusted legacy config errors
        raise StackMigrationError("legacy_voice_config_invalid") from None
    try:
        OrchestratorConfig.from_mapping(orchestrator_document)
    except BearerTokenFormatError:
        if not rotate_weak_bearer_tokens:
            raise StackMigrationError("legacy_bearer_token_rotation_required") from None
        orchestrator_document["control_bearer_token"] = secrets.token_urlsafe(48)
        orchestrator_document["inference_bearer_token"] = secrets.token_urlsafe(48)
        if orchestrator_document.get(
            "research_enabled"
        ) is True or orchestrator_document.get("research_bearer_token"):
            orchestrator_document["research_bearer_token"] = secrets.token_urlsafe(48)
        try:
            OrchestratorConfig.from_mapping(orchestrator_document)
        except Exception:  # noqa: BLE001 - keep migration output content-free
            raise StackMigrationError("legacy_orchestrator_config_invalid") from None
    except Exception:  # noqa: BLE001 - normalize untrusted legacy config errors
        raise StackMigrationError("legacy_orchestrator_config_invalid") from None

    # Parse inference evidence strictly here.  PrivateStackConfig performs the
    # semantic alias and digest checks after the new composition is written.
    _strict_private_json(manifest_path, "legacy_inference_manifest_unavailable")
    _strict_private_json(
        qualification_path, "legacy_inference_qualification_unavailable"
    )
    return _LegacyStack(
        voice_config_path=voice_path,
        orchestrator_config_path=orchestrator_path,
        runner_path=runner_path,
        manifest_path=manifest_path,
        qualification_report_path=qualification_path,
        voice_document=voice_document,
        orchestrator_document=orchestrator_document,
    )


def _validate_output_parent(output_directory: Path) -> None:
    try:
        parent = output_directory.parent.lstat()
    except OSError:
        raise StackMigrationError("migration_output_parent_invalid") from None
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        raise StackMigrationError("migration_output_parent_invalid")
    if os.name == "nt":  # pragma: no cover - native Windows CI
        try:
            validate_windows_private_path(output_directory.parent)
        except WindowsPrivateFileError:
            raise StackMigrationError("migration_output_parent_invalid") from None


def _create_output_directory(output_directory: Path) -> tuple[int, int]:
    _validate_output_parent(output_directory)
    try:
        os.mkdir(output_directory, 0o700)
    except FileExistsError:
        raise StackMigrationError("migration_output_not_fresh") from None
    except OSError:
        raise StackMigrationError("migration_output_create_failed") from None
    try:
        if os.name == "nt":  # pragma: no cover - native Windows CI
            try:
                harden_windows_private_path(output_directory)
            except WindowsPrivateFileError:
                raise StackMigrationError("migration_output_create_failed") from None
        else:
            output_directory.chmod(0o700)
        metadata = output_directory.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or (os.name != "nt" and metadata.st_mode & 0o077)
        ):
            raise StackMigrationError("migration_output_create_failed")
        return metadata.st_dev, metadata.st_ino
    except BaseException:
        try:
            shutil.rmtree(output_directory)
        except OSError:
            pass
        raise


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError):
        raise StackMigrationError("migration_document_invalid") from None


def _create_private_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = _canonical_json(value)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(path, flags, 0o600)
        created = True
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or (
            os.name != "nt" and metadata.st_mode & 0o077
        ):
            raise OSError
    except OSError:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise StackMigrationError("migration_output_write_failed") from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                if created:
                    try:
                        path.unlink()
                    except OSError:
                        pass
                raise StackMigrationError("migration_output_write_failed") from None
    if os.name == "nt":  # pragma: no cover - native Windows CI
        try:
            harden_windows_private_path(path)
        except WindowsPrivateFileError:
            try:
                path.unlink()
            except OSError:
                pass
            raise StackMigrationError("migration_output_write_failed") from None


def _rollback(output_directory: Path, identity: tuple[int, int]) -> None:
    try:
        metadata = output_directory.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != identity
        ):
            raise OSError
        shutil.rmtree(output_directory)
    except OSError:
        raise StackMigrationError("migration_rollback_failed") from None


def prepare_auth_v2_stack(
    legacy_manifest_path: str | Path,
    output_directory: str | Path,
    *,
    macos_ptt_key: str | None = None,
    safe_beta_scope: bool = False,
    rotate_weak_bearer_tokens: bool = False,
) -> StackMigrationResult:
    """Create a new auth-enabled v2 stack without mutating any legacy input.

    ``output_directory`` must be an absolute path that does not exist.  On any
    failure after its exclusive creation, only that newly-created directory is
    removed.
    """

    if not isinstance(rotate_weak_bearer_tokens, bool):
        raise StackMigrationError("migration_bearer_rotation_flag_invalid")
    legacy = _load_legacy_stack(
        legacy_manifest_path,
        rotate_weak_bearer_tokens=rotate_weak_bearer_tokens,
    )
    if macos_ptt_key not in {None, "CONTROL_OPTION", "RIGHT_OPTION"}:
        raise StackMigrationError("migration_macos_ptt_key_invalid")
    if not isinstance(safe_beta_scope, bool):
        raise StackMigrationError("migration_safe_beta_scope_invalid")
    destination = _selected_absolute_path(
        output_directory, "migration_output_path_invalid"
    )
    identity: tuple[int, int] | None = None
    try:
        identity = _create_output_directory(destination)
        auth_key_path = destination / VOICE_AUTH_KEY_NAME
        voice_path = destination / VOICE_CONFIG_NAME
        orchestrator_path = destination / ORCHESTRATOR_CONFIG_NAME
        stack_path = destination / STACK_CONFIG_NAME

        create_voice_auth_key_file(auth_key_path)
        voice_document = dict(legacy.voice_document)
        voice_document["voice_auth_key_file"] = str(auth_key_path)
        if macos_ptt_key is not None:
            voice_document["ptt_key"] = macos_ptt_key
        orchestrator_document = dict(legacy.orchestrator_document)
        orchestrator_document["voice_auth_key_file"] = str(auth_key_path)
        if safe_beta_scope:
            orchestrator_document.update(
                {
                    "action_host_qualification_file": None,
                    "action_host_qualification_sha256": "",
                    "action_host_socket_path": None,
                    "action_host_token_file": None,
                    "cloud_enabled": False,
                    "cloud_qualification_file": None,
                    "cloud_release_qualification_file": None,
                    "cloud_release_qualification_sha256": "",
                    "cloud_secret_file": None,
                    "cloud_terra_enabled": False,
                    "memory_shadow_enabled": False,
                    "research_browser_fallback": False,
                    "research_browser_qualification_sha256": "",
                    "research_bearer_token": "",
                    "research_enabled": False,
                    "research_language_detector_sha256": "",
                    "research_language_detector_version": "disabled",
                    "research_model_sha256": "",
                    "research_profile_sha256": "",
                    "research_qualification_sha256": "",
                    "scheduler_compiler_qualification_file": None,
                    "scheduler_compiler_qualification_sha256": "",
                    "scheduler_contract_sha256": "",
                    "scheduler_qualification_manifest_sha256": "",
                    "tool_app_ids": [],
                    "tool_policy_sha256": "",
                    "tool_roots": None,
                    "tools_enabled": False,
                    "voice_scheduler_callback_qualification_sha256": "",
                    "voice_scheduler_compatibility_sha256": "",
                }
            )
        try:
            voice_config_from_mapping(voice_document)
            OrchestratorConfig.from_mapping(orchestrator_document)
        except Exception:  # noqa: BLE001 - keep migration output content-free
            raise StackMigrationError("migration_component_config_invalid") from None
        _create_private_json(voice_path, voice_document)
        _create_private_json(orchestrator_path, orchestrator_document)
        _create_private_json(
            stack_path,
            {
                "schema_version": STACK_SCHEMA_VERSION,
                "voice_config_path": str(voice_path),
                "orchestrator_config_path": str(orchestrator_path),
                "inference": {
                    "runner_path": str(legacy.runner_path),
                    "manifest_path": str(legacy.manifest_path),
                    "qualification_report_path": str(legacy.qualification_report_path),
                },
            },
        )
        try:
            PrivateStackConfig.load(stack_path)
        except Exception:  # noqa: BLE001 - keep validation output content-free
            raise StackMigrationError("migration_validation_failed") from None
        return StackMigrationResult()
    except BaseException as exc:
        if identity is not None:
            _rollback(destination, identity)
        if isinstance(exc, StackMigrationError):
            raise
        raise StackMigrationError("stack_migration_failed") from None


__all__ = [
    "LEGACY_STACK_SCHEMA_VERSION",
    "MIGRATION_RESULT_VERSION",
    "ORCHESTRATOR_CONFIG_NAME",
    "STACK_CONFIG_NAME",
    "VOICE_AUTH_KEY_NAME",
    "VOICE_CONFIG_NAME",
    "StackMigrationError",
    "StackMigrationResult",
    "prepare_auth_v2_stack",
]
