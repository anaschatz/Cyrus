"""Strict, explicit Orchestrator configuration."""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .memory_release import MEMORY_RELEASE_POLICY_VERSION
from .windows_private_file import (
    WindowsPrivateFileError,
    validate_windows_private_descriptor,
)

_MAX_CONFIGURATION_BYTES = 256 * 1024
_MIN_BEARER_TOKEN_DECODED_BYTES = 32
_MAX_BEARER_TOKEN_DECODED_BYTES = 384
_MAX_BEARER_TOKEN_ENCODED_BYTES = 512
_JSON_WHITESPACE = " \t\r\n"


class BearerTokenFormatError(ValueError):
    """A content-free canonical bearer-token format violation."""

    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(f"{field} must be a canonical bounded bearer token")


def _is_windows_reparse(metadata: os.stat_result) -> bool:
    return bool(
        os.name == "nt"
        and getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _file_snapshot(metadata: os.stat_result) -> tuple[int, ...]:
    return tuple(
        int(getattr(metadata, field, 0))
        for field in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_file_attributes",
        )
    )


def _validate_opened_configuration(metadata: os.stat_result, descriptor: int) -> None:
    if not stat.S_ISREG(metadata.st_mode) or _is_windows_reparse(metadata):
        raise ValueError("configuration file must be a private regular file")
    if os.name == "posix":
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) not in {
            0o400,
            0o600,
        }:
            raise ValueError("configuration file must be a private regular file")
        return
    if os.name == "nt":
        try:
            validate_windows_private_descriptor(descriptor)
        except WindowsPrivateFileError:
            raise ValueError(
                "configuration file must be a private regular file"
            ) from None
        return
    raise ValueError("configuration file platform is unsupported")


def _read_private_configuration(path: str | Path) -> bytes:
    """Read one immutable snapshot of a private regular configuration file."""

    selected = os.fspath(Path(path).expanduser())
    try:
        lexical = os.lstat(selected)
    except OSError:
        raise ValueError("configuration file must be a private regular file") from None
    if (
        stat.S_ISLNK(lexical.st_mode)
        or _is_windows_reparse(lexical)
        or not stat.S_ISREG(lexical.st_mode)
    ):
        raise ValueError("configuration file must be a private regular file")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(selected, flags)
    except OSError:
        raise ValueError("configuration file must be a private regular file") from None
    try:
        before = os.fstat(descriptor)
        path_after_open = os.lstat(selected)
        if (
            (before.st_dev, before.st_ino) != (lexical.st_dev, lexical.st_ino)
            or stat.S_ISLNK(path_after_open.st_mode)
            or _is_windows_reparse(path_after_open)
            or not stat.S_ISREG(path_after_open.st_mode)
            or (path_after_open.st_dev, path_after_open.st_ino)
            != (before.st_dev, before.st_ino)
            or _file_snapshot(before) != _file_snapshot(os.fstat(descriptor))
        ):
            raise ValueError("configuration file changed while opening")
        _validate_opened_configuration(before, descriptor)
        if before.st_size < 0 or before.st_size > _MAX_CONFIGURATION_BYTES:
            raise ValueError("configuration file exceeds maximum size")
        raw = bytearray()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise ValueError("configuration file changed while reading")
            raw.extend(chunk)
            remaining -= len(chunk)
        extra = os.read(descriptor, 1)
        after = os.fstat(descriptor)
        path_after_read = os.lstat(selected)
        if (
            extra
            or len(raw) != before.st_size
            or _file_snapshot(after) != _file_snapshot(before)
            or stat.S_ISLNK(path_after_read.st_mode)
            or _is_windows_reparse(path_after_read)
            or not stat.S_ISREG(path_after_read.st_mode)
            or (path_after_read.st_dev, path_after_read.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise ValueError("configuration file changed while reading")
        _validate_opened_configuration(after, descriptor)
        return bytes(raw)
    except OSError:
        raise ValueError("configuration file could not be read securely") from None
    finally:
        os.close(descriptor)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("configuration contains a duplicate object key")
        result[key] = value
    return result


def _reject_non_finite_number(_value: str) -> None:
    raise ValueError("configuration contains a non-finite number")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("configuration contains a non-finite number")
    return parsed


def _parse_strict_json(raw: bytes) -> Any:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("configuration must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ValueError("configuration must be strict UTF-8") from None
    start = len(text) - len(text.lstrip(_JSON_WHITESPACE))
    decoder = json.JSONDecoder(
        object_pairs_hook=_unique_object,
        parse_constant=_reject_non_finite_number,
        parse_float=_parse_finite_float,
    )
    try:
        parsed, end = decoder.raw_decode(text, start)
    except json.JSONDecodeError:
        raise ValueError("configuration must contain exactly one JSON value") from None
    except RecursionError:
        raise ValueError("configuration JSON nesting is invalid") from None
    if text[end:].strip(_JSON_WHITESPACE):
        raise ValueError("configuration contains unexpected trailing data")
    return parsed


def _validate_canonical_bearer_token(field: str, token: str) -> None:
    try:
        encoded = token.encode("ascii", errors="strict")
    except (AttributeError, UnicodeEncodeError):
        raise BearerTokenFormatError(field) from None
    if (
        not encoded
        or len(encoded) > _MAX_BEARER_TOKEN_ENCODED_BYTES
        or b"=" in encoded
        or len(encoded) % 4 == 1
    ):
        raise BearerTokenFormatError(field)
    try:
        decoded = base64.b64decode(
            encoded + b"=" * (-len(encoded) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError):
        raise BearerTokenFormatError(field) from None
    if (
        not _MIN_BEARER_TOKEN_DECODED_BYTES
        <= len(decoded)
        <= _MAX_BEARER_TOKEN_DECODED_BYTES
        or base64.urlsafe_b64encode(decoded).rstrip(b"=") != encoded
    ):
        raise BearerTokenFormatError(field)


def _validate_loaded_bearer_tokens(config: OrchestratorConfig) -> None:
    _validate_canonical_bearer_token(
        "control_bearer_token", config.control_bearer_token
    )
    _validate_canonical_bearer_token(
        "inference_bearer_token", config.inference_bearer_token
    )
    if config.research_enabled or config.research_bearer_token:
        _validate_canonical_bearer_token(
            "research_bearer_token", config.research_bearer_token
        )


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    database_path: Path
    voice_host: str = "127.0.0.1"
    voice_port: int = 8765
    voice_auth_key_file: Path | None = None
    control_host: str = "127.0.0.1"
    control_port: int = 8767
    control_bearer_token: str = dataclass_field(default="", repr=False)
    inference_base_url: str = "http://127.0.0.1:8766/v1"
    inference_bearer_token: str = dataclass_field(default="", repr=False)
    model_alias: str = "jarvis-local"
    readiness_signing_key_file: Path | None = None
    readiness_trusted_public_key_file: Path | None = None
    readiness_signer_key_id: str = ""
    readiness_trusted_public_key_sha256: str = ""
    readiness_build_artifact_file: Path | None = None
    readiness_build_artifact_sha256: str = ""
    readiness_model_artifact_file: Path | None = None
    readiness_model_artifact_sha256: str = ""
    readiness_inference_manifest_sha256: str = ""
    readiness_inference_runtime_identity_sha256: str = ""
    model_timeout_seconds: float = 120.0
    default_speech_locale: str = "el-GR"
    active_budget_seconds: float = 120.0
    absolute_lifetime_seconds: float = 900.0
    approval_ttl_seconds: float = 600.0
    reconnect_ttl_seconds: float = 30.0
    delivery_ttl_seconds: float = 30.0
    model_request_limit: int = 4
    output_token_limit: int = 2048
    final_response_output_tokens: int = 256
    memory_enabled: bool = False
    memory_timeout_seconds: float = 0.25
    memory_recall_limit: int = 5
    memory_shadow_enabled: bool = False
    memory_shadow_mode: str = "structured_fallback"
    memory_shadow_timeout_seconds: float = 0.25
    memory_release_policy: str = MEMORY_RELEASE_POLICY_VERSION
    research_enabled: bool = False
    research_base_url: str = "http://127.0.0.1:8770/v1"
    research_bearer_token: str = dataclass_field(default="", repr=False)
    research_timeout_seconds: float = 45.0
    research_browser_fallback: bool = False
    research_language_detector_version: str = "disabled"
    research_language_detector_sha256: str = ""
    research_authority_registry_version: str = "authority-registry/1"
    research_timezone: str = "Europe/Athens"
    research_model_sha256: str = ""
    research_profile_sha256: str = ""
    research_qualification_sha256: str = ""
    research_browser_qualification_sha256: str = ""
    cloud_enabled: bool = False
    cloud_secret_file: Path | None = None
    cloud_qualification_file: Path | None = None
    cloud_release_qualification_file: Path | None = None
    cloud_release_qualification_sha256: str = ""
    cloud_terra_enabled: bool = False
    cloud_timeout_seconds: float = 60.0
    cloud_monthly_cap_microusd: int = 5_000_000
    scheduler_compiler_qualification_file: Path | None = None
    scheduler_compiler_qualification_sha256: str = ""
    scheduler_default_timezone: str = "Europe/Athens"
    scheduler_default_locale: str = "el-GR"
    voice_scheduler_compatibility_sha256: str = ""
    voice_scheduler_callback_qualification_sha256: str = ""
    tools_enabled: bool = False
    tool_roots: Mapping[str, str] | None = None
    tool_app_ids: tuple[str, ...] = ()
    tool_policy_sha256: str = ""
    scheduler_contract_sha256: str = ""
    scheduler_qualification_manifest_sha256: str = ""
    action_host_socket_path: Path | None = None
    action_host_token_file: Path | None = None
    action_host_qualification_file: Path | None = None
    action_host_qualification_sha256: str = ""
    max_model_requests_per_tool_task: int = 6
    max_tool_calls_per_tool_task: int = 4
    max_tool_continuations_per_tool_task: int = 4

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> OrchestratorConfig:
        """Build a runtime config and enforce the production secret policy."""

        return cls._from_mapping(value, enforce_bearer_policy=True)

    @classmethod
    def from_test_mapping(cls, value: Mapping[str, Any]) -> OrchestratorConfig:
        """Build an offline test/evaluation fixture; never use for runtime input."""

        return cls._from_mapping(value, enforce_bearer_policy=False)

    @classmethod
    def _from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        enforce_bearer_policy: bool,
    ) -> OrchestratorConfig:
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError("unknown configuration field")
        data = dict(value)
        if "database_path" not in data:
            raise ValueError("database_path is required")
        data["database_path"] = Path(str(data["database_path"])).expanduser()
        for field in (
            "voice_auth_key_file",
            "readiness_signing_key_file",
            "readiness_trusted_public_key_file",
            "readiness_build_artifact_file",
            "readiness_model_artifact_file",
            "cloud_secret_file",
            "cloud_qualification_file",
            "cloud_release_qualification_file",
            "scheduler_compiler_qualification_file",
            "action_host_socket_path",
            "action_host_token_file",
            "action_host_qualification_file",
        ):
            if data.get(field) is not None:
                data[field] = Path(str(data[field])).expanduser()
        if "tool_app_ids" in data:
            if not isinstance(data["tool_app_ids"], list):
                raise ValueError("tool_app_ids must be an array")
            data["tool_app_ids"] = tuple(data["tool_app_ids"])
        if data.get("tool_roots") is not None:
            if not isinstance(data["tool_roots"], Mapping):
                raise ValueError("tool_roots must be an object")
            data["tool_roots"] = {
                str(key): str(Path(str(path)).expanduser())
                for key, path in data["tool_roots"].items()
            }
        config = cls(**data)
        if enforce_bearer_policy:
            _validate_loaded_bearer_tokens(config)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> OrchestratorConfig:
        parsed = _parse_strict_json(_read_private_configuration(path))
        if not isinstance(parsed, dict):
            raise ValueError("configuration must be an object")
        return cls.from_mapping(parsed)

    def validate(self) -> None:
        if not self.database_path.is_absolute():
            raise ValueError("database_path must be absolute")
        for field in ("voice_host", "control_host"):
            if ipaddress.ip_address(getattr(self, field)) != ipaddress.ip_address("127.0.0.1"):
                raise ValueError(f"{field} must be 127.0.0.1")
        if self.voice_auth_key_file is not None and not self.voice_auth_key_file.is_absolute():
            raise ValueError("voice_auth_key_file must be absolute")
        parsed = urlparse(self.inference_base_url)
        if parsed.scheme != "http" or parsed.hostname is None:
            raise ValueError("inference_base_url is invalid")
        if parsed.path.rstrip("/") != "/v1" or parsed.params or parsed.query or parsed.fragment or parsed.username is not None:
            raise ValueError("inference_base_url must be the fixed /v1 endpoint")
        if not ipaddress.ip_address(parsed.hostname).is_loopback:
            raise ValueError("inference_base_url must be loopback")
        if self.model_alias != "jarvis-local":
            raise ValueError("model_alias must be jarvis-local")
        readiness_fields = (
            self.readiness_signing_key_file,
            self.readiness_trusted_public_key_file,
            self.readiness_signer_key_id,
            self.readiness_trusted_public_key_sha256,
            self.readiness_build_artifact_file,
            self.readiness_build_artifact_sha256,
            self.readiness_model_artifact_file,
            self.readiness_model_artifact_sha256,
            self.readiness_inference_manifest_sha256,
            self.readiness_inference_runtime_identity_sha256,
        )
        if any(readiness_fields):
            if not all(readiness_fields):
                raise ValueError("readiness evidence configuration is incomplete")
            for field in (
                "readiness_signing_key_file",
                "readiness_trusted_public_key_file",
                "readiness_build_artifact_file",
                "readiness_model_artifact_file",
            ):
                path = getattr(self, field)
                if not isinstance(path, Path) or not path.is_absolute():
                    raise ValueError("readiness evidence paths must be absolute")
            if (
                not isinstance(self.readiness_signer_key_id, str)
                or not self.readiness_signer_key_id.isascii()
                or not 1 <= len(self.readiness_signer_key_id) <= 64
                or self.readiness_signer_key_id[0].isalnum() is False
                or any(
                    character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-"
                    for character in self.readiness_signer_key_id
                )
            ):
                raise ValueError("readiness signer key id is invalid")
            for field in (
                "readiness_trusted_public_key_sha256",
                "readiness_build_artifact_sha256",
                "readiness_model_artifact_sha256",
                "readiness_inference_manifest_sha256",
                "readiness_inference_runtime_identity_sha256",
            ):
                digest = getattr(self, field)
                if len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest
                ):
                    raise ValueError("readiness evidence digest is invalid")
        if (
            not isinstance(self.control_bearer_token, str)
            or not self.control_bearer_token
            or len(self.control_bearer_token.encode("utf-8"))
            > _MAX_BEARER_TOKEN_ENCODED_BYTES
        ):
            raise ValueError("a bounded control bearer token is required")
        if (
            not isinstance(self.inference_bearer_token, str)
            or not self.inference_bearer_token
            or len(self.inference_bearer_token.encode("utf-8"))
            > _MAX_BEARER_TOKEN_ENCODED_BYTES
        ):
            raise ValueError("an inference bearer token is required")
        positive = (
            self.voice_port,
            self.control_port,
            self.model_timeout_seconds,
            self.active_budget_seconds,
            self.absolute_lifetime_seconds,
            self.approval_ttl_seconds,
            self.reconnect_ttl_seconds,
            self.delivery_ttl_seconds,
            self.model_request_limit,
            self.output_token_limit,
            self.memory_timeout_seconds,
            self.memory_recall_limit,
            self.memory_shadow_timeout_seconds,
            self.research_timeout_seconds,
            self.cloud_timeout_seconds,
        )
        if any(isinstance(item, bool) or item <= 0 for item in positive):
            raise ValueError("numeric configuration values must be positive")
        if (
            isinstance(self.final_response_output_tokens, bool)
            or not isinstance(self.final_response_output_tokens, int)
            or not 32 <= self.final_response_output_tokens <= 1024
        ):
            raise ValueError(
                "final_response_output_tokens must be an integer from 32 to 1024"
            )
        if not isinstance(self.memory_shadow_enabled, bool):
            raise ValueError("memory_shadow_enabled must be boolean")
        if self.memory_shadow_mode not in {
            "structured_fallback",
            "deterministic_only",
        }:
            raise ValueError("memory_shadow_mode is invalid")
        if not isinstance(self.memory_enabled, bool):
            raise ValueError("memory_enabled must be boolean")
        if self.memory_release_policy != MEMORY_RELEASE_POLICY_VERSION:
            raise ValueError("memory_release_policy is invalid")
        if not isinstance(self.research_enabled, bool) or not isinstance(self.research_browser_fallback, bool):
            raise ValueError("research feature flags must be boolean")
        if not isinstance(self.cloud_enabled, bool) or not isinstance(self.cloud_terra_enabled, bool):
            raise ValueError("cloud feature flags must be boolean")
        if self.scheduler_default_locale not in {"el-GR", "en-US"}:
            raise ValueError("scheduler_default_locale is invalid")
        try:
            ZoneInfo(self.scheduler_default_timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("scheduler_default_timezone is invalid") from exc
        if not isinstance(self.tools_enabled, bool):
            raise ValueError("tools_enabled must be boolean")
        if self.max_model_requests_per_tool_task != 6:
            raise ValueError("max_model_requests_per_tool_task is frozen at 6")
        if self.max_tool_calls_per_tool_task != 4 or self.max_tool_continuations_per_tool_task != 4:
            raise ValueError("tool call and continuation limits are frozen at 4")
        if self.tools_enabled:
            if self.model_request_limit < self.max_model_requests_per_tool_task:
                raise ValueError("tools require model_request_limit of at least 6")
            if not isinstance(self.tool_roots, Mapping) or not 1 <= len(self.tool_roots) <= 4:
                raise ValueError("tools require one to four configured roots")
            for root_id, root_path in self.tool_roots.items():
                if not isinstance(root_id, str) or not root_id.isascii() or not root_id:
                    raise ValueError("tool root ids must be non-empty ASCII")
                selected_root = Path(root_path)
                if not selected_root.is_absolute() or selected_root == Path(selected_root.anchor):
                    raise ValueError("tool roots must be absolute non-root paths")
            qualified_apps = {
                "finder", "preview", "textedit", "calendar", "reminders", "mail"
            }
            if not self.tool_app_ids or set(self.tool_app_ids) - qualified_apps or len(set(self.tool_app_ids)) != len(self.tool_app_ids):
                raise ValueError("tool_app_ids must be a unique subset of qualified apps")
            for field in ("tool_policy_sha256", "scheduler_contract_sha256"):
                digest = getattr(self, field)
                if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                    raise ValueError(f"{field} is invalid")
            if self.scheduler_qualification_manifest_sha256 and (
                len(self.scheduler_qualification_manifest_sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.scheduler_qualification_manifest_sha256)
            ):
                raise ValueError("scheduler_qualification_manifest_sha256 is invalid")
            if self.action_host_socket_path is None or not self.action_host_socket_path.is_absolute():
                raise ValueError("action_host_socket_path must be absolute")
            if self.action_host_token_file is None or not self.action_host_token_file.is_absolute():
                raise ValueError("action_host_token_file must be absolute")
            if (
                self.action_host_qualification_file is None
                or not self.action_host_qualification_file.is_absolute()
            ):
                raise ValueError("tools require an absolute action host qualification file")
            if len(self.action_host_qualification_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in self.action_host_qualification_sha256
            ):
                raise ValueError("action_host_qualification_sha256 is invalid")
        elif any(
            (
                self.tool_roots,
                self.tool_app_ids,
                self.tool_policy_sha256,
                self.scheduler_contract_sha256,
                self.scheduler_qualification_manifest_sha256,
                self.action_host_socket_path,
                self.action_host_token_file,
                self.action_host_qualification_file,
                self.action_host_qualification_sha256,
            )
        ):
            raise ValueError("tool configuration requires tools_enabled")
        if self.cloud_monthly_cap_microusd != 5_000_000:
            raise ValueError("cloud_monthly_cap_microusd is frozen at 5000000")
        if self.scheduler_compiler_qualification_file is not None:
            if not self.scheduler_compiler_qualification_file.is_absolute():
                raise ValueError("scheduler_compiler_qualification_file must be absolute")
            if len(self.scheduler_compiler_qualification_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in self.scheduler_compiler_qualification_sha256
            ):
                raise ValueError("scheduler_compiler_qualification_sha256 is invalid")
        elif self.scheduler_compiler_qualification_sha256:
            raise ValueError("scheduler compiler qualification file is required")
        if self.voice_scheduler_compatibility_sha256 and (
            len(self.voice_scheduler_compatibility_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.voice_scheduler_compatibility_sha256)
        ):
            raise ValueError("voice_scheduler_compatibility_sha256 is invalid")
        if self.voice_scheduler_callback_qualification_sha256 and (
            len(self.voice_scheduler_callback_qualification_sha256) != 64
            or any(
                c not in "0123456789abcdef"
                for c in self.voice_scheduler_callback_qualification_sha256
            )
        ):
            raise ValueError("voice_scheduler_callback_qualification_sha256 is invalid")
        if self.cloud_enabled:
            if self.cloud_secret_file is None or not self.cloud_secret_file.is_absolute():
                raise ValueError("cloud_secret_file must be an absolute external secret reference")
            if self.cloud_qualification_file is None or not self.cloud_qualification_file.is_absolute():
                raise ValueError("cloud_qualification_file must be absolute")
            if (
                self.cloud_release_qualification_file is None
                or not self.cloud_release_qualification_file.is_absolute()
            ):
                raise ValueError("cloud_release_qualification_file must be absolute")
            if len(self.cloud_release_qualification_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in self.cloud_release_qualification_sha256
            ):
                raise ValueError("cloud_release_qualification_sha256 is invalid")
            repository_root = Path(__file__).resolve().parents[3]
            secret = self.cloud_secret_file.resolve(strict=False)
            if secret == repository_root or repository_root in secret.parents:
                raise ValueError("cloud secret must not be stored in the repository")
            if self.research_enabled:
                # Research evidence is never a cloud fallback in M1C. Both
                # features may be enabled, but the service routes these flows
                # through disjoint ports and projections.
                pass
        research_url = urlparse(self.research_base_url)
        if (
            research_url.scheme != "http"
            or research_url.hostname is None
            or not ipaddress.ip_address(research_url.hostname).is_loopback
            or research_url.path.rstrip("/") != "/v1"
            or research_url.params
            or research_url.query
            or research_url.fragment
            or research_url.username is not None
        ):
            raise ValueError("research_base_url must be the fixed loopback /v1 endpoint")
        if self.research_timeout_seconds != 45.0:
            raise ValueError("research_timeout_seconds must be 45")
        if self.research_enabled:
            if self.model_request_limit < 2 or self.output_token_limit < 1152:
                raise ValueError("research requires two requests and 1152 output tokens")
            if (
                not isinstance(self.research_bearer_token, str)
                or not self.research_bearer_token
                or len(self.research_bearer_token.encode("utf-8"))
                > _MAX_BEARER_TOKEN_ENCODED_BYTES
            ):
                raise ValueError("bounded research bearer token is required")
            if self.research_language_detector_version == "disabled":
                raise ValueError("qualified research language detector is required")
            if len(self.research_language_detector_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in self.research_language_detector_sha256
            ):
                raise ValueError("research language detector digest is invalid")
            for field in (
                "research_model_sha256",
                "research_profile_sha256",
                "research_qualification_sha256",
            ):
                digest = getattr(self, field)
                if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                    raise ValueError(f"{field} is invalid")
        if self.research_browser_fallback:
            if not self.research_enabled:
                raise ValueError("browser fallback requires research_enabled")
            digest = self.research_browser_qualification_sha256
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("research browser qualification digest is invalid")
        try:
            ZoneInfo(self.research_timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("research_timezone is invalid") from exc
        if (
            isinstance(self.memory_recall_limit, bool)
            or not isinstance(self.memory_recall_limit, int)
            or self.memory_recall_limit > 20
        ):
            raise ValueError("memory_recall_limit must be an integer from 1 to 20")
