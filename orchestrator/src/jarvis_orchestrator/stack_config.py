"""Strict private composition manifest for the real local JARVIS stack."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .config import OrchestratorConfig


STACK_SCHEMA_VERSION = "jarvis-private-stack/1"


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
    if not path.is_absolute() or not path.is_file():
        raise StackConfigError(code)
    return path.resolve()


def _private_mode(path: Path) -> None:
    if os.name != "nt" and path.stat().st_mode & 0o077:
        raise StackConfigError("private_file_permissions")


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
        source = Path(source_path).expanduser()
        if not source.is_absolute() or not source.is_file():
            raise StackConfigError("stack_config_unavailable")
        result = cls(
            source_path=source.resolve(),
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
        return hashlib.sha256(_canonical_document(self.canonical())).hexdigest()


def write_private_stack_config(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically create a mode-0600 private manifest after strict validation."""

    selected = Path(path).expanduser()
    if not selected.is_absolute():
        raise StackConfigError("stack_output_invalid")
    selected.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_document(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{selected.name}.", dir=selected.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, selected)
        if os.name != "nt":
            selected.chmod(0o600)
        PrivateStackConfig.load(selected)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "InferenceStackConfig",
    "PrivateStackConfig",
    "STACK_SCHEMA_VERSION",
    "StackConfigError",
    "write_private_stack_config",
]
