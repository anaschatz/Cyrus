"""Fail-closed native Action Host qualification and passive readiness contracts."""

from __future__ import annotations

import hashlib
import os
import stat
import platform
import re
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from .action_host_protocol import strict_json
from .action_host_v2_protocol import (
    ACTION_HOST_V2_CONTRACT_SHA256,
    ACTION_HOST_V2_PROTOCOL,
)
from .jcs import domain_sha256, jcs_bytes


NATIVE_QUALIFICATION_CONTRACT = "jarvis.action-host-native-qualification/1"
NATIVE_READINESS_CONTRACT = "jarvis.action-host-native-readiness/1"
MAX_QUALIFICATION_BYTES = 64 * 1024
MAX_QUALIFICATION_AGE = timedelta(days=30)

CALENDAR_STATES = frozenset(
    {"FULL_ACCESS", "WRITE_ONLY", "DENIED", "RESTRICTED", "NOT_DETERMINED"}
)
REMINDER_STATES = frozenset(
    {"FULL_ACCESS", "DENIED", "RESTRICTED", "NOT_DETERMINED"}
)
MAIL_STATES = frozenset({"AUTHORIZED", "DENIED", "NOT_DETERMINED", "UNAVAILABLE"})
BINARY_STATES = frozenset({"READY", "UNAVAILABLE"})


class NativeQualificationError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise NativeQualificationError(f"{name}_invalid")
    return value


def _utc(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", value
    ):
        raise NativeQualificationError(f"{name}_invalid")
    try:
        selected = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise NativeQualificationError(f"{name}_invalid") from exc
    return selected


def _exact(value: Any, fields: set[str], code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise NativeQualificationError(code)
    return dict(value)


@dataclass(frozen=True, slots=True)
class NativeReadinessV1:
    observed_at_utc: str
    calendar: str
    reminders: str
    mail_automation: str
    spotlight: str
    workspace: str
    private_notes: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NativeReadinessV1":
        row = _exact(
            value,
            {
                "contract",
                "observed_at_utc",
                "calendar",
                "reminders",
                "mail_automation",
                "spotlight",
                "workspace",
                "private_notes",
            },
            "native_readiness_invalid",
        )
        if row.pop("contract") != NATIVE_READINESS_CONTRACT:
            raise NativeQualificationError("native_readiness_invalid")
        _utc(row["observed_at_utc"], "native_readiness_observed_at")
        if not isinstance(row["calendar"], str) or row["calendar"] not in CALENDAR_STATES:
            raise NativeQualificationError("native_calendar_readiness_invalid")
        if not isinstance(row["reminders"], str) or row["reminders"] not in REMINDER_STATES:
            raise NativeQualificationError("native_reminder_readiness_invalid")
        if not isinstance(row["mail_automation"], str) or row["mail_automation"] not in MAIL_STATES:
            raise NativeQualificationError("native_mail_readiness_invalid")
        for name in ("spotlight", "workspace", "private_notes"):
            if not isinstance(row[name], str) or row[name] not in BINARY_STATES:
                raise NativeQualificationError(f"native_{name}_readiness_invalid")
        return cls(**row)

    def to_dict(self) -> dict[str, str]:
        return {
            "contract": NATIVE_READINESS_CONTRACT,
            "observed_at_utc": self.observed_at_utc,
            "calendar": self.calendar,
            "reminders": self.reminders,
            "mail_automation": self.mail_automation,
            "spotlight": self.spotlight,
            "workspace": self.workspace,
            "private_notes": self.private_notes,
        }

    def supports(self, tool_id: str) -> bool:
        if not isinstance(tool_id, str):
            return False
        if tool_id == "file.search@1":
            return self.spotlight == "READY"
        if tool_id in {"file.open@1", "app.open@1"}:
            return self.workspace == "READY"
        if tool_id == "calendar.read@1":
            return self.calendar == "FULL_ACCESS"
        if tool_id == "calendar.create@1":
            return self.calendar in {"FULL_ACCESS", "WRITE_ONLY"}
        if tool_id in {"reminder.list@1", "reminder.create@1", "reminder.complete@1"}:
            return self.reminders == "FULL_ACCESS"
        if tool_id == "note.capture@1":
            return self.private_notes == "READY"
        if tool_id == "email.draft@1":
            return self.mail_automation == "AUTHORIZED"
        return False


@dataclass(frozen=True, slots=True)
class NativeQualificationV1:
    generated_at_utc: str
    expires_at_utc: str
    platform: str
    architecture: str
    os_build: str
    protocol_sha256: str
    registry_sha256: str
    helper_binary_sha256: str
    helper_code_identity_sha256: str
    readiness_sha256: str
    test_manifest_sha256: str
    qualified_tools: tuple[str, ...]
    qualification_sha256: str

    @property
    def inputs(self) -> dict[str, Any]:
        return {
            "contract": NATIVE_QUALIFICATION_CONTRACT,
            "generated_at_utc": self.generated_at_utc,
            "expires_at_utc": self.expires_at_utc,
            "platform": self.platform,
            "architecture": self.architecture,
            "os_build": self.os_build,
            "protocol": ACTION_HOST_V2_PROTOCOL,
            "protocol_sha256": self.protocol_sha256,
            "registry_sha256": self.registry_sha256,
            "helper_binary_sha256": self.helper_binary_sha256,
            "helper_code_identity_sha256": self.helper_code_identity_sha256,
            "readiness_sha256": self.readiness_sha256,
            "test_manifest_sha256": self.test_manifest_sha256,
            "qualified_tools": list(self.qualified_tools),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.inputs, "qualification_sha256": self.qualification_sha256}


def build_native_qualification(
    *,
    generated_at_utc: str,
    expires_at_utc: str,
    os_build: str,
    registry_sha256: str,
    helper_binary_sha256: str,
    helper_code_identity_sha256: str,
    readiness: NativeReadinessV1,
    test_manifest_sha256: str,
    qualified_tools: tuple[str, ...],
) -> NativeQualificationV1:
    generated = _utc(generated_at_utc, "qualification_generated_at")
    expires = _utc(expires_at_utc, "qualification_expires_at")
    if expires <= generated or expires - generated > MAX_QUALIFICATION_AGE:
        raise NativeQualificationError("native_qualification_expiry_invalid")
    observed = _utc(readiness.observed_at_utc, "native_readiness_observed_at")
    if abs(generated - observed) > timedelta(minutes=5):
        raise NativeQualificationError("native_readiness_stale")
    if not isinstance(os_build, str) or not os_build or len(os_build.encode()) > 128:
        raise NativeQualificationError("native_os_build_invalid")
    for name, digest in (
        ("registry_sha256", registry_sha256),
        ("helper_binary_sha256", helper_binary_sha256),
        ("helper_code_identity_sha256", helper_code_identity_sha256),
        ("test_manifest_sha256", test_manifest_sha256),
    ):
        _sha(digest, name)
    if not isinstance(qualified_tools, (tuple, list)) or any(not isinstance(item, str) for item in qualified_tools):
        raise NativeQualificationError("native_qualified_tools_invalid")
    tools = tuple(sorted(qualified_tools))
    if not tools or len(set(tools)) != len(tools) or any(not readiness.supports(item) for item in tools):
        raise NativeQualificationError("native_qualified_tools_invalid")
    readiness_sha = readiness_digest(readiness)
    provisional = NativeQualificationV1(
        generated_at_utc,
        expires_at_utc,
        "macos",
        "arm64",
        os_build,
        ACTION_HOST_V2_CONTRACT_SHA256,
        registry_sha256,
        helper_binary_sha256,
        helper_code_identity_sha256,
        readiness_sha,
        test_manifest_sha256,
        tools,
        "0" * 64,
    )
    return replace(provisional, qualification_sha256=domain_sha256(
        NATIVE_QUALIFICATION_CONTRACT, provisional.inputs
    ))


def read_native_evidence_bytes(path: Path) -> bytes:
    """Read one bounded regular-file snapshot for native setup inputs."""

    with open(path, "rb", opener=lambda name, flags: os.open(
        name, flags | getattr(os, "O_NONBLOCK", 0)
    )) as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("native evidence must be a regular file")
        raw = stream.read(MAX_QUALIFICATION_BYTES + 1)
    if not raw or len(raw) > MAX_QUALIFICATION_BYTES:
        raise ValueError("native evidence size is invalid")
    return raw


def load_native_qualification(
    path: Path,
    *,
    expected_sha256: str,
    registry_sha256: str,
    helper_binary_sha256: str | None = None,
    helper_code_identity_sha256: str | None = None,
    expected_os_build: str | None = None,
    now: datetime | None = None,
) -> NativeQualificationV1:
    _sha(expected_sha256, "native_qualification_sha256")
    try:
        raw = read_native_evidence_bytes(path)
        value = strict_json(raw, maximum=MAX_QUALIFICATION_BYTES)
    except Exception as exc:
        raise NativeQualificationError("native_qualification_unavailable") from exc
    row = _exact(
        value,
        {
            "contract", "generated_at_utc", "expires_at_utc", "platform",
            "architecture", "os_build", "protocol", "protocol_sha256",
            "registry_sha256", "helper_binary_sha256",
            "helper_code_identity_sha256", "readiness_sha256",
            "test_manifest_sha256", "qualified_tools", "qualification_sha256",
        },
        "native_qualification_invalid",
    )
    if row["contract"] != NATIVE_QUALIFICATION_CONTRACT or row["protocol"] != ACTION_HOST_V2_PROTOCOL:
        raise NativeQualificationError("native_qualification_invalid")
    if row["platform"] != "macos" or row["architecture"] != "arm64":
        raise NativeQualificationError("native_qualification_platform_mismatch")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise NativeQualificationError("native_qualification_platform_mismatch")
    tools = row["qualified_tools"]
    if not isinstance(tools, list) or not tools or any(not isinstance(item, str) for item in tools) or tools != sorted(set(tools)):
        raise NativeQualificationError("native_qualified_tools_invalid")
    if not isinstance(row["os_build"], str) or not row["os_build"] or len(row["os_build"].encode()) > 128:
        raise NativeQualificationError("native_os_build_invalid")
    for name in (
        "protocol_sha256", "registry_sha256", "helper_binary_sha256",
        "helper_code_identity_sha256", "readiness_sha256",
        "test_manifest_sha256", "qualification_sha256",
    ):
        _sha(row[name], name)
    selected = NativeQualificationV1(
        row["generated_at_utc"], row["expires_at_utc"], row["platform"],
        row["architecture"], row["os_build"], row["protocol_sha256"],
        row["registry_sha256"], row["helper_binary_sha256"],
        row["helper_code_identity_sha256"], row["readiness_sha256"],
        row["test_manifest_sha256"], tuple(tools), row["qualification_sha256"],
    )
    calculated = domain_sha256(NATIVE_QUALIFICATION_CONTRACT, selected.inputs)
    if selected.qualification_sha256 != calculated or calculated != expected_sha256:
        raise NativeQualificationError("native_qualification_mismatch")
    if selected.protocol_sha256 != ACTION_HOST_V2_CONTRACT_SHA256 or selected.registry_sha256 != registry_sha256:
        raise NativeQualificationError("native_qualification_contract_mismatch")
    if helper_binary_sha256 is not None and selected.helper_binary_sha256 != helper_binary_sha256:
        raise NativeQualificationError("native_helper_identity_mismatch")
    if helper_code_identity_sha256 is not None and selected.helper_code_identity_sha256 != helper_code_identity_sha256:
        raise NativeQualificationError("native_helper_identity_mismatch")
    if expected_os_build is not None and selected.os_build != expected_os_build:
        raise NativeQualificationError("native_os_build_mismatch")
    current = now or datetime.now(UTC)
    generated = _utc(selected.generated_at_utc, "qualification_generated_at")
    expires = _utc(selected.expires_at_utc, "qualification_expires_at")
    if expires <= generated or expires - generated > MAX_QUALIFICATION_AGE:
        raise NativeQualificationError("native_qualification_expiry_invalid")
    if generated > current + timedelta(minutes=5) or not generated <= current < expires:
        raise NativeQualificationError("native_qualification_expired")
    return selected


def readiness_digest(readiness: NativeReadinessV1) -> str:
    value = readiness.to_dict()
    value.pop("observed_at_utc")
    return domain_sha256(NATIVE_READINESS_CONTRACT, value)


def current_macos_build() -> str:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise NativeQualificationError("native_qualification_platform_mismatch")
    completed = subprocess.run(
        ["/usr/bin/sw_vers", "-buildVersion"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        shell=False,
    )
    try:
        value = completed.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise NativeQualificationError("native_os_build_invalid") from exc
    if completed.returncode != 0 or not value or len(value) > 128:
        raise NativeQualificationError("native_os_build_invalid")
    return value


def test_manifest_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(jcs_bytes(value)).hexdigest()
