"""Strict, content-free metadata for same-install encrypted-memory backups.

The sidecar is a commit record, not a portable key export.  It contains no
saved-memory content, blind tokens, confirmation HMACs, or provider secrets.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Protocol

from .jcs import jcs_bytes
from .memory_crypto import MemoryEpochProviderAttestationV1

MEMORY_BACKUP_MANIFEST_CONTRACT = "jarvis.sqlite-memory-backup/1"
MEMORY_BACKUP_MANIFEST_SUFFIX = ".memory-backup-v1.json"
MAX_MEMORY_BACKUP_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_MEMORY_BACKUP_REFERENCED_EPOCHS = 10_001
MEMORY_BACKUP_SCHEMA_VERSION = 10

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class MemoryBackupError(RuntimeError):
    """Stable, content-free manifest or provider-verification failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require_hex64(value: object, code: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise MemoryBackupError(code)
    return value


def _require_uuid4(value: object, code: str) -> str:
    if type(value) is not str:
        raise MemoryBackupError(code)
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise MemoryBackupError(code) from exc
    if parsed.version != 4 or str(parsed) != value:
        raise MemoryBackupError(code)
    return value


def _require_version(value: object, code: str) -> int:
    if type(value) is not int or not 1 <= value <= 2_147_483_647:
        raise MemoryBackupError(code)
    return value


@dataclass(frozen=True, slots=True, order=True)
class MemoryBackupEpochV1:
    """Content-free identity for one provider epoch needed by a backup."""

    key_version: int
    epoch_id: str

    def __post_init__(self) -> None:
        _require_version(self.key_version, "memory_backup_manifest_invalid")
        _require_uuid4(self.epoch_id, "memory_backup_manifest_invalid")

    def document(self) -> dict[str, object]:
        return {"epoch_id": self.epoch_id, "key_version": self.key_version}

    @classmethod
    def from_document(cls, value: object) -> MemoryBackupEpochV1:
        if type(value) is not dict or set(value) != {"epoch_id", "key_version"}:
            raise MemoryBackupError("memory_backup_manifest_invalid")
        return cls(
            key_version=value.get("key_version"),  # type: ignore[arg-type]
            epoch_id=value.get("epoch_id"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class EncryptedMemoryBackupStateV1:
    """Read-only SQLite facts required to construct or validate a sidecar."""

    schema_version: int
    provider_install_id: str
    write_epoch_id: str
    memory_revision: int
    referenced_epochs: tuple[MemoryBackupEpochV1, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != MEMORY_BACKUP_SCHEMA_VERSION
        ):
            raise MemoryBackupError("memory_backup_schema_unsupported")
        _require_hex64(
            self.provider_install_id,
            "memory_backup_provider_install_invalid",
        )
        _require_uuid4(self.write_epoch_id, "memory_backup_state_invalid")
        if type(self.memory_revision) is not int or self.memory_revision < 0:
            raise MemoryBackupError("memory_backup_state_invalid")
        if (
            type(self.referenced_epochs) is not tuple
            or not 1
            <= len(self.referenced_epochs)
            <= MAX_MEMORY_BACKUP_REFERENCED_EPOCHS
            or any(
                type(epoch) is not MemoryBackupEpochV1
                for epoch in self.referenced_epochs
            )
            or tuple(sorted(set(self.referenced_epochs))) != self.referenced_epochs
        ):
            raise MemoryBackupError("memory_backup_state_invalid")
        epoch_ids = {epoch.epoch_id for epoch in self.referenced_epochs}
        versions = {epoch.key_version for epoch in self.referenced_epochs}
        if (
            len(epoch_ids) != len(self.referenced_epochs)
            or len(versions) != len(self.referenced_epochs)
            or self.write_epoch_id not in epoch_ids
        ):
            raise MemoryBackupError("memory_backup_state_invalid")

    @property
    def write_key_version(self) -> int:
        for epoch in self.referenced_epochs:
            if epoch.epoch_id == self.write_epoch_id:
                return epoch.key_version
        raise MemoryBackupError("memory_backup_state_invalid")

    @property
    def referenced_versions(self) -> tuple[int, ...]:
        return tuple(epoch.key_version for epoch in self.referenced_epochs)


class MemoryBackupProviderAttestor(Protocol):
    """Narrow key-presence seam; implementations must not expose key bytes."""

    def attest(self) -> MemoryEpochProviderAttestationV1: ...

    def exists(self, version: int) -> bool: ...


@dataclass(frozen=True, slots=True)
class MemoryBackupProviderSnapshotV1:
    install_id: str
    active_version: int
    high_water: int

    def __post_init__(self) -> None:
        _require_hex64(self.install_id, "memory_backup_provider_attestation_invalid")
        _require_version(
            self.active_version,
            "memory_backup_provider_attestation_invalid",
        )
        if (
            type(self.high_water) is not int
            or not self.active_version <= self.high_water <= 2_147_483_647
        ):
            raise MemoryBackupError("memory_backup_provider_attestation_invalid")


def verify_memory_backup_provider(
    state: EncryptedMemoryBackupStateV1,
    attestor: MemoryBackupProviderAttestor | None,
    *,
    minimum_high_water: int | None = None,
) -> MemoryBackupProviderSnapshotV1:
    """Verify one provider after all SQLite readers/transactions are closed."""

    if type(state) is not EncryptedMemoryBackupStateV1:
        raise MemoryBackupError("memory_backup_state_invalid")
    if attestor is None:
        raise MemoryBackupError("memory_backup_provider_required")
    try:
        attestation = attestor.attest()
    except Exception as exc:
        raise MemoryBackupError("memory_backup_provider_unavailable") from exc
    if type(attestation) is not MemoryEpochProviderAttestationV1:
        raise MemoryBackupError("memory_backup_provider_attestation_invalid")
    snapshot = MemoryBackupProviderSnapshotV1(
        install_id=attestation.install_id,
        active_version=attestation.active_version,
        high_water=attestation.high_water,
    )
    if snapshot.install_id != state.provider_install_id:
        raise MemoryBackupError("memory_backup_provider_install_mismatch")
    if snapshot.active_version != state.write_key_version:
        raise MemoryBackupError("memory_backup_provider_active_mismatch")
    required_high_water = max(state.referenced_versions)
    if minimum_high_water is not None:
        if type(minimum_high_water) is not int or minimum_high_water < 1:
            raise MemoryBackupError("memory_backup_manifest_invalid")
        required_high_water = max(required_high_water, minimum_high_water)
    if snapshot.high_water < required_high_water:
        raise MemoryBackupError("memory_backup_provider_high_water_invalid")
    for version in state.referenced_versions:
        try:
            exists = attestor.exists(version)
        except Exception as exc:
            raise MemoryBackupError("memory_backup_provider_unavailable") from exc
        if type(exists) is not bool:
            raise MemoryBackupError("memory_backup_provider_attestation_invalid")
        if not exists:
            raise MemoryBackupError("memory_backup_provider_key_missing")
    return snapshot


@dataclass(frozen=True, slots=True)
class MemoryBackupManifestV1:
    """Canonical commit record for one exact encrypted SQLite backup."""

    backup_sha256: str
    backup_bytes: int
    memory_state: EncryptedMemoryBackupStateV1
    provider_active_version: int
    provider_high_water: int
    same_install_only: bool = True
    portable: bool = False
    transcript_encrypted: bool = False
    physical_erasure_claimed: bool = False
    contract: str = MEMORY_BACKUP_MANIFEST_CONTRACT

    def __post_init__(self) -> None:
        _require_hex64(self.backup_sha256, "memory_backup_manifest_invalid")
        if type(self.backup_bytes) is not int or self.backup_bytes < 1:
            raise MemoryBackupError("memory_backup_manifest_invalid")
        if type(self.memory_state) is not EncryptedMemoryBackupStateV1:
            raise MemoryBackupError("memory_backup_manifest_invalid")
        _require_version(
            self.provider_active_version,
            "memory_backup_manifest_invalid",
        )
        if (
            type(self.provider_high_water) is not int
            or not self.provider_active_version
            <= self.provider_high_water
            <= 2_147_483_647
            or self.provider_active_version != self.memory_state.write_key_version
            or self.provider_high_water < max(self.memory_state.referenced_versions)
            or self.same_install_only is not True
            or self.portable is not False
            or self.transcript_encrypted is not False
            or self.physical_erasure_claimed is not False
            or self.contract != MEMORY_BACKUP_MANIFEST_CONTRACT
        ):
            raise MemoryBackupError("memory_backup_manifest_invalid")

    def document(self) -> dict[str, object]:
        return {
            "backup_bytes": self.backup_bytes,
            "backup_sha256": self.backup_sha256,
            "contract": self.contract,
            "memory_revision": self.memory_state.memory_revision,
            "memory_storage_state": "ENCRYPTED",
            "physical_erasure_claimed": self.physical_erasure_claimed,
            "portable": self.portable,
            "provider_active_version": self.provider_active_version,
            "provider_high_water": self.provider_high_water,
            "provider_install_id": self.memory_state.provider_install_id,
            "referenced_epochs": [
                epoch.document() for epoch in self.memory_state.referenced_epochs
            ],
            "same_install_only": self.same_install_only,
            "schema_version": self.memory_state.schema_version,
            "transcript_encrypted": self.transcript_encrypted,
            "write_epoch_id": self.memory_state.write_epoch_id,
        }

    def to_jcs_bytes(self) -> bytes:
        payload = jcs_bytes(self.document())
        if len(payload) > MAX_MEMORY_BACKUP_MANIFEST_BYTES:
            raise MemoryBackupError("memory_backup_manifest_too_large")
        return payload

    @classmethod
    def from_jcs_bytes(cls, payload: object) -> MemoryBackupManifestV1:
        if (
            type(payload) is not bytes
            or not 1 <= len(payload) <= MAX_MEMORY_BACKUP_MANIFEST_BYTES
        ):
            raise MemoryBackupError("memory_backup_manifest_invalid")
        try:
            document = json.loads(payload.decode("utf-8", errors="strict"))
        except (ValueError, RecursionError) as exc:
            raise MemoryBackupError("memory_backup_manifest_invalid") from exc
        expected = {
            "backup_bytes",
            "backup_sha256",
            "contract",
            "memory_revision",
            "memory_storage_state",
            "physical_erasure_claimed",
            "portable",
            "provider_active_version",
            "provider_high_water",
            "provider_install_id",
            "referenced_epochs",
            "same_install_only",
            "schema_version",
            "transcript_encrypted",
            "write_epoch_id",
        }
        if type(document) is not dict or set(document) != expected:
            raise MemoryBackupError("memory_backup_manifest_invalid")
        epoch_values = document.get("referenced_epochs")
        if (
            type(epoch_values) is not list
            or not 1 <= len(epoch_values) <= MAX_MEMORY_BACKUP_REFERENCED_EPOCHS
        ):
            raise MemoryBackupError("memory_backup_manifest_invalid")
        state = EncryptedMemoryBackupStateV1(
            schema_version=document.get("schema_version"),  # type: ignore[arg-type]
            provider_install_id=document.get("provider_install_id"),  # type: ignore[arg-type]
            write_epoch_id=document.get("write_epoch_id"),  # type: ignore[arg-type]
            memory_revision=document.get("memory_revision"),  # type: ignore[arg-type]
            referenced_epochs=tuple(
                MemoryBackupEpochV1.from_document(value) for value in epoch_values
            ),
        )
        manifest = cls(
            backup_sha256=document.get("backup_sha256"),  # type: ignore[arg-type]
            backup_bytes=document.get("backup_bytes"),  # type: ignore[arg-type]
            memory_state=state,
            provider_active_version=document.get("provider_active_version"),  # type: ignore[arg-type]
            provider_high_water=document.get("provider_high_water"),  # type: ignore[arg-type]
            same_install_only=document.get("same_install_only"),  # type: ignore[arg-type]
            portable=document.get("portable"),  # type: ignore[arg-type]
            transcript_encrypted=document.get("transcript_encrypted"),  # type: ignore[arg-type]
            physical_erasure_claimed=document.get("physical_erasure_claimed"),  # type: ignore[arg-type]
            contract=document.get("contract"),  # type: ignore[arg-type]
        )
        if document.get("memory_storage_state") != "ENCRYPTED":
            raise MemoryBackupError("memory_backup_manifest_invalid")
        if manifest.to_jcs_bytes() != payload:
            raise MemoryBackupError("memory_backup_manifest_invalid")
        return manifest


__all__ = [
    "MEMORY_BACKUP_MANIFEST_CONTRACT",
    "MEMORY_BACKUP_MANIFEST_SUFFIX",
    "EncryptedMemoryBackupStateV1",
    "MemoryBackupEpochV1",
    "MemoryBackupError",
    "MemoryBackupManifestV1",
    "MemoryBackupProviderAttestor",
    "MemoryBackupProviderSnapshotV1",
    "verify_memory_backup_provider",
]
