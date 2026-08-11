"""AES-256-GCM envelopes for transient and durable private tool content."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Protocol

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .jcs import jcs_bytes


TOOL_AEAD_CONTRACT = "jarvis.tool-payload-aead/1"
KEY_BYTES = 32
NONCE_BYTES = 12


class ToolCryptoError(RuntimeError):
    pass


class ToolKeyProvider(Protocol):
    def active_version(self) -> int: ...
    def load(self, version: int) -> bytes: ...
    def create_next(self) -> tuple[int, bytes]: ...


class InMemoryToolKeyProvider:
    """Deterministic test seam; production configuration never selects it."""

    def __init__(self, key: bytes | None = None) -> None:
        selected = key or bytes(range(KEY_BYTES))
        if len(selected) != KEY_BYTES:
            raise ValueError("tool key must be 32 bytes")
        self._keys = {1: bytes(selected)}
        self._active = 1

    def active_version(self) -> int:
        return self._active

    def load(self, version: int) -> bytes:
        try:
            return self._keys[version]
        except KeyError as exc:
            raise ToolCryptoError("tool_key_unavailable") from exc

    def create_next(self) -> tuple[int, bytes]:
        self._active += 1
        key = os.urandom(KEY_BYTES)
        self._keys[self._active] = key
        return self._active, key


@dataclass(frozen=True, slots=True)
class ToolPayloadBindingV1:
    execution_id: str
    tool_id: str
    schema_version: int
    record_kind: str

    def aad(self, *, key_version: int, content_sha256: str) -> dict[str, object]:
        return {
            "contract": TOOL_AEAD_CONTRACT,
            "execution_id": self.execution_id,
            "tool_id": self.tool_id,
            "schema_version": self.schema_version,
            "record_kind": self.record_kind,
            "key_version": key_version,
            "content_sha256": content_sha256,
        }


@dataclass(frozen=True, slots=True)
class EncryptedToolPayloadV1:
    key_version: int
    nonce: bytes
    ciphertext_and_tag: bytes
    content_sha256: str
    aad_sha256: str


class ToolPayloadCipherV1:
    def __init__(self, provider: ToolKeyProvider) -> None:
        self.provider = provider

    def encrypt(self, content: bytes, binding: ToolPayloadBindingV1) -> EncryptedToolPayloadV1:
        key_version = self.provider.active_version()
        key = self.provider.load(key_version)
        if len(key) != KEY_BYTES:
            raise ToolCryptoError("tool_key_invalid")
        nonce = os.urandom(NONCE_BYTES)
        content_sha256 = hashlib.sha256(content).hexdigest()
        aad = jcs_bytes(binding.aad(key_version=key_version, content_sha256=content_sha256))
        return EncryptedToolPayloadV1(
            key_version,
            nonce,
            AESGCM(key).encrypt(nonce, content, aad),
            content_sha256,
            hashlib.sha256(aad).hexdigest(),
        )

    def decrypt(self, envelope: EncryptedToolPayloadV1, binding: ToolPayloadBindingV1) -> bytes:
        key = self.provider.load(envelope.key_version)
        if len(key) != KEY_BYTES or len(envelope.nonce) != NONCE_BYTES:
            raise ToolCryptoError("tool_key_invalid")
        aad = jcs_bytes(binding.aad(key_version=envelope.key_version, content_sha256=envelope.content_sha256))
        if hashlib.sha256(aad).hexdigest() != envelope.aad_sha256:
            raise ToolCryptoError("tool_aad_invalid")
        try:
            content = AESGCM(key).decrypt(envelope.nonce, envelope.ciphertext_and_tag, aad)
        except Exception as exc:
            raise ToolCryptoError("tool_ciphertext_invalid") from exc
        if hashlib.sha256(content).hexdigest() != envelope.content_sha256:
            raise ToolCryptoError("tool_content_invalid")
        return content
