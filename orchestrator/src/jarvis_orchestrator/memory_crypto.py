"""Closed v10 memory-envelope primitives with destroyable epoch keys.

This module is deliberately storage-agnostic.  It provides the cryptographic
contract needed by a later schema migration, but it does not migrate SQLite,
manage backups, or claim physical erasure of legacy plaintext/WAL pages.

Each record receives a random data-encryption key (DEK).  The payload is
encrypted with that DEK and the DEK is wrapped by a per-principal epoch key.
HKDF gives wrapping and blind-indexing independent subkeys.  Destroying an
epoch key therefore makes envelopes still wrapped by that epoch unavailable;
the caller must prove that no retained record references it first.

The provider version namespace is global because v9 has exactly one
``local_owner`` principal.  A future multi-principal store must namespace
provider epochs by principal; it must not share this global destruction seam.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import re
import threading
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .jcs import jcs_bytes

MEMORY_ENVELOPE_CONTRACT = "jarvis.memory-envelope/1"
MEMORY_PAYLOAD_AAD_CONTRACT = "jarvis.memory-payload-aad/1"
MEMORY_WRAP_AAD_CONTRACT = "jarvis.memory-dek-wrap-aad/1"
MEMORY_HKDF_CONTRACT = "jarvis.memory-hkdf/1"
MEMORY_BLIND_INDEX_CONTRACT = "jarvis.memory-blind-index/1"
MEMORY_NORMALIZER_VERSION = "jarvis.memory-normalizer/1"
MEMORY_PROVIDER_ATTESTATION_CONTRACT = "jarvis.memory-provider-attestation/1"

MEMORY_RECORD_SCHEMA_VERSION = 10
MEMORY_ENVELOPE_VERSION = 1
ROOT_KEY_BYTES = 32
DEK_BYTES = 32
NONCE_BYTES = 12
GCM_TAG_BYTES = 16
BLIND_TOKEN_BYTES = 32
MAX_KEY_VERSION = 2_147_483_647
MAX_MEMORY_CONTENT_BYTES = 4 * 1024
MAX_MEMORY_PAYLOAD_BYTES = 16 * 1024
MAX_MEMORY_RECORD_TERMS = 128
MAX_MEMORY_QUERY_TERMS = 32

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_MILLIS = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z$"
)
_B64URL = re.compile(r"^[A-Za-z0-9_-]+$")
_HKDF_SALT = hashlib.sha256(b"JARVIS\x00jarvis.memory-hkdf/salt/1\x00").digest()


class MemoryCryptoError(RuntimeError):
    """Base class exposing a stable, content-free failure code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class MemoryKeyUnavailable(MemoryCryptoError):
    """An epoch key is missing, malformed, or inaccessible."""


class MemoryEnvelopeInvalid(MemoryCryptoError):
    """An envelope, binding, or authentication tag is invalid."""


@dataclass(frozen=True, slots=True)
class MemoryEpochProviderAttestationV1:
    """Content-free provider identity and reconciled lifecycle health."""

    install_id: str
    active_version: int
    high_water: int
    reconciliation_state: str = "clean"
    contract: str = MEMORY_PROVIDER_ATTESTATION_CONTRACT

    def __post_init__(self) -> None:
        if (
            type(self.install_id) is not str
            or _HEX_SHA256.fullmatch(self.install_id) is None
        ):
            raise MemoryKeyUnavailable("memory_key_attestation_invalid")
        _require_key_version(self.active_version)
        if (
            type(self.high_water) is not int
            or not self.active_version <= self.high_water <= MAX_KEY_VERSION
            or self.reconciliation_state != "clean"
            or self.contract != MEMORY_PROVIDER_ATTESTATION_CONTRACT
        ):
            raise MemoryKeyUnavailable("memory_key_attestation_invalid")


class MemoryEpochKeyProvider(Protocol):
    """Destroy-capable key seam; production providers live outside this module."""

    @property
    def install_id(self) -> str: ...

    def attest(self) -> MemoryEpochProviderAttestationV1: ...

    def bootstrap(self) -> tuple[int, bytes]: ...

    def active_version(self) -> int: ...

    def load(self, version: int) -> bytes: ...

    def create_next(self) -> tuple[int, bytes]: ...

    def destroy(self, version: int) -> None: ...

    def exists(self, version: int) -> bool: ...


class InMemoryMemoryEpochKeyProvider:
    """Thread-safe test seam; deletion is logical and never a forensic claim."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._keys: dict[int, bytes] = {}
        self._active: int | None = None
        self._high_water = 0
        self._install_id = os.urandom(ROOT_KEY_BYTES).hex()

    @property
    def install_id(self) -> str:
        return self._install_id

    def attest(self) -> MemoryEpochProviderAttestationV1:
        with self._lock:
            if self._active is None:
                raise MemoryKeyUnavailable("memory_key_not_initialized")
            return MemoryEpochProviderAttestationV1(
                install_id=self.install_id,
                active_version=self._active,
                high_water=self._high_water,
            )

    def bootstrap(self) -> tuple[int, bytes]:
        """Explicitly create the first epoch; repeated calls are idempotent."""

        with self._lock:
            if self._active is not None:
                return self._active, self._keys[self._active]
            if self._keys or self._high_water:
                raise MemoryKeyUnavailable("memory_key_state_invalid")
            key = os.urandom(ROOT_KEY_BYTES)
            self._keys[1] = key
            self._active = 1
            self._high_water = 1
            return 1, key

    @classmethod
    def initialized(
        cls, key: bytes | None = None, *, version: int = 1
    ) -> InMemoryMemoryEpochKeyProvider:
        """Construct an explicitly initialized provider for isolated tests."""

        _require_key_version(version)
        selected = os.urandom(ROOT_KEY_BYTES) if key is None else key
        if type(selected) is not bytes or len(selected) != ROOT_KEY_BYTES:
            raise ValueError("memory epoch key must be exactly 32 bytes")
        provider = cls()
        provider._keys[version] = bytes(selected)
        provider._active = version
        provider._high_water = version
        return provider

    def active_version(self) -> int:
        with self._lock:
            if self._active is None or self._active not in self._keys:
                raise MemoryKeyUnavailable("memory_key_not_initialized")
            return self._active

    def load(self, version: int) -> bytes:
        _require_key_version(version)
        with self._lock:
            if self._active is None:
                raise MemoryKeyUnavailable("memory_key_not_initialized")
            try:
                return self._keys[version]
            except KeyError as exc:
                raise MemoryKeyUnavailable("memory_key_not_found") from exc

    def create_next(self) -> tuple[int, bytes]:
        with self._lock:
            if self._active is None:
                raise MemoryKeyUnavailable("memory_key_not_initialized")
            if self._high_water >= MAX_KEY_VERSION:
                raise MemoryKeyUnavailable("memory_key_version_exhausted")
            version = self._high_water + 1
            self._high_water = version
            key = os.urandom(ROOT_KEY_BYTES)
            self._keys[version] = key
            self._active = version
            return version, key

    def destroy(self, version: int) -> None:
        _require_key_version(version)
        with self._lock:
            if self._active == version:
                raise MemoryKeyUnavailable("memory_key_active_destroy_forbidden")
            if self._active is None:
                raise MemoryKeyUnavailable("memory_key_not_initialized")
            if version not in self._keys:
                raise MemoryKeyUnavailable("memory_key_not_found")
            del self._keys[version]

    def exists(self, version: int) -> bool:
        _require_key_version(version)
        with self._lock:
            if self._active is None:
                raise MemoryKeyUnavailable("memory_key_not_initialized")
            return version in self._keys


def _invalid_binding() -> MemoryEnvelopeInvalid:
    return MemoryEnvelopeInvalid("memory_binding_invalid")


def _require_key_version(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_KEY_VERSION:
        raise MemoryKeyUnavailable("memory_key_version_invalid")
    return value


def _require_uuid4(value: object) -> str:
    if type(value) is not str:
        raise _invalid_binding()
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise _invalid_binding() from exc
    if parsed.version != 4 or str(parsed) != value:
        raise _invalid_binding()
    return value


def _parse_utc_millis(value: object) -> datetime:
    if type(value) is not str or _UTC_MILLIS.fullmatch(value) is None:
        raise _invalid_binding()
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise _invalid_binding() from exc
    if parsed.tzinfo != UTC:
        raise _invalid_binding()
    return parsed


def _require_nfc_text(value: object, *, max_bytes: int) -> str:
    if (
        type(value) is not str
        or not value
        or unicodedata.normalize("NFC", value) != value
    ):
        raise MemoryCryptoError("memory_index_input_invalid")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise MemoryCryptoError("memory_index_input_invalid") from exc
    if len(encoded) > max_bytes:
        raise MemoryCryptoError("memory_index_input_invalid")
    return value


@dataclass(frozen=True, slots=True)
class MemoryRecordBindingV1:
    """Immutable, content-free columns authenticated with one memory row."""

    principal_id: str
    record_id: str
    conversation_id: str
    task_id: str
    source_event_id: str
    created_at: str
    expires_at: str
    content_bytes: int
    term_count: int
    normalizer_version: str = MEMORY_NORMALIZER_VERSION
    record_schema_version: int = MEMORY_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_uuid4(self.principal_id)
        _require_uuid4(self.record_id)
        _require_uuid4(self.conversation_id)
        _require_uuid4(self.task_id)
        _require_uuid4(self.source_event_id)
        created = _parse_utc_millis(self.created_at)
        expires = _parse_utc_millis(self.expires_at)
        if expires <= created:
            raise _invalid_binding()
        if (
            type(self.content_bytes) is not int
            or not 1 <= self.content_bytes <= MAX_MEMORY_CONTENT_BYTES
        ):
            raise _invalid_binding()
        if (
            type(self.term_count) is not int
            or not 1 <= self.term_count <= MAX_MEMORY_RECORD_TERMS
        ):
            raise _invalid_binding()
        if self.normalizer_version != MEMORY_NORMALIZER_VERSION:
            raise _invalid_binding()
        if self.record_schema_version != MEMORY_RECORD_SCHEMA_VERSION:
            raise _invalid_binding()

    def payload_aad_value(self) -> dict[str, object]:
        return {
            "content_bytes": self.content_bytes,
            "contract": MEMORY_PAYLOAD_AAD_CONTRACT,
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "envelope_version": MEMORY_ENVELOPE_VERSION,
            "expires_at": self.expires_at,
            "normalizer_version": self.normalizer_version,
            "principal_id": self.principal_id,
            "record_id": self.record_id,
            "record_schema_version": self.record_schema_version,
            "source_event_id": self.source_event_id,
            "task_id": self.task_id,
            "term_count": self.term_count,
        }

    def payload_aad(self) -> bytes:
        return jcs_bytes(self.payload_aad_value())

    def binding_sha256(self) -> str:
        return hashlib.sha256(self.payload_aad()).hexdigest()


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64url_decode(
    value: object,
    *,
    minimum: int,
    maximum: int,
    exact: int | None = None,
) -> bytes:
    if type(value) is not str or not 0 <= minimum <= maximum:
        raise MemoryEnvelopeInvalid("memory_envelope_invalid")
    minimum_encoded = (minimum * 4 + 2) // 3
    maximum_encoded = (maximum * 4 + 2) // 3
    if (
        not minimum_encoded <= len(value) <= maximum_encoded
        or _B64URL.fullmatch(value) is None
        or "=" in value
    ):
        raise MemoryEnvelopeInvalid("memory_envelope_invalid")
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MemoryEnvelopeInvalid("memory_envelope_invalid") from exc
    if (
        not minimum <= len(decoded) <= maximum
        or (exact is not None and len(decoded) != exact)
        or _b64url_encode(decoded) != value
    ):
        raise MemoryEnvelopeInvalid("memory_envelope_invalid")
    return decoded


@dataclass(frozen=True, slots=True)
class MemoryEnvelopeV1:
    """Strict serialized envelope; no unknown or optional fields exist."""

    key_version: int
    payload_nonce: bytes
    payload_ciphertext_and_tag: bytes
    wrap_nonce: bytes
    wrapped_dek_and_tag: bytes
    binding_sha256: str

    _DOCUMENT_KEYS = frozenset(
        {
            "binding_sha256",
            "contract",
            "key_version",
            "payload_ciphertext_b64",
            "payload_nonce_b64",
            "wrap_nonce_b64",
            "wrapped_dek_b64",
        }
    )

    def __post_init__(self) -> None:
        try:
            _require_key_version(self.key_version)
        except MemoryKeyUnavailable as exc:
            raise MemoryEnvelopeInvalid("memory_envelope_invalid") from exc
        if (
            type(self.payload_nonce) is not bytes
            or len(self.payload_nonce) != NONCE_BYTES
        ):
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")
        if type(self.wrap_nonce) is not bytes or len(self.wrap_nonce) != NONCE_BYTES:
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")
        if (
            type(self.payload_ciphertext_and_tag) is not bytes
            or not GCM_TAG_BYTES + 1
            <= len(self.payload_ciphertext_and_tag)
            <= MAX_MEMORY_PAYLOAD_BYTES + GCM_TAG_BYTES
        ):
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")
        if (
            type(self.wrapped_dek_and_tag) is not bytes
            or len(self.wrapped_dek_and_tag) != DEK_BYTES + GCM_TAG_BYTES
        ):
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")
        if (
            type(self.binding_sha256) is not str
            or len(self.binding_sha256) != 64
            or _HEX_SHA256.fullmatch(self.binding_sha256) is None
        ):
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")

    def to_document(self) -> dict[str, object]:
        return {
            "binding_sha256": self.binding_sha256,
            "contract": MEMORY_ENVELOPE_CONTRACT,
            "key_version": self.key_version,
            "payload_ciphertext_b64": _b64url_encode(self.payload_ciphertext_and_tag),
            "payload_nonce_b64": _b64url_encode(self.payload_nonce),
            "wrap_nonce_b64": _b64url_encode(self.wrap_nonce),
            "wrapped_dek_b64": _b64url_encode(self.wrapped_dek_and_tag),
        }

    def to_jcs_bytes(self) -> bytes:
        return jcs_bytes(self.to_document())

    @classmethod
    def from_document(cls, value: object) -> MemoryEnvelopeV1:
        if type(value) is not dict or set(value) != cls._DOCUMENT_KEYS:
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")
        if value.get("contract") != MEMORY_ENVELOPE_CONTRACT:
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")
        return cls(
            key_version=value.get("key_version"),  # type: ignore[arg-type]
            payload_nonce=_b64url_decode(
                value.get("payload_nonce_b64"),
                minimum=NONCE_BYTES,
                maximum=NONCE_BYTES,
                exact=NONCE_BYTES,
            ),
            payload_ciphertext_and_tag=_b64url_decode(
                value.get("payload_ciphertext_b64"),
                minimum=GCM_TAG_BYTES + 1,
                maximum=MAX_MEMORY_PAYLOAD_BYTES + GCM_TAG_BYTES,
            ),
            wrap_nonce=_b64url_decode(
                value.get("wrap_nonce_b64"),
                minimum=NONCE_BYTES,
                maximum=NONCE_BYTES,
                exact=NONCE_BYTES,
            ),
            wrapped_dek_and_tag=_b64url_decode(
                value.get("wrapped_dek_b64"),
                minimum=DEK_BYTES + GCM_TAG_BYTES,
                maximum=DEK_BYTES + GCM_TAG_BYTES,
                exact=DEK_BYTES + GCM_TAG_BYTES,
            ),
            binding_sha256=value.get("binding_sha256"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class MemoryBlindTokensV1:
    key_version: int
    content_hmac: bytes
    normalized_hmac: bytes
    term_hmacs: tuple[bytes, ...]

    def __post_init__(self) -> None:
        try:
            _require_key_version(self.key_version)
        except MemoryKeyUnavailable as exc:
            raise MemoryCryptoError("memory_index_token_invalid") from exc
        if (
            type(self.content_hmac) is not bytes
            or len(self.content_hmac) != BLIND_TOKEN_BYTES
        ):
            raise MemoryCryptoError("memory_index_token_invalid")
        if (
            type(self.normalized_hmac) is not bytes
            or len(self.normalized_hmac) != BLIND_TOKEN_BYTES
        ):
            raise MemoryCryptoError("memory_index_token_invalid")
        if (
            type(self.term_hmacs) is not tuple
            or not 1 <= len(self.term_hmacs) <= MAX_MEMORY_RECORD_TERMS
        ):
            raise MemoryCryptoError("memory_index_token_invalid")
        if any(
            type(item) is not bytes or len(item) != BLIND_TOKEN_BYTES
            for item in self.term_hmacs
        ):
            raise MemoryCryptoError("memory_index_token_invalid")

    def hex_document(self) -> dict[str, object]:
        return {
            "content_hmac": self.content_hmac.hex(),
            "contract": MEMORY_BLIND_INDEX_CONTRACT,
            "key_version": self.key_version,
            "normalized_hmac": self.normalized_hmac.hex(),
            "term_hmacs": [item.hex() for item in self.term_hmacs],
        }


class MemoryEpochSessionV1:
    """Provider-free, principal-scoped crypto session for one key epoch.

    The session retains derived wrapping and blind-index keys, not the root
    epoch key, and never calls a provider after construction.  CPython cannot
    guarantee zeroization of immutable ``bytes`` or their transient copies;
    callers needing stronger memory-erasure guarantees require a native
    locked-memory boundary or process isolation.
    """

    __slots__ = (
        "__index_key",
        "__wrap_key",
        "_install_id",
        "_key_version",
        "_principal_id",
    )

    def __init__(
        self,
        *,
        install_id: str,
        principal_id: str,
        key_version: int,
        wrap_key: bytes,
        index_key: bytes,
    ) -> None:
        if type(install_id) is not str or _HEX_SHA256.fullmatch(install_id) is None:
            raise MemoryKeyUnavailable("memory_key_session_invalid")
        _require_uuid4(principal_id)
        _require_key_version(key_version)
        if (
            type(wrap_key) is not bytes
            or len(wrap_key) != ROOT_KEY_BYTES
            or type(index_key) is not bytes
            or len(index_key) != ROOT_KEY_BYTES
        ):
            raise MemoryKeyUnavailable("memory_key_session_invalid")
        self._install_id = install_id
        self._principal_id = principal_id
        self._key_version = key_version
        self.__wrap_key = bytes(wrap_key)
        self.__index_key = bytes(index_key)

    @property
    def install_id(self) -> str:
        return self._install_id

    @property
    def principal_id(self) -> str:
        return self._principal_id

    @property
    def key_version(self) -> int:
        return self._key_version

    def _binding(self, binding: MemoryRecordBindingV1) -> MemoryRecordBindingV1:
        if (
            type(binding) is not MemoryRecordBindingV1
            or binding.principal_id != self.principal_id
        ):
            raise _invalid_binding()
        return binding

    def encrypt(
        self,
        payload: bytes,
        binding: MemoryRecordBindingV1,
    ) -> MemoryEnvelopeV1:
        if (
            type(payload) is not bytes
            or not 1 <= len(payload) <= MAX_MEMORY_PAYLOAD_BYTES
        ):
            raise MemoryCryptoError("memory_plaintext_invalid")
        selected_binding = self._binding(binding)
        binding_sha256 = selected_binding.binding_sha256()
        payload_nonce = os.urandom(NONCE_BYTES)
        wrap_nonce = os.urandom(NONCE_BYTES)
        dek = os.urandom(DEK_BYTES)
        payload_ciphertext = AESGCM(dek).encrypt(
            payload_nonce,
            payload,
            selected_binding.payload_aad(),
        )
        wrapped_dek = AESGCM(self.__wrap_key).encrypt(
            wrap_nonce,
            dek,
            MemoryEnvelopeCipherV1._wrap_aad(
                selected_binding,
                key_version=self.key_version,
                binding_sha256=binding_sha256,
            ),
        )
        return MemoryEnvelopeV1(
            key_version=self.key_version,
            payload_nonce=payload_nonce,
            payload_ciphertext_and_tag=payload_ciphertext,
            wrap_nonce=wrap_nonce,
            wrapped_dek_and_tag=wrapped_dek,
            binding_sha256=binding_sha256,
        )

    def decrypt(
        self,
        envelope: MemoryEnvelopeV1,
        binding: MemoryRecordBindingV1,
    ) -> bytes:
        if type(envelope) is not MemoryEnvelopeV1:
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")
        selected_binding = self._binding(binding)
        if envelope.key_version != self.key_version:
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")
        expected_binding = selected_binding.binding_sha256()
        if not hmac.compare_digest(envelope.binding_sha256, expected_binding):
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")
        try:
            dek = AESGCM(self.__wrap_key).decrypt(
                envelope.wrap_nonce,
                envelope.wrapped_dek_and_tag,
                MemoryEnvelopeCipherV1._wrap_aad(
                    selected_binding,
                    key_version=self.key_version,
                    binding_sha256=expected_binding,
                ),
            )
            if len(dek) != DEK_BYTES:
                raise ValueError
            return AESGCM(dek).decrypt(
                envelope.payload_nonce,
                envelope.payload_ciphertext_and_tag,
                selected_binding.payload_aad(),
            )
        except Exception as exc:
            raise MemoryEnvelopeInvalid("memory_envelope_invalid") from exc

    def _token(self, *, kind: str, value: str) -> bytes:
        return MemoryEnvelopeCipherV1._blind_hmac(
            self.__index_key,
            principal_id=self.principal_id,
            key_version=self.key_version,
            kind=kind,
            value=value,
        )

    def blind_tokens(
        self,
        *,
        content: str,
        normalized_content: str,
        terms: tuple[str, ...],
    ) -> MemoryBlindTokensV1:
        exact_content = _require_nfc_text(content, max_bytes=MAX_MEMORY_CONTENT_BYTES)
        normalized = _require_nfc_text(
            normalized_content,
            max_bytes=MAX_MEMORY_CONTENT_BYTES,
        )
        if type(terms) is not tuple or not 1 <= len(terms) <= MAX_MEMORY_RECORD_TERMS:
            raise MemoryCryptoError("memory_index_input_invalid")
        checked_terms = tuple(
            _require_nfc_text(term, max_bytes=MAX_MEMORY_CONTENT_BYTES)
            for term in terms
        )
        if tuple(sorted(set(checked_terms))) != checked_terms:
            raise MemoryCryptoError("memory_index_input_invalid")
        return MemoryBlindTokensV1(
            key_version=self.key_version,
            content_hmac=self._token(kind="content", value=exact_content),
            normalized_hmac=self._token(kind="normalized", value=normalized),
            term_hmacs=tuple(
                self._token(kind="term", value=term) for term in checked_terms
            ),
        )

    def blind_token(self, *, kind: str, value: str) -> bytes:
        if type(kind) is not str or kind not in {"content", "normalized", "term"}:
            raise MemoryCryptoError("memory_index_input_invalid")
        checked = _require_nfc_text(value, max_bytes=MAX_MEMORY_CONTENT_BYTES)
        return self._token(kind=kind, value=checked)

    def blind_term_tokens(self, *, terms: tuple[str, ...]) -> tuple[bytes, ...]:
        if type(terms) is not tuple or not 1 <= len(terms) <= MAX_MEMORY_QUERY_TERMS:
            raise MemoryCryptoError("memory_index_input_invalid")
        checked = tuple(
            _require_nfc_text(term, max_bytes=MAX_MEMORY_CONTENT_BYTES)
            for term in terms
        )
        if tuple(sorted(set(checked))) != checked:
            raise MemoryCryptoError("memory_index_input_invalid")
        return tuple(self._token(kind="term", value=term) for term in checked)


class MemoryEnvelopeCipherV1:
    """Envelope and index operations for one externally serialized key domain.

    Storage integration must serialize epoch creation, record/reference updates,
    and destruction.  A caller-provided reference count cannot make those
    separate resources atomic on its own.
    """

    def __init__(self, provider: MemoryEpochKeyProvider) -> None:
        self.provider = provider

    def _load_root(self, version: int) -> bytes:
        _require_key_version(version)
        try:
            key = self.provider.load(version)
        except MemoryCryptoError:
            raise
        except Exception as exc:
            raise MemoryKeyUnavailable("memory_key_unavailable") from exc
        if type(key) is not bytes or len(key) != ROOT_KEY_BYTES:
            raise MemoryKeyUnavailable("memory_key_invalid")
        return key

    @staticmethod
    def _derive(
        root_key: bytes, *, purpose: str, principal_id: str, key_version: int
    ) -> bytes:
        info = jcs_bytes(
            {
                "contract": MEMORY_HKDF_CONTRACT,
                "key_version": key_version,
                "principal_id": principal_id,
                "purpose": purpose,
            }
        )
        return HKDF(
            algorithm=hashes.SHA256(),
            length=ROOT_KEY_BYTES,
            salt=_HKDF_SALT,
            info=info,
        ).derive(root_key)

    @staticmethod
    def _wrap_aad(
        binding: MemoryRecordBindingV1, *, key_version: int, binding_sha256: str
    ) -> bytes:
        return jcs_bytes(
            {
                "binding_sha256": binding_sha256,
                "contract": MEMORY_WRAP_AAD_CONTRACT,
                "key_version": key_version,
                "principal_id": binding.principal_id,
                "record_id": binding.record_id,
            }
        )

    def active_version(self) -> int:
        try:
            version = self.provider.active_version()
        except MemoryCryptoError:
            raise
        except Exception as exc:
            raise MemoryKeyUnavailable("memory_key_unavailable") from exc
        self._load_root(version)
        return version

    def open_epoch_session(
        self,
        *,
        principal_id: str,
        key_version: int,
    ) -> MemoryEpochSessionV1:
        """Load one root once, then return a provider-free scoped session."""

        _require_uuid4(principal_id)
        selected_version = _require_key_version(key_version)
        try:
            install_id = self.provider.install_id
        except MemoryCryptoError:
            raise
        except Exception as exc:
            raise MemoryKeyUnavailable("memory_key_unavailable") from exc
        root = self._load_root(selected_version)
        return MemoryEpochSessionV1(
            install_id=install_id,
            principal_id=principal_id,
            key_version=selected_version,
            wrap_key=self._derive(
                root,
                purpose="dek-wrap",
                principal_id=principal_id,
                key_version=selected_version,
            ),
            index_key=self._derive(
                root,
                purpose="blind-index",
                principal_id=principal_id,
                key_version=selected_version,
            ),
        )

    def bootstrap_epoch(self) -> int:
        """Explicitly initialize an empty provider and verify its first key."""

        try:
            created = self.provider.bootstrap()
        except MemoryCryptoError:
            raise
        except Exception as exc:
            raise MemoryKeyUnavailable("memory_key_create_failed") from exc
        if type(created) is not tuple or len(created) != 2:
            raise MemoryKeyUnavailable("memory_key_create_failed")
        version, key = created
        _require_key_version(version)
        if type(key) is not bytes or len(key) != ROOT_KEY_BYTES:
            raise MemoryKeyUnavailable("memory_key_invalid")
        if not hmac.compare_digest(key, self._load_root(version)):
            raise MemoryKeyUnavailable("memory_key_invalid")
        return version

    def create_epoch(self) -> int:
        try:
            created = self.provider.create_next()
        except MemoryCryptoError:
            raise
        except Exception as exc:
            raise MemoryKeyUnavailable("memory_key_create_failed") from exc
        if type(created) is not tuple or len(created) != 2:
            raise MemoryKeyUnavailable("memory_key_create_failed")
        version, key = created
        _require_key_version(version)
        if type(key) is not bytes or len(key) != ROOT_KEY_BYTES:
            raise MemoryKeyUnavailable("memory_key_invalid")
        loaded = self._load_root(version)
        if not hmac.compare_digest(key, loaded):
            raise MemoryKeyUnavailable("memory_key_invalid")
        try:
            exists = self.provider.exists(version)
        except Exception as exc:
            raise MemoryKeyUnavailable("memory_key_unavailable") from exc
        if exists is not True:
            raise MemoryKeyUnavailable("memory_key_invalid")
        return version

    def destroy_epoch(self, version: int, *, live_reference_count: int) -> None:
        _require_key_version(version)
        if type(live_reference_count) is not int or live_reference_count < 0:
            raise MemoryCryptoError("memory_key_reference_count_invalid")
        if live_reference_count != 0:
            raise MemoryCryptoError("memory_key_still_referenced")
        try:
            exists_before = self.provider.exists(version)
        except MemoryCryptoError:
            raise
        except Exception as exc:
            raise MemoryKeyUnavailable("memory_key_unavailable") from exc
        if exists_before is not True:
            raise MemoryKeyUnavailable("memory_key_unavailable")
        if self.active_version() == version:
            raise MemoryCryptoError("memory_key_is_active")
        try:
            self.provider.destroy(version)
            still_exists = self.provider.exists(version)
        except MemoryCryptoError:
            raise
        except Exception as exc:
            raise MemoryKeyUnavailable("memory_key_destroy_failed") from exc
        if still_exists is not False:
            raise MemoryKeyUnavailable("memory_key_destroy_failed")

    def encrypt(
        self,
        payload: bytes,
        binding: MemoryRecordBindingV1,
        *,
        key_version: int | None = None,
    ) -> MemoryEnvelopeV1:
        if (
            type(payload) is not bytes
            or not 1 <= len(payload) <= MAX_MEMORY_PAYLOAD_BYTES
        ):
            raise MemoryCryptoError("memory_plaintext_invalid")
        if type(binding) is not MemoryRecordBindingV1:
            raise _invalid_binding()
        version = (
            self.active_version()
            if key_version is None
            else _require_key_version(key_version)
        )
        root_key = self._load_root(version)
        wrap_key = self._derive(
            root_key,
            purpose="dek-wrap",
            principal_id=binding.principal_id,
            key_version=version,
        )
        binding_sha256 = binding.binding_sha256()
        payload_nonce = os.urandom(NONCE_BYTES)
        wrap_nonce = os.urandom(NONCE_BYTES)
        dek = os.urandom(DEK_BYTES)
        payload_ciphertext = AESGCM(dek).encrypt(
            payload_nonce, payload, binding.payload_aad()
        )
        wrapped_dek = AESGCM(wrap_key).encrypt(
            wrap_nonce,
            dek,
            self._wrap_aad(binding, key_version=version, binding_sha256=binding_sha256),
        )
        return MemoryEnvelopeV1(
            key_version=version,
            payload_nonce=payload_nonce,
            payload_ciphertext_and_tag=payload_ciphertext,
            wrap_nonce=wrap_nonce,
            wrapped_dek_and_tag=wrapped_dek,
            binding_sha256=binding_sha256,
        )

    def _unwrap_dek(
        self, envelope: MemoryEnvelopeV1, binding: MemoryRecordBindingV1
    ) -> bytes:
        if type(envelope) is not MemoryEnvelopeV1:
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")
        if type(binding) is not MemoryRecordBindingV1:
            raise _invalid_binding()
        expected_binding = binding.binding_sha256()
        if not hmac.compare_digest(envelope.binding_sha256, expected_binding):
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")
        root_key = self._load_root(envelope.key_version)
        wrap_key = self._derive(
            root_key,
            purpose="dek-wrap",
            principal_id=binding.principal_id,
            key_version=envelope.key_version,
        )
        try:
            dek = AESGCM(wrap_key).decrypt(
                envelope.wrap_nonce,
                envelope.wrapped_dek_and_tag,
                self._wrap_aad(
                    binding,
                    key_version=envelope.key_version,
                    binding_sha256=expected_binding,
                ),
            )
        except Exception as exc:
            raise MemoryEnvelopeInvalid("memory_envelope_invalid") from exc
        if len(dek) != DEK_BYTES:
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")
        return dek

    def decrypt(
        self, envelope: MemoryEnvelopeV1, binding: MemoryRecordBindingV1
    ) -> bytes:
        dek = self._unwrap_dek(envelope, binding)
        return self._decrypt_payload(envelope, binding, dek)

    @staticmethod
    def _decrypt_payload(
        envelope: MemoryEnvelopeV1,
        binding: MemoryRecordBindingV1,
        dek: bytes,
    ) -> bytes:
        try:
            return AESGCM(dek).decrypt(
                envelope.payload_nonce,
                envelope.payload_ciphertext_and_tag,
                binding.payload_aad(),
            )
        except Exception as exc:
            raise MemoryEnvelopeInvalid("memory_envelope_invalid") from exc

    def rewrap(
        self,
        envelope: MemoryEnvelopeV1,
        binding: MemoryRecordBindingV1,
        *,
        new_key_version: int,
    ) -> MemoryEnvelopeV1:
        if type(envelope) is not MemoryEnvelopeV1:
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")
        if type(binding) is not MemoryRecordBindingV1:
            raise _invalid_binding()
        _require_key_version(new_key_version)
        if new_key_version <= envelope.key_version:
            raise MemoryCryptoError("memory_key_version_not_forward")
        if self.active_version() != new_key_version:
            raise MemoryCryptoError("memory_key_target_not_active")
        dek = self._unwrap_dek(envelope, binding)
        # Rotation is a read-authenticate-write operation.  Never preserve a
        # corrupt payload under a successor key that may outlive the old key.
        self._decrypt_payload(envelope, binding, dek)
        new_root = self._load_root(new_key_version)
        new_wrap_key = self._derive(
            new_root,
            purpose="dek-wrap",
            principal_id=binding.principal_id,
            key_version=new_key_version,
        )
        wrap_nonce = os.urandom(NONCE_BYTES)
        wrapped_dek = AESGCM(new_wrap_key).encrypt(
            wrap_nonce,
            dek,
            self._wrap_aad(
                binding,
                key_version=new_key_version,
                binding_sha256=envelope.binding_sha256,
            ),
        )
        return MemoryEnvelopeV1(
            key_version=new_key_version,
            payload_nonce=envelope.payload_nonce,
            payload_ciphertext_and_tag=envelope.payload_ciphertext_and_tag,
            wrap_nonce=wrap_nonce,
            wrapped_dek_and_tag=wrapped_dek,
            binding_sha256=envelope.binding_sha256,
        )

    def blind_tokens(
        self,
        *,
        principal_id: str,
        key_version: int,
        content: str,
        normalized_content: str,
        terms: tuple[str, ...],
    ) -> MemoryBlindTokensV1:
        _require_uuid4(principal_id)
        _require_key_version(key_version)
        exact_content = _require_nfc_text(content, max_bytes=MAX_MEMORY_CONTENT_BYTES)
        normalized = _require_nfc_text(
            normalized_content, max_bytes=MAX_MEMORY_CONTENT_BYTES
        )
        if type(terms) is not tuple or not 1 <= len(terms) <= MAX_MEMORY_RECORD_TERMS:
            raise MemoryCryptoError("memory_index_input_invalid")
        checked_terms = tuple(
            _require_nfc_text(term, max_bytes=MAX_MEMORY_CONTENT_BYTES)
            for term in terms
        )
        if tuple(sorted(set(checked_terms))) != checked_terms:
            raise MemoryCryptoError("memory_index_input_invalid")
        index_key = self._blind_index_key(
            principal_id=principal_id, key_version=key_version
        )

        def token(kind: str, value: str) -> bytes:
            return self._blind_hmac(
                index_key,
                principal_id=principal_id,
                key_version=key_version,
                kind=kind,
                value=value,
            )

        return MemoryBlindTokensV1(
            key_version=key_version,
            content_hmac=token("content", exact_content),
            normalized_hmac=token("normalized", normalized),
            term_hmacs=tuple(token("term", term) for term in checked_terms),
        )

    def _blind_index_key(self, *, principal_id: str, key_version: int) -> bytes:
        root = self._load_root(key_version)
        return self._derive(
            root,
            purpose="blind-index",
            principal_id=principal_id,
            key_version=key_version,
        )

    @staticmethod
    def _blind_hmac(
        index_key: bytes,
        *,
        principal_id: str,
        key_version: int,
        kind: str,
        value: str,
    ) -> bytes:
        message = jcs_bytes(
            {
                "contract": MEMORY_BLIND_INDEX_CONTRACT,
                "key_version": key_version,
                "kind": kind,
                "principal_id": principal_id,
                "value": value,
            }
        )
        return hmac.new(index_key, message, hashlib.sha256).digest()

    def blind_token(
        self,
        *,
        principal_id: str,
        key_version: int,
        kind: str,
        value: str,
    ) -> bytes:
        """Derive one query token for exact, normalized, or term lookup."""

        _require_uuid4(principal_id)
        _require_key_version(key_version)
        if type(kind) is not str or kind not in {"content", "normalized", "term"}:
            raise MemoryCryptoError("memory_index_input_invalid")
        checked = _require_nfc_text(value, max_bytes=MAX_MEMORY_CONTENT_BYTES)
        return self._blind_hmac(
            self._blind_index_key(principal_id=principal_id, key_version=key_version),
            principal_id=principal_id,
            key_version=key_version,
            kind=kind,
            value=checked,
        )

    def blind_term_tokens(
        self,
        *,
        principal_id: str,
        key_version: int,
        terms: tuple[str, ...],
    ) -> tuple[bytes, ...]:
        """Derive one bounded term-query batch with a single root-key load."""

        _require_uuid4(principal_id)
        _require_key_version(key_version)
        if type(terms) is not tuple or not 1 <= len(terms) <= MAX_MEMORY_QUERY_TERMS:
            raise MemoryCryptoError("memory_index_input_invalid")
        checked = tuple(
            _require_nfc_text(term, max_bytes=MAX_MEMORY_CONTENT_BYTES)
            for term in terms
        )
        if tuple(sorted(set(checked))) != checked:
            raise MemoryCryptoError("memory_index_input_invalid")
        index_key = self._blind_index_key(
            principal_id=principal_id,
            key_version=key_version,
        )
        return tuple(
            self._blind_hmac(
                index_key,
                principal_id=principal_id,
                key_version=key_version,
                kind="term",
                value=term,
            )
            for term in checked
        )


__all__ = [
    "BLIND_TOKEN_BYTES",
    "DEK_BYTES",
    "MAX_MEMORY_PAYLOAD_BYTES",
    "MAX_MEMORY_QUERY_TERMS",
    "MEMORY_BLIND_INDEX_CONTRACT",
    "MEMORY_ENVELOPE_CONTRACT",
    "MEMORY_HKDF_CONTRACT",
    "MEMORY_PAYLOAD_AAD_CONTRACT",
    "MEMORY_PROVIDER_ATTESTATION_CONTRACT",
    "MEMORY_WRAP_AAD_CONTRACT",
    "NONCE_BYTES",
    "ROOT_KEY_BYTES",
    "InMemoryMemoryEpochKeyProvider",
    "MemoryBlindTokensV1",
    "MemoryCryptoError",
    "MemoryEnvelopeCipherV1",
    "MemoryEnvelopeInvalid",
    "MemoryEnvelopeV1",
    "MemoryEpochKeyProvider",
    "MemoryEpochProviderAttestationV1",
    "MemoryEpochSessionV1",
    "MemoryKeyUnavailable",
    "MemoryRecordBindingV1",
]
