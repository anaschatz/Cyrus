"""Fail-closed encrypted-at-rest support for scheduler content.

The scheduler deliberately encrypts application payloads before they reach
SQLite (including WAL pages).  Envelope metadata remains content-free and is
authenticated as associated data so ciphertext cannot be moved between rows.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .jcs import domain_sha256, jcs_bytes


AEAD_CONTRACT = "scheduler-content-aead/1"
NONCE_BYTES = 12
KEY_BYTES = 32


class SchedulerCryptoError(RuntimeError):
    """Base class for scheduler-content cryptographic failures."""


class SchedulerKeyUnavailable(SchedulerCryptoError):
    """The user/device-bound scheduler key cannot be accessed."""


class SchedulerKeyNotFound(SchedulerKeyUnavailable):
    """The key item does not exist; unlike a locked keystore, creation is safe."""


class SchedulerContentInvalid(SchedulerCryptoError):
    """Ciphertext, AAD, or its authenticated content hash is invalid."""


class SchedulerKeyProvider(Protocol):
    def active_version(self) -> int: ...

    def load(self, version: int) -> bytes: ...

    def create_next(self) -> tuple[int, bytes]: ...


@dataclass(frozen=True, slots=True)
class EncryptedContent:
    key_version: int
    nonce: bytes
    ciphertext_and_tag: bytes
    content_sha256: str
    aad_sha256: str


@dataclass(frozen=True, slots=True)
class ContentBinding:
    store_role: str
    record_type: str
    record_id: str
    job_id: str | None
    job_revision: int

    def aad(self, *, content_sha256: str, key_version: int) -> dict[str, object]:
        return {
            "content_sha256": content_sha256,
            "contract": AEAD_CONTRACT,
            "job_id": self.job_id,
            "job_revision": self.job_revision,
            "key_version": key_version,
            "record_id": self.record_id,
            "record_type": self.record_type,
            "store_role": self.store_role,
        }


class InMemorySchedulerKeyProvider:
    """Deterministic seam for tests; never selected by production config."""

    def __init__(self, key: bytes | None = None) -> None:
        selected = key or bytes(range(KEY_BYTES))
        if len(selected) != KEY_BYTES:
            raise ValueError("scheduler key must be 32 bytes")
        self._keys = {1: bytes(selected)}
        self._active = 1

    def active_version(self) -> int:
        return self._active

    def load(self, version: int) -> bytes:
        try:
            return self._keys[version]
        except KeyError as exc:
            raise SchedulerKeyUnavailable("scheduler key version is unavailable") from exc

    def create_next(self) -> tuple[int, bytes]:
        self._active += 1
        key = os.urandom(KEY_BYTES)
        self._keys[self._active] = key
        return self._active, key


class _WindowsDpapiKeyProvider:
    """Current-user DPAPI wrapped keys; no machine-scope or UI fallback."""

    class _Blob(ctypes.Structure):
        _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    CRYPTPROTECT_UI_FORBIDDEN = 0x1

    def __init__(self, root: Path, namespace: str) -> None:
        self.root = root
        self.namespace = namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self._crypt32 = ctypes.windll.crypt32
        self._kernel32 = ctypes.windll.kernel32

    def _path(self, version: int) -> Path:
        return self.root / f"{self.namespace}.v{version}.dpapi"

    def _versions(self) -> list[int]:
        values: list[int] = []
        for path in self.root.glob(f"{self.namespace}.v*.dpapi"):
            try:
                values.append(int(path.name.rsplit(".v", 1)[1].split(".", 1)[0]))
            except (ValueError, IndexError):
                continue
        return sorted(values)

    def active_version(self) -> int:
        versions = self._versions()
        if versions:
            return versions[-1]
        return self.create_next()[0]

    @classmethod
    def _blob(cls, value: bytes) -> tuple["_WindowsDpapiKeyProvider._Blob", object]:
        buffer = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
        return cls._Blob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer

    def _protect(self, value: bytes) -> bytes:
        source, keepalive = self._blob(value)
        target = self._Blob()
        ok = self._crypt32.CryptProtectData(
            ctypes.byref(source),
            "JARVIS scheduler content key",
            None,
            None,
            None,
            self.CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(target),
        )
        del keepalive
        if not ok:
            raise SchedulerKeyUnavailable("DPAPI failed to protect scheduler key")
        try:
            return ctypes.string_at(target.pbData, target.cbData)
        finally:
            self._kernel32.LocalFree(target.pbData)

    def _unprotect(self, value: bytes) -> bytes:
        source, keepalive = self._blob(value)
        target = self._Blob()
        ok = self._crypt32.CryptUnprotectData(
            ctypes.byref(source),
            None,
            None,
            None,
            None,
            self.CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(target),
        )
        del keepalive
        if not ok:
            raise SchedulerKeyUnavailable("DPAPI failed to unprotect scheduler key")
        try:
            return ctypes.string_at(target.pbData, target.cbData)
        finally:
            self._kernel32.LocalFree(target.pbData)

    def load(self, version: int) -> bytes:
        try:
            wrapped = self._path(version).read_bytes()
        except OSError as exc:
            raise SchedulerKeyUnavailable("DPAPI key blob is unavailable") from exc
        key = self._unprotect(wrapped)
        if len(key) != KEY_BYTES:
            raise SchedulerKeyUnavailable("DPAPI key has invalid length")
        return key

    def create_next(self) -> tuple[int, bytes]:
        version = (self._versions()[-1] + 1) if self._versions() else 1
        key = os.urandom(KEY_BYTES)
        wrapped = self._protect(key)
        destination = self._path(version)
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as target:
                target.write(wrapped)
                target.flush()
                os.fsync(target.fileno())
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        return version, key


class _MacOSKeychainKeyProvider:
    """Data Protection Keychain with AfterFirstUnlockThisDeviceOnly items."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        try:
            self._security = ctypes.CDLL(
                "/System/Library/Frameworks/Security.framework/Security"
            )
            self._cf = ctypes.CDLL(
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
            )
        except OSError as exc:
            raise SchedulerKeyUnavailable("macOS Keychain is unavailable") from exc
        self._configure_ffi()

    def _configure_ffi(self) -> None:
        self._cf.CFDictionaryCreateMutable.restype = ctypes.c_void_p
        self._cf.CFDictionaryCreateMutable.argtypes = [ctypes.c_void_p, ctypes.c_long,
                                                        ctypes.c_void_p, ctypes.c_void_p]
        self._cf.CFDictionarySetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self._cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        self._cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        self._cf.CFDataCreate.restype = ctypes.c_void_p
        self._cf.CFDataCreate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_long]
        self._cf.CFDataGetLength.restype = ctypes.c_long
        self._cf.CFDataGetLength.argtypes = [ctypes.c_void_p]
        self._cf.CFDataGetBytePtr.restype = ctypes.POINTER(ctypes.c_ubyte)
        self._cf.CFDataGetBytePtr.argtypes = [ctypes.c_void_p]
        self._cf.CFRelease.argtypes = [ctypes.c_void_p]
        self._security.SecItemAdd.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self._security.SecItemAdd.restype = ctypes.c_int32
        self._security.SecItemCopyMatching.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self._security.SecItemCopyMatching.restype = ctypes.c_int32
        self._security.SecItemUpdate.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._security.SecItemUpdate.restype = ctypes.c_int32

    def _constant(self, name: str) -> ctypes.c_void_p:
        return ctypes.c_void_p.in_dll(self._security, name)

    def _boolean_true(self) -> ctypes.c_void_p:
        return ctypes.c_void_p.in_dll(self._cf, "kCFBooleanTrue")

    def _string(self, value: str) -> ctypes.c_void_p:
        result = self._cf.CFStringCreateWithCString(None, value.encode("utf-8"), 0x08000100)
        if not result:
            raise SchedulerKeyUnavailable("macOS Keychain string allocation failed")
        return ctypes.c_void_p(result)

    def _data(self, value: bytes) -> ctypes.c_void_p:
        buffer = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
        result = self._cf.CFDataCreate(None, buffer, len(value))
        if not result:
            raise SchedulerKeyUnavailable("macOS Keychain data allocation failed")
        return ctypes.c_void_p(result)

    def _dictionary(self, pairs: list[tuple[ctypes.c_void_p, ctypes.c_void_p]]) -> ctypes.c_void_p:
        result = self._cf.CFDictionaryCreateMutable(None, 0, None, None)
        if not result:
            raise SchedulerKeyUnavailable("macOS Keychain dictionary allocation failed")
        dictionary = ctypes.c_void_p(result)
        for key, value in pairs:
            self._cf.CFDictionarySetValue(dictionary, key, value)
        return dictionary

    def _query(self, account: str, *, return_data: bool) -> tuple[ctypes.c_void_p, list[ctypes.c_void_p]]:
        service = self._string(self.namespace)
        account_value = self._string(account)
        owned = [service, account_value]
        pairs = [
            (self._constant("kSecClass"), self._constant("kSecClassGenericPassword")),
            (self._constant("kSecAttrService"), service),
            (self._constant("kSecAttrAccount"), account_value),
            (self._constant("kSecUseDataProtectionKeychain"), self._boolean_true()),
        ]
        if return_data:
            pairs.extend([
                (self._constant("kSecReturnData"), self._boolean_true()),
                (self._constant("kSecMatchLimit"), self._constant("kSecMatchLimitOne")),
            ])
        return self._dictionary(pairs), owned

    def _release_all(self, values: list[ctypes.c_void_p]) -> None:
        for value in reversed(values):
            if value and value.value:
                self._cf.CFRelease(value)

    def _account(self, version: int) -> str:
        return f"key-v{version}"

    def _known_versions(self) -> list[int]:
        try:
            raw = self._load_secret("active-version").decode("ascii").strip()
            return [int(raw)]
        except (SchedulerKeyNotFound, UnicodeError, ValueError):
            return []

    def active_version(self) -> int:
        versions = self._known_versions()
        if versions:
            return versions[-1]
        return self.create_next()[0]

    def load(self, version: int) -> bytes:
        try:
            key = self._load_secret(self._account(version))
        except SchedulerKeyUnavailable as exc:
            raise SchedulerKeyUnavailable("macOS scheduler key is unavailable") from exc
        if len(key) != KEY_BYTES:
            raise SchedulerKeyUnavailable("macOS scheduler key has invalid length")
        return key

    def _load_secret(self, account: str) -> bytes:
        query, owned = self._query(account, return_data=True)
        result = ctypes.c_void_p()
        try:
            status = self._security.SecItemCopyMatching(query, ctypes.byref(result))
            if status == -25300:  # errSecItemNotFound
                raise SchedulerKeyNotFound("macOS Keychain item not found")
            if status != 0 or not result.value:
                raise SchedulerKeyUnavailable("macOS Keychain item unavailable")
            length = self._cf.CFDataGetLength(result)
            pointer = self._cf.CFDataGetBytePtr(result)
            return bytes(pointer[index] for index in range(length))
        finally:
            if result.value:
                self._cf.CFRelease(result)
            self._release_all([query, *owned])

    def _store_secret(self, account: str, secret: bytes) -> None:
        query, owned = self._query(account, return_data=False)
        value = self._data(secret)
        accessible = self._constant("kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly")
        additions = self._dictionary([
            (self._constant("kSecValueData"), value),
            (self._constant("kSecAttrAccessible"), accessible),
        ])
        try:
            add_dictionary = self._dictionary([
                (self._constant("kSecClass"), self._constant("kSecClassGenericPassword")),
                (self._constant("kSecAttrService"), owned[0]),
                (self._constant("kSecAttrAccount"), owned[1]),
                (self._constant("kSecUseDataProtectionKeychain"), self._boolean_true()),
                (self._constant("kSecValueData"), value),
                (self._constant("kSecAttrAccessible"), accessible),
            ])
            try:
                status = self._security.SecItemAdd(add_dictionary, None)
            finally:
                self._cf.CFRelease(add_dictionary)
            if status == -25299:  # errSecDuplicateItem
                status = self._security.SecItemUpdate(query, additions)
            if status != 0:
                raise SchedulerKeyUnavailable("macOS Keychain write failed")
        finally:
            self._release_all([additions, value, query, *owned])

    def create_next(self) -> tuple[int, bytes]:
        versions = self._known_versions()
        version = (versions[-1] + 1) if versions else 1
        key = os.urandom(KEY_BYTES)
        self._store_secret(self._account(version), key)
        self._store_secret("active-version", str(version).encode("ascii"))
        return version, key


def platform_key_provider(*, store_role: str, state_root: Path) -> SchedulerKeyProvider:
    namespace = f"com.jarvis.{store_role}.scheduler-content"
    if sys.platform == "darwin":
        return _MacOSKeychainKeyProvider(namespace)
    if sys.platform == "win32":
        return _WindowsDpapiKeyProvider(state_root / "keys", namespace)
    raise SchedulerKeyUnavailable("scheduler platform keystore is unsupported")


class SchedulerContentCipher:
    def __init__(self, provider: SchedulerKeyProvider) -> None:
        self.provider = provider

    def rotate_key(self) -> int:
        version, key = self.provider.create_next()
        if len(key) != KEY_BYTES:
            raise SchedulerKeyUnavailable("rotated scheduler key has invalid length")
        return version

    def probe(self) -> int:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise SchedulerKeyUnavailable("cryptography AES-GCM runtime is unavailable") from exc
        version = self.provider.active_version()
        key = self.provider.load(version)
        if len(key) != KEY_BYTES:
            raise SchedulerKeyUnavailable("scheduler key has invalid length")
        AESGCM(key)
        return version

    @staticmethod
    def _aesgcm(key: bytes):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise SchedulerKeyUnavailable("cryptography AES-GCM runtime is unavailable") from exc
        return AESGCM(key)

    def encrypt(self, plaintext: bytes, binding: ContentBinding) -> EncryptedContent:
        version = self.provider.active_version()
        key = self.provider.load(version)
        if len(key) != KEY_BYTES:
            raise SchedulerKeyUnavailable("scheduler key has invalid length")
        content_sha256 = domain_sha256("scheduler-content-plaintext/1", plaintext.decode("utf-8"))
        aad_value = binding.aad(content_sha256=content_sha256, key_version=version)
        aad = jcs_bytes(aad_value)
        nonce = os.urandom(NONCE_BYTES)
        ciphertext = self._aesgcm(key).encrypt(nonce, plaintext, aad)
        return EncryptedContent(
            key_version=version,
            nonce=nonce,
            ciphertext_and_tag=ciphertext,
            content_sha256=content_sha256,
            aad_sha256=domain_sha256("scheduler-content-aad/1", aad_value),
        )

    def decrypt(self, encrypted: EncryptedContent, binding: ContentBinding) -> bytes:
        if len(encrypted.nonce) != NONCE_BYTES:
            raise SchedulerContentInvalid("scheduler nonce length is invalid")
        aad_value = binding.aad(
            content_sha256=encrypted.content_sha256,
            key_version=encrypted.key_version,
        )
        if domain_sha256("scheduler-content-aad/1", aad_value) != encrypted.aad_sha256:
            raise SchedulerContentInvalid("scheduler AAD digest mismatch")
        try:
            plaintext = self._aesgcm(self.provider.load(encrypted.key_version)).decrypt(
                encrypted.nonce,
                encrypted.ciphertext_and_tag,
                jcs_bytes(aad_value),
            )
            decoded = plaintext.decode("utf-8", errors="strict")
        except SchedulerCryptoError:
            raise
        except Exception as exc:
            raise SchedulerContentInvalid("scheduler content authentication failed") from exc
        if domain_sha256("scheduler-content-plaintext/1", decoded) != encrypted.content_sha256:
            raise SchedulerContentInvalid("scheduler plaintext digest mismatch")
        return plaintext
