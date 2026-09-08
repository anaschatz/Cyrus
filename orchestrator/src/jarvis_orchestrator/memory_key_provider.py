# ruff: noqa: BLE001, S110
"""Production platform epoch-key providers for encrypted personal memory.

The module is intentionally independent from configuration and SQLite.  It
implements the ``MemoryEpochKeyProvider`` contract using either the macOS Data
user login Keychain or current-user Windows DPAPI files.  All public failures
are stable and content-free; paths, native status text, and key bytes are never
included in an exception message.

Version allocation is serialized by an owner-private interprocess file lock.
An immutable initialization marker distinguishes a fresh installation from a
lost active pointer, preventing silent epoch-version reuse.  The provider
never deletes the active epoch and verifies every destructive operation.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import ctypes
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .jcs import jcs_bytes
from .memory_crypto import (
    ROOT_KEY_BYTES,
    MemoryEpochProviderAttestationV1,
    MemoryKeyUnavailable,
)
from .private_state import (
    PrivateStateError,
    absolute_path_no_resolve,
    ensure_private_directory,
    open_private_lock,
)

MEMORY_KEY_NAMESPACE = "memory-records"
MEMORY_KEYCHAIN_SERVICE = "com.jarvis.memory-records.login-keychain-v1"
MEMORY_KEY_STATE_CONTRACT = "jarvis.memory-key-state/1"
MEMORY_KEY_MARKER_CONTRACT = "jarvis.memory-key-initialized/1"
MEMORY_KEY_LEDGER_CONTRACT = "jarvis.memory-key-ledger/1"
MEMORY_KEY_PENDING_CONTRACT = "jarvis.memory-key-pending-create/1"
MEMORY_KEY_ANCHOR_CONTRACT = "jarvis.memory-key-installation-anchor/1"
MEMORY_DPAPI_BLOB_CONTRACT = "jarvis.memory-dpapi-key/1"
MEMORY_DPAPI_ENTROPY_CONTRACT = "jarvis.memory-dpapi-entropy/1"

MAX_KEY_VERSION = 2_147_483_647
MAX_STATE_BYTES = 512
MAX_MARKER_BYTES = 256
MAX_LEDGER_BYTES = 512
MAX_PENDING_BYTES = 2048
MAX_ANCHOR_BYTES = 512
MAX_DPAPI_BLOB_BYTES = 16 * 1024
MAX_KEYCHAIN_ITEM_BYTES = 16 * 1024
PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600

_STATE_ACCOUNT = "active-state"
_MARKER_ACCOUNT = "initialized-v1"
_LEDGER_ACCOUNT = "used-versions-v1"
_PENDING_ACCOUNT = "pending-create-v1"
_ANCHOR_FILE_NAME = "installation-anchor-v1.json"
_PHYSICAL_LOCK_FILE_NAME = "provider.lock"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_B64URL = re.compile(r"^[A-Za-z0-9_-]+$")


def _fsync_directory(path: Path) -> None:
    """Durably flush one directory on POSIX or a native Windows handle."""

    if os.name != "nt":
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise _provider_error("memory_key_durability_unavailable", exc)
        try:
            os.fsync(descriptor)
        except OSError as exc:
            raise _provider_error("memory_key_durability_unavailable", exc)
        finally:
            os.close(descriptor)
        return

    # Windows does not document FlushFileBuffers for directory handles.  All
    # authoritative Windows name commits use MoveFileExW + WRITE_THROUGH
    # instead; callers must not silently substitute a weaker directory flush.
    raise _provider_error("memory_key_durability_unavailable")


def _atomic_move(source: Path, destination: Path, *, replace: bool) -> None:
    """Commit a complete staged file name with documented durability."""

    if os.name != "nt":
        try:
            if replace:
                os.replace(source, destination)
            else:
                os.link(source, destination)
                source.unlink()
            _fsync_directory(destination.parent)
        except MemoryKeyProviderError:
            raise
        except OSError as exc:
            raise _provider_error("memory_key_durability_unavailable", exc)
        return
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        move_file = kernel32.MoveFileExW
        move_file.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
        move_file.restype = wintypes.BOOL
        flags = 0x00000008 | (0x00000001 if replace else 0)
        committed = move_file(
            str(source),
            str(destination),
            flags,
        )
    except Exception as exc:
        raise _provider_error("memory_key_durability_unavailable", exc)
    if not committed:
        raise _provider_error("memory_key_durability_unavailable")


def _recover_staging_links(path: Path, before: os.stat_result) -> os.stat_result:
    """Finish interrupted POSIX exclusive publication under the provider lock."""
    if (
        os.name != "posix" or before.st_nlink <= 1
        or not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != PRIVATE_FILE_MODE
    ):
        return before
    pattern = re.compile(r"\." + re.escape(path.name) + r"\.[0-9a-f]{32}\.(tmp|stage)")
    removed = False
    try:
        with os.scandir(path.parent) as entries:
            for entry in entries:
                if pattern.fullmatch(entry.name) is None:
                    continue
                candidate = entry.stat(follow_symlinks=False)
                if (candidate.st_dev, candidate.st_ino) == (before.st_dev, before.st_ino):
                    os.unlink(entry.path)
                    removed = True
        if removed:
            _fsync_directory(path.parent)
        current = path.lstat()
        if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
            raise _provider_error("memory_key_private_state_invalid")
        return current
    except OSError as exc:
        raise _provider_error("memory_key_store_unavailable", exc)


def _durable_unlink(path: Path) -> None:
    """Durably remove an authoritative name without claiming key erasure."""

    if os.name != "nt":
        try:
            path.unlink()
            _fsync_directory(path.parent)
        except OSError as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        return
    tombstone = path.parent / f".{path.name}.{secrets.token_hex(16)}.deleted"
    _atomic_move(path, tombstone, replace=False)
    try:
        tombstone.unlink()
    except OSError:
        # MOVEFILE_WRITE_THROUGH already made the authoritative name absence
        # durable.  A hidden tombstone is cleanup residue, not live authority.
        pass


class MemoryKeyProviderError(MemoryKeyUnavailable):
    """Stable provider failure compatible with ``MemoryEnvelopeCipherV1``."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.__suppress_context__ = True


def _provider_error(
    code: str, cause: BaseException | None = None
) -> MemoryKeyProviderError:
    # The native/path exception is deliberately discarded.  Full tracebacks
    # must not turn this content-free API into a path or platform-status leak.
    del cause
    return MemoryKeyProviderError(code)


def _require_version(value: object, *, allow_next: bool = False) -> int:
    maximum = MAX_KEY_VERSION + (1 if allow_next else 0)
    if type(value) is not int or not 1 <= value <= maximum:
        raise _provider_error("memory_key_version_invalid")
    return value


def _require_install_id(value: object) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise _provider_error("memory_key_install_invalid")
    return value


def _physical_key_root(key_root: Path) -> Path:
    """Resolve aliases to the one physical lock/anchor directory."""

    try:
        selected = absolute_path_no_resolve(key_root).resolve(strict=True)
        metadata = selected.stat()
    except OSError as exc:
        raise _provider_error("memory_key_private_state_invalid", exc)
    if not stat.S_ISDIR(metadata.st_mode):
        raise _provider_error("memory_key_private_state_invalid")
    return selected


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _require_keychain_account(value: object) -> str:
    if type(value) is not str:
        raise _provider_error("memory_key_namespace_invalid")
    try:
        prefix, account = value.split("/", 1)
    except ValueError:
        raise _provider_error("memory_key_namespace_invalid")
    if not prefix.startswith("install-"):
        raise _provider_error("memory_key_namespace_invalid")
    _require_install_id(prefix.removeprefix("install-"))
    if account in {
        _STATE_ACCOUNT,
        _MARKER_ACCOUNT,
        _LEDGER_ACCOUNT,
        _PENDING_ACCOUNT,
    }:
        return value
    if not account.startswith("epoch-v"):
        raise _provider_error("memory_key_namespace_invalid")
    raw = account.removeprefix("epoch-v")
    if not raw.isascii() or not raw.isdecimal() or raw.startswith("0"):
        raise _provider_error("memory_key_namespace_invalid")
    try:
        version = int(raw)
        _require_version(version)
    except (ValueError, MemoryKeyProviderError):
        raise _provider_error("memory_key_namespace_invalid")
    if account != f"epoch-v{version}":
        raise _provider_error("memory_key_namespace_invalid")
    return value


def _b64url_decode(value: object, *, maximum: int) -> bytes:
    maximum_encoded = (maximum * 4 + 2) // 3
    if (
        type(value) is not str
        or not 1 <= len(value) <= maximum_encoded
        or _B64URL.fullmatch(value) is None
        or "=" in value
    ):
        raise _provider_error("memory_key_blob_invalid")
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _provider_error("memory_key_blob_invalid", exc)
    if not decoded or len(decoded) > maximum or _b64url_encode(decoded) != value:
        raise _provider_error("memory_key_blob_invalid")
    return decoded


@dataclass(frozen=True, slots=True)
class MemoryInstallationAnchorV1:
    """Durable identity and monotonic epoch evidence for one installation."""

    install_id: str
    high_water: int = 0
    initialized: bool = False

    def __post_init__(self) -> None:
        _require_install_id(self.install_id)
        if (
            type(self.high_water) is not int
            or not 0 <= self.high_water <= MAX_KEY_VERSION
            or type(self.initialized) is not bool
            or (self.initialized and self.high_water < 1)
        ):
            raise _provider_error("memory_key_anchor_invalid")

    def document(self) -> dict[str, object]:
        return {
            "contract": MEMORY_KEY_ANCHOR_CONTRACT,
            "high_water": self.high_water,
            "initialized": self.initialized,
            "install_id": self.install_id,
            "namespace": MEMORY_KEY_NAMESPACE,
        }

    def encode(self) -> bytes:
        return jcs_bytes(self.document())

    @classmethod
    def decode(cls, payload: object) -> MemoryInstallationAnchorV1:
        if type(payload) is not bytes or not 1 <= len(payload) <= MAX_ANCHOR_BYTES:
            raise _provider_error("memory_key_anchor_invalid")
        try:
            document = json.loads(payload.decode("utf-8", errors="strict"))
        except (ValueError, RecursionError) as exc:
            raise _provider_error("memory_key_anchor_invalid", exc)
        if type(document) is not dict or set(document) != {
            "contract",
            "high_water",
            "initialized",
            "install_id",
            "namespace",
        }:
            raise _provider_error("memory_key_anchor_invalid")
        if (
            document.get("contract") != MEMORY_KEY_ANCHOR_CONTRACT
            or document.get("namespace") != MEMORY_KEY_NAMESPACE
        ):
            raise _provider_error("memory_key_anchor_invalid")
        try:
            anchor = cls(
                install_id=document.get("install_id"),
                high_water=document.get("high_water"),
                initialized=document.get("initialized"),
            )
        except MemoryKeyProviderError as exc:
            raise _provider_error("memory_key_anchor_invalid", exc)
        if anchor.encode() != payload:
            raise _provider_error("memory_key_anchor_invalid")
        return anchor


class _InstallationAnchorFile:
    """Owner-private, durably replaceable anchor below the physical key root."""

    def __init__(
        self,
        root: Path,
        *,
        windows_guard: WindowsPrivateGuard | None = None,
    ) -> None:
        self.root = _physical_key_root(root)
        self.path = self.root / _ANCHOR_FILE_NAME
        self.windows_guard = windows_guard

    def _validate_root(self) -> None:
        if self.windows_guard is None:
            try:
                ensure_private_directory(self.root)
            except PrivateStateError as exc:
                raise _provider_error("memory_key_private_state_invalid", exc)
            return
        try:
            self.windows_guard.validate_directory(self.root)
        except MemoryKeyProviderError:
            raise
        except Exception as exc:
            raise _provider_error("memory_key_private_state_invalid", exc)

    def _validate_metadata(self, metadata: os.stat_result) -> None:
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise _provider_error("memory_key_private_state_invalid")
        if os.name == "posix" and (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != PRIVATE_FILE_MODE
        ):
            raise _provider_error("memory_key_private_state_invalid")

    def read(self) -> MemoryInstallationAnchorV1 | None:
        self._validate_root()
        try:
            before = self.path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        before = _recover_staging_links(self.path, before)
        self._validate_metadata(before)
        if self.windows_guard is not None:
            try:
                self.windows_guard.validate_file(self.path)
            except MemoryKeyProviderError:
                raise
            except Exception as exc:
                raise _provider_error("memory_key_private_state_invalid", exc)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        if os.name == "posix":
            flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
        except OSError as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        try:
            opened = os.fstat(descriptor)
            self._validate_metadata(opened)
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise _provider_error("memory_key_private_state_invalid")
            if self.windows_guard is not None:
                try:
                    self.windows_guard.validate_descriptor(descriptor)
                except MemoryKeyProviderError:
                    raise
                except Exception as exc:
                    raise _provider_error("memory_key_private_state_invalid", exc)
            if not 1 <= opened.st_size <= MAX_ANCHOR_BYTES:
                raise _provider_error("memory_key_anchor_invalid")
            payload = os.read(descriptor, MAX_ANCHOR_BYTES + 1)
            if len(payload) != opened.st_size or len(payload) > MAX_ANCHOR_BYTES:
                raise _provider_error("memory_key_anchor_invalid")
            return MemoryInstallationAnchorV1.decode(payload)
        except OSError as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        finally:
            os.close(descriptor)

    def _fsync_root(self) -> None:
        _fsync_directory(self.root)

    def create(self, anchor: MemoryInstallationAnchorV1) -> bool:
        if type(anchor) is not MemoryInstallationAnchorV1:
            raise _provider_error("memory_key_anchor_invalid")
        self._validate_root()
        try:
            self.path.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        else:
            return False
        temporary = self.root / (f".{_ANCHOR_FILE_NAME}.{secrets.token_hex(16)}.tmp")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        if os.name == "posix":
            flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(temporary, flags, PRIVATE_FILE_MODE)
        except OSError as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        write_error: Exception | None = None
        try:
            if os.name == "posix":
                os.fchmod(descriptor, PRIVATE_FILE_MODE)
            if self.windows_guard is not None:
                self.windows_guard.harden_file(temporary)
                self.windows_guard.validate_descriptor(descriptor)
            payload = anchor.encode()
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError
                view = view[written:]
            os.fsync(descriptor)
            self._validate_metadata(os.fstat(descriptor))
        except Exception as exc:
            write_error = exc
        finally:
            os.close(descriptor)
        if write_error is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise _provider_error("memory_key_commit_uncertain", write_error)
        try:
            try:
                self.path.lstat()
            except FileNotFoundError:
                pass
            else:
                temporary.unlink(missing_ok=True)
                return False
            # The authoritative name appears only after a complete, fsynced
            # payload exists.  A crash while writing can therefore leave at
            # most an ignored temporary file, never a malformed anchor.
            _atomic_move(temporary, self.path, replace=False)
            if self.windows_guard is not None:
                self.windows_guard.harden_file(self.path)
                self.windows_guard.validate_file(self.path)
        except Exception as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise _provider_error("memory_key_commit_uncertain", exc)
        if self.read() != anchor:
            raise _provider_error("memory_key_commit_uncertain")
        return True

    def replace_exact(
        self,
        expected: MemoryInstallationAnchorV1,
        selected: MemoryInstallationAnchorV1,
    ) -> None:
        """Atomically advance exact lifecycle evidence or fail uncertain."""

        if (
            type(expected) is not MemoryInstallationAnchorV1
            or type(selected) is not MemoryInstallationAnchorV1
            or expected.install_id != selected.install_id
            or selected.high_water < expected.high_water
            or expected.initialized > selected.initialized
            or not (
                (
                    selected.high_water == expected.high_water + 1
                    and selected.initialized == expected.initialized
                )
                or (
                    selected.high_water == expected.high_water
                    and not expected.initialized
                    and selected.initialized
                )
            )
        ):
            raise _provider_error("memory_key_anchor_invalid")
        observed = self.read()
        if observed == selected:
            return
        if observed != expected:
            raise _provider_error("memory_key_state_invalid")

        temporary = self.root / f".{_ANCHOR_FILE_NAME}.{secrets.token_hex(16)}.tmp"
        payload = selected.encode()
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        if os.name == "posix":
            flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary, flags, PRIVATE_FILE_MODE)
            if os.name == "posix":
                os.fchmod(descriptor, PRIVATE_FILE_MODE)
            if self.windows_guard is not None:
                self.windows_guard.harden_file(temporary)
                self.windows_guard.validate_descriptor(descriptor)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError
                view = view[written:]
            os.fsync(descriptor)
            self._validate_metadata(os.fstat(descriptor))
            os.close(descriptor)
            descriptor = None
            if self.windows_guard is not None:
                self.windows_guard.validate_file(temporary)
            if self.read() != expected:
                raise _provider_error("memory_key_state_invalid")
            _atomic_move(temporary, self.path, replace=True)
            if self.read() != selected:
                raise _provider_error("memory_key_commit_uncertain")
        except Exception as exc:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if isinstance(exc, MemoryKeyProviderError) and str(exc) in {
                "memory_key_anchor_invalid",
                "memory_key_state_invalid",
            }:
                raise
            # A replace or durability operation that raised is never success,
            # even when retry-time readback later observes ``selected``.
            raise _provider_error("memory_key_commit_uncertain", exc)

    def confirm_durable(self, expected: MemoryInstallationAnchorV1) -> None:
        """Re-establish directory durability before adopting an anchor."""

        if type(expected) is not MemoryInstallationAnchorV1:
            raise _provider_error("memory_key_anchor_invalid")
        if self.read() != expected:
            raise _provider_error("memory_key_state_invalid")
        if os.name != "nt":
            try:
                self._fsync_root()
            except Exception as exc:
                raise _provider_error("memory_key_commit_uncertain", exc)


@dataclass(frozen=True, slots=True)
class MemoryKeyStateV1:
    install_id: str
    active_version: int
    next_version: int

    def __post_init__(self) -> None:
        _require_install_id(self.install_id)
        _require_version(self.active_version)
        _require_version(self.next_version, allow_next=True)
        if self.next_version <= self.active_version:
            raise _provider_error("memory_key_state_invalid")

    def document(self) -> dict[str, object]:
        return {
            "active_version": self.active_version,
            "contract": MEMORY_KEY_STATE_CONTRACT,
            "install_id": self.install_id,
            "namespace": MEMORY_KEY_NAMESPACE,
            "next_version": self.next_version,
        }

    def encode(self) -> bytes:
        return jcs_bytes(self.document())

    @classmethod
    def decode(cls, payload: object) -> MemoryKeyStateV1:
        if type(payload) is not bytes or not 1 <= len(payload) <= MAX_STATE_BYTES:
            raise _provider_error("memory_key_state_invalid")
        try:
            document = json.loads(payload.decode("utf-8", errors="strict"))
        except (ValueError, RecursionError) as exc:
            raise _provider_error("memory_key_state_invalid", exc)
        expected = {
            "active_version",
            "contract",
            "install_id",
            "namespace",
            "next_version",
        }
        if type(document) is not dict or set(document) != expected:
            raise _provider_error("memory_key_state_invalid")
        if (
            document.get("contract") != MEMORY_KEY_STATE_CONTRACT
            or document.get("namespace") != MEMORY_KEY_NAMESPACE
        ):
            raise _provider_error("memory_key_state_invalid")
        try:
            state = cls(
                install_id=document.get("install_id"),
                active_version=document.get("active_version"),
                next_version=document.get("next_version"),
            )
        except MemoryKeyProviderError as exc:
            raise _provider_error("memory_key_state_invalid", exc)
        if state.encode() != payload:
            raise _provider_error("memory_key_state_invalid")
        return state


def _marker_payload(install_id: str) -> bytes:
    _require_install_id(install_id)
    return jcs_bytes(
        {
            "contract": MEMORY_KEY_MARKER_CONTRACT,
            "install_id": install_id,
            "namespace": MEMORY_KEY_NAMESPACE,
        }
    )


@dataclass(frozen=True, slots=True)
class MemoryKeyLedgerV1:
    install_id: str
    high_water: int
    initialized: bool

    def __post_init__(self) -> None:
        _require_install_id(self.install_id)
        if (
            type(self.high_water) is not int
            or not 0 <= self.high_water <= MAX_KEY_VERSION
        ):
            raise _provider_error("memory_key_ledger_invalid")
        if type(self.initialized) is not bool:
            raise _provider_error("memory_key_ledger_invalid")
        if self.initialized and self.high_water < 1:
            raise _provider_error("memory_key_ledger_invalid")

    def document(self) -> dict[str, object]:
        return {
            "contract": MEMORY_KEY_LEDGER_CONTRACT,
            "high_water": self.high_water,
            "initialized": self.initialized,
            "install_id": self.install_id,
            "namespace": MEMORY_KEY_NAMESPACE,
        }

    def encode(self) -> bytes:
        return jcs_bytes(self.document())

    @classmethod
    def decode(cls, payload: object) -> MemoryKeyLedgerV1:
        if type(payload) is not bytes or not 1 <= len(payload) <= MAX_LEDGER_BYTES:
            raise _provider_error("memory_key_ledger_invalid")
        try:
            document = json.loads(payload.decode("utf-8", errors="strict"))
        except (ValueError, RecursionError) as exc:
            raise _provider_error("memory_key_ledger_invalid", exc)
        if type(document) is not dict or set(document) != {
            "contract",
            "high_water",
            "initialized",
            "install_id",
            "namespace",
        }:
            raise _provider_error("memory_key_ledger_invalid")
        if (
            document.get("contract") != MEMORY_KEY_LEDGER_CONTRACT
            or document.get("namespace") != MEMORY_KEY_NAMESPACE
        ):
            raise _provider_error("memory_key_ledger_invalid")
        try:
            ledger = cls(
                install_id=document.get("install_id"),
                high_water=document.get("high_water"),
                initialized=document.get("initialized"),
            )
        except MemoryKeyProviderError as exc:
            raise _provider_error("memory_key_ledger_invalid", exc)
        if ledger.encode() != payload:
            raise _provider_error("memory_key_ledger_invalid")
        return ledger


_PENDING_PHASES = (
    "reserved",
    "high_water_committed",
    "key_created",
    "state_committed",
)


@dataclass(frozen=True, slots=True)
class MemoryKeyPendingCreateV1:
    install_id: str
    phase: str
    version: int
    key_sha256: str
    previous_state: MemoryKeyStateV1 | None
    selected_state: MemoryKeyStateV1

    def __post_init__(self) -> None:
        _require_install_id(self.install_id)
        _require_version(self.version)
        if type(self.phase) is not str or self.phase not in _PENDING_PHASES:
            raise _provider_error("memory_key_pending_invalid")
        if (
            type(self.key_sha256) is not str
            or _HEX64.fullmatch(self.key_sha256) is None
        ):
            raise _provider_error("memory_key_pending_invalid")
        if type(self.selected_state) is not MemoryKeyStateV1 or (
            self.previous_state is not None and type(self.previous_state) is not MemoryKeyStateV1
        ):
            raise _provider_error("memory_key_pending_invalid")
        if self.selected_state.install_id != self.install_id:
            raise _provider_error("memory_key_pending_invalid")
        if self.selected_state.active_version != self.version:
            raise _provider_error("memory_key_pending_invalid")
        if self.selected_state.next_version != self.version + 1:
            raise _provider_error("memory_key_pending_invalid")
        if self.previous_state is not None:
            if self.previous_state.install_id != self.install_id:
                raise _provider_error("memory_key_pending_invalid")
            if self.previous_state.active_version >= self.version:
                raise _provider_error("memory_key_pending_invalid")
            if self.previous_state.next_version != self.version:
                raise _provider_error("memory_key_pending_invalid")

    def document(self) -> dict[str, object]:
        return {
            "contract": MEMORY_KEY_PENDING_CONTRACT,
            "install_id": self.install_id,
            "key_sha256": self.key_sha256,
            "namespace": MEMORY_KEY_NAMESPACE,
            "phase": self.phase,
            "previous_state": (
                None if self.previous_state is None else self.previous_state.document()
            ),
            "selected_state": self.selected_state.document(),
            "version": self.version,
        }

    def encode(self) -> bytes:
        return jcs_bytes(self.document())

    def at_phase(self, phase: str) -> MemoryKeyPendingCreateV1:
        return MemoryKeyPendingCreateV1(
            install_id=self.install_id,
            phase=phase,
            version=self.version,
            key_sha256=self.key_sha256,
            previous_state=self.previous_state,
            selected_state=self.selected_state,
        )

    @classmethod
    def decode(cls, payload: object) -> MemoryKeyPendingCreateV1:
        if type(payload) is not bytes or not 1 <= len(payload) <= MAX_PENDING_BYTES:
            raise _provider_error("memory_key_pending_invalid")
        try:
            document = json.loads(payload.decode("utf-8", errors="strict"))
        except (ValueError, RecursionError) as exc:
            raise _provider_error("memory_key_pending_invalid", exc)
        if type(document) is not dict or set(document) != {
            "contract",
            "install_id",
            "key_sha256",
            "namespace",
            "phase",
            "previous_state",
            "selected_state",
            "version",
        }:
            raise _provider_error("memory_key_pending_invalid")
        if (
            document.get("contract") != MEMORY_KEY_PENDING_CONTRACT
            or document.get("namespace") != MEMORY_KEY_NAMESPACE
            or type(document.get("selected_state")) is not dict
        ):
            raise _provider_error("memory_key_pending_invalid")
        try:
            selected_payload = jcs_bytes(document["selected_state"])
            selected = MemoryKeyStateV1.decode(selected_payload)
            previous_document = document.get("previous_state")
            previous = (
                None
                if previous_document is None
                else MemoryKeyStateV1.decode(jcs_bytes(previous_document))
            )
            pending = cls(
                install_id=document.get("install_id"),
                phase=document.get("phase"),
                version=document.get("version"),
                key_sha256=document.get("key_sha256"),
                previous_state=previous,
                selected_state=selected,
            )
        except (MemoryKeyProviderError, TypeError, ValueError) as exc:
            raise _provider_error("memory_key_pending_invalid", exc)
        if pending.encode() != payload:
            raise _provider_error("memory_key_pending_invalid")
        return pending


class MemoryKeyBackend(Protocol):
    """Durable add-only key and atomic state operations."""

    def read_state(self) -> bytes | None: ...

    def write_state(self, payload: bytes) -> None: ...

    def read_ledger(self) -> bytes | None: ...

    def write_ledger(self, payload: bytes) -> None: ...

    def read_pending(self) -> bytes | None: ...

    def write_pending(self, payload: bytes) -> None: ...

    def delete_pending(self) -> bool: ...

    def marker_exists(self) -> bool: ...

    def create_marker(self) -> None: ...

    def delete_marker(self) -> bool: ...

    def key_exists(self, version: int) -> bool: ...

    def create_key(self, version: int, key: bytes) -> None: ...

    def load_key(self, version: int) -> bytes: ...

    def delete_key(self, version: int) -> bool: ...


class MemoryProviderLock(Protocol):
    def hold(self) -> AbstractContextManager[None]: ...


class InMemoryMemoryProviderLock:
    """Shared process-local seam for deterministic concurrency tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    @contextlib.contextmanager
    def hold(self):
        with self._lock:
            yield


class _InterprocessFileLock:
    def __init__(
        self, path: Path, *, windows_guard: WindowsPrivateGuard | None = None
    ) -> None:
        self.path = absolute_path_no_resolve(path)
        self.windows_guard = windows_guard
        self._thread_lock = threading.RLock()

    def _open_windows(self) -> int:
        if self.windows_guard is None:
            raise _provider_error("memory_key_private_state_invalid")
        self.windows_guard.validate_directory(self.path.parent)
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
        created = False
        descriptor: int | None = None
        try:
            try:
                descriptor = os.open(
                    self.path,
                    flags | os.O_CREAT | os.O_EXCL,
                    PRIVATE_FILE_MODE,
                )
                created = True
            except FileExistsError:
                self.windows_guard.validate_file(self.path)
                descriptor = os.open(self.path, flags)
            if created:
                self.windows_guard.harden_file(self.path)
            self.windows_guard.validate_file(self.path)
            self.windows_guard.validate_descriptor(descriptor)
            try:
                path_metadata = self.path.stat(follow_symlinks=False)
                descriptor_metadata = os.fstat(descriptor)
            except OSError as exc:
                raise _provider_error("memory_key_private_state_invalid", exc)
            if not os.path.samestat(path_metadata, descriptor_metadata):
                raise _provider_error("memory_key_private_state_invalid")
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\x00")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            return descriptor
        except MemoryKeyProviderError:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            raise
        except Exception as exc:
            if descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
            raise _provider_error("memory_key_lock_failed", exc)

    @contextlib.contextmanager
    def hold(self):
        with self._thread_lock:
            descriptor: int | None = None
            try:
                if os.name == "nt":
                    import msvcrt  # type: ignore[import-not-found]

                    descriptor = self._open_windows()
                    msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
                else:
                    import fcntl

                    descriptor = open_private_lock(self.path)
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
            except MemoryKeyProviderError:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                raise
            except PrivateStateError as exc:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                raise _provider_error("memory_key_private_state_invalid", exc)
            except Exception as exc:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                raise _provider_error("memory_key_lock_failed", exc)
            try:
                # Exceptions from the protected operation must propagate to
                # the provider's store/commit classifier, not be mislabeled as
                # lock-acquisition failures.
                yield
            finally:
                if descriptor is not None:
                    try:
                        if os.name == "nt":
                            import msvcrt  # type: ignore[import-not-found]

                            os.lseek(descriptor, 0, os.SEEK_SET)
                            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                        else:
                            import fcntl

                            fcntl.flock(descriptor, fcntl.LOCK_UN)
                    except OSError:
                        pass
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass


class AtomicMemoryKeyProvider:
    """Explicit-bootstrap, crash-reconciling memory epoch-key provider."""

    def __init__(
        self,
        backend: MemoryKeyBackend | None,
        lock: MemoryProviderLock,
        *,
        install_id: str | None = "0" * 64,
    ) -> None:
        self._backend = backend
        self._lock = lock
        self._install_id = (
            None if install_id is None else _require_install_id(install_id)
        )
        self._thread_lock = threading.RLock()

    def _ensure_backend(self, *, create_identity: bool) -> None:
        del create_identity
        if self._backend is None or self._install_id is None:
            raise _provider_error("memory_key_not_initialized")

    def _validate_empty_identity(self) -> None:
        """Validate durable identity evidence before a metadata-free bootstrap."""

    def _validate_identity_ledger(self, ledger: MemoryKeyLedgerV1) -> None:
        """Bind a stable provider's optional identity evidence to its ledger."""

        del ledger

    def _reserve_identity_version(
        self,
        version: int,
        *,
        initialized: bool,
        allow_committed: bool = False,
    ) -> None:
        """Durably reserve a version outside the backend key namespace."""

        del version, initialized, allow_committed

    def _commit_identity_version(self, version: int) -> None:
        """Durably bind successful initialization to identity evidence."""

        del version

    @property
    def install_id(self) -> str:
        self._ensure_backend(create_identity=False)
        if self._install_id is None:
            raise _provider_error("memory_key_not_initialized")
        return self._install_id

    @contextlib.contextmanager
    def _exclusive(self):
        with self._thread_lock:
            try:
                with self._lock.hold():
                    yield
            except MemoryKeyProviderError:
                raise
            except Exception as exc:
                raise _provider_error("memory_key_store_unavailable", exc)

    def _read_state_payload(self) -> bytes | None:
        try:
            return self._backend.read_state()
        except MemoryKeyProviderError:
            raise
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)

    def _read_ledger(self) -> MemoryKeyLedgerV1 | None:
        try:
            payload = self._backend.read_ledger()
        except MemoryKeyProviderError:
            raise
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        if payload is None:
            return None
        ledger = MemoryKeyLedgerV1.decode(payload)
        if ledger.install_id != self.install_id:
            raise _provider_error("memory_key_ledger_invalid")
        return ledger

    def _read_pending(self) -> MemoryKeyPendingCreateV1 | None:
        try:
            payload = self._backend.read_pending()
        except MemoryKeyProviderError:
            raise
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        if payload is None:
            return None
        pending = MemoryKeyPendingCreateV1.decode(payload)
        if pending.install_id != self.install_id:
            raise _provider_error("memory_key_pending_invalid")
        return pending

    def _marked(self) -> bool:
        try:
            marked = self._backend.marker_exists()
        except MemoryKeyProviderError:
            raise
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        if type(marked) is not bool:
            raise _provider_error("memory_key_store_unavailable")
        return marked

    def _checked_key(self, version: int) -> bytes:
        try:
            key = self._backend.load_key(version)
        except MemoryKeyProviderError:
            raise
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        if type(key) is not bytes or len(key) != ROOT_KEY_BYTES:
            raise _provider_error("memory_key_invalid")
        return key

    def _key_exists(self, version: int) -> bool:
        try:
            result = self._backend.key_exists(version)
        except MemoryKeyProviderError:
            raise
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        if type(result) is not bool:
            raise _provider_error("memory_key_store_unavailable")
        return result

    @staticmethod
    def _new_key() -> bytes:
        try:
            key = os.urandom(ROOT_KEY_BYTES)
        except Exception as exc:
            raise _provider_error("memory_key_create_failed", exc)
        if type(key) is not bytes or len(key) != ROOT_KEY_BYTES:
            raise _provider_error("memory_key_create_failed")
        return key

    def _write_ledger_exact(self, ledger: MemoryKeyLedgerV1) -> None:
        payload = ledger.encode()
        try:
            self._backend.write_ledger(payload)
            observed = self._backend.read_ledger()
        except Exception as exc:
            raise _provider_error("memory_key_commit_uncertain", exc)
        if observed != payload:
            raise _provider_error("memory_key_commit_uncertain")

    def _write_pending_exact(self, pending: MemoryKeyPendingCreateV1) -> None:
        payload = pending.encode()
        try:
            self._backend.write_pending(payload)
            observed = self._backend.read_pending()
        except Exception as exc:
            raise _provider_error("memory_key_commit_uncertain", exc)
        if observed != payload:
            raise _provider_error("memory_key_commit_uncertain")

    def _delete_pending_exact(self) -> None:
        try:
            deleted = self._backend.delete_pending()
        except Exception as exc:
            try:
                self._backend.read_pending()
            except Exception as read_exc:
                raise _provider_error("memory_key_commit_uncertain", read_exc)
            raise _provider_error("memory_key_commit_uncertain", exc)
        try:
            observed = self._backend.read_pending()
        except Exception as exc:
            raise _provider_error("memory_key_commit_uncertain", exc)
        if deleted is not True or observed is not None:
            raise _provider_error("memory_key_commit_uncertain")

    def _write_state_exact(self, state: MemoryKeyStateV1) -> None:
        payload = state.encode()
        try:
            self._backend.write_state(payload)
            observed = self._backend.read_state()
        except Exception as exc:
            raise _provider_error("memory_key_commit_uncertain", exc)
        if observed != payload:
            raise _provider_error("memory_key_commit_uncertain")

    def _validate_active(self, state: MemoryKeyStateV1) -> None:
        if state.install_id != self.install_id:
            raise _provider_error("memory_key_state_invalid")
        if not self._key_exists(state.active_version):
            raise _provider_error("memory_key_state_invalid")
        self._checked_key(state.active_version)

    def _reconcile_pending(
        self, pending: MemoryKeyPendingCreateV1
    ) -> MemoryKeyStateV1 | None:
        """Finish or safely burn one journaled CREATE under the provider lock."""

        marked = self._marked()
        ledger = self._read_ledger()
        if not marked or ledger is None or pending.install_id != self.install_id:
            raise _provider_error("memory_key_state_invalid")
        previous_payload = (
            None if pending.previous_state is None else pending.previous_state.encode()
        )
        selected_payload = pending.selected_state.encode()
        observed_state = self._read_state_payload()
        if observed_state not in {previous_payload, selected_payload}:
            raise _provider_error("memory_key_state_invalid")
        if pending.selected_state.next_version != pending.version + 1:
            raise _provider_error("memory_key_pending_invalid")
        if (
            pending.previous_state is not None
            and pending.previous_state.next_version != pending.version
        ):
            raise _provider_error("memory_key_pending_invalid")
        if ledger.high_water == pending.version - 1:
            ledger = MemoryKeyLedgerV1(
                install_id=self.install_id,
                high_water=pending.version,
                initialized=ledger.initialized,
            )
            self._write_ledger_exact(ledger)
        elif ledger.high_water != pending.version:
            raise _provider_error("memory_key_ledger_invalid")

        key_exists = self._key_exists(pending.version)
        self._reserve_identity_version(
            pending.version,
            initialized=pending.previous_state is not None,
            allow_committed=key_exists,
        )
        if key_exists:
            key = self._checked_key(pending.version)
            if not secrets.compare_digest(
                hashlib.sha256(key).hexdigest(), pending.key_sha256
            ):
                raise _provider_error("memory_key_state_invalid")
            if observed_state == previous_payload:
                self._write_pending_exact(pending.at_phase("key_created"))
                self._write_state_exact(pending.selected_state)
            committed_ledger = MemoryKeyLedgerV1(
                install_id=self.install_id,
                high_water=pending.version,
                initialized=True,
            )
            if ledger != committed_ledger:
                self._write_ledger_exact(committed_ledger)
            self._commit_identity_version(pending.version)
            self._write_pending_exact(pending.at_phase("state_committed"))
            self._delete_pending_exact()
            return pending.selected_state

        if observed_state == selected_payload or pending.phase in {
            "key_created",
            "state_committed",
        }:
            raise _provider_error("memory_key_state_invalid")
        if pending.previous_state is None:
            burned_state = None
        else:
            burned_state = MemoryKeyStateV1(
                install_id=self.install_id,
                active_version=pending.previous_state.active_version,
                next_version=pending.version + 1,
            )
            self._write_state_exact(burned_state)
        self._delete_pending_exact()
        return burned_state

    def _current_state(
        self,
    ) -> tuple[MemoryKeyStateV1 | None, MemoryKeyLedgerV1 | None]:
        pending = self._read_pending()
        if pending is not None:
            self._reconcile_pending(pending)
        payload = self._read_state_payload()
        marked = self._marked()
        ledger = self._read_ledger()
        if payload is None:
            if not marked and ledger is None:
                self._validate_empty_identity()
                return None, None
            if (
                marked
                and ledger is not None
                and ledger.install_id == self.install_id
                and not ledger.initialized
            ):
                self._validate_identity_ledger(ledger)
                return None, ledger
            raise _provider_error("memory_key_state_invalid")
        if not marked or ledger is None or not ledger.initialized:
            raise _provider_error("memory_key_state_invalid")
        state = MemoryKeyStateV1.decode(payload)
        if (
            state.install_id != self.install_id
            or state.next_version != ledger.high_water + 1
        ):
            raise _provider_error("memory_key_state_invalid")
        self._validate_identity_ledger(ledger)
        self._validate_active(state)
        return state, ledger

    def _prepare_bootstrap_ledger(self) -> MemoryKeyLedgerV1:
        payload = self._read_state_payload()
        pending = self._read_pending()
        marked = self._marked()
        ledger = self._read_ledger()
        if pending is not None:
            self._reconcile_pending(pending)
            payload = self._read_state_payload()
            marked = self._marked()
            ledger = self._read_ledger()
        if payload is not None:
            state, selected_ledger = self._current_state()
            if state is None or selected_ledger is None:
                raise _provider_error("memory_key_state_invalid")
            return selected_ledger
        if ledger is not None and ledger.initialized:
            raise _provider_error("memory_key_state_invalid")
        if ledger is None:
            if marked:
                # Ledger creation is ordered before the marker.  A marker by
                # itself can therefore only be evidence loss, never a valid
                # recoverable bootstrap phase.
                raise _provider_error("memory_key_state_invalid")
            self._validate_empty_identity()
            ledger = MemoryKeyLedgerV1(
                install_id=self.install_id,
                high_water=0,
                initialized=False,
            )
            self._write_ledger_exact(ledger)
        self._validate_identity_ledger(ledger)
        if not marked:
            try:
                self._backend.create_marker()
            except Exception as exc:
                try:
                    self._marked()
                except Exception as read_exc:
                    raise _provider_error("memory_key_commit_uncertain", read_exc)
                raise _provider_error("memory_key_commit_uncertain", exc)
            if not self._marked():
                raise _provider_error("memory_key_commit_uncertain")
        return ledger

    def _create_from(
        self,
        previous: MemoryKeyStateV1 | None,
        ledger: MemoryKeyLedgerV1,
    ) -> tuple[int, bytes]:
        if ledger.high_water >= MAX_KEY_VERSION:
            raise _provider_error("memory_key_version_exhausted")
        version = ledger.high_water + 1
        if previous is not None and previous.next_version != version:
            raise _provider_error("memory_key_state_invalid")
        selected = MemoryKeyStateV1(
            install_id=self.install_id,
            active_version=version,
            next_version=version + 1,
        )
        key = self._new_key()
        pending = MemoryKeyPendingCreateV1(
            install_id=self.install_id,
            phase="reserved",
            version=version,
            key_sha256=hashlib.sha256(key).hexdigest(),
            previous_state=previous,
            selected_state=selected,
        )
        if self._read_pending() is not None or self._key_exists(version):
            raise _provider_error("memory_key_state_invalid")
        self._write_pending_exact(pending)
        advanced = MemoryKeyLedgerV1(
            install_id=self.install_id,
            high_water=version,
            initialized=ledger.initialized,
        )
        self._write_ledger_exact(advanced)
        self._reserve_identity_version(
            version,
            initialized=ledger.initialized,
        )
        self._write_pending_exact(pending.at_phase("high_water_committed"))
        try:
            self._backend.create_key(version, key)
        except Exception as exc:
            try:
                exists_after = self._key_exists(version)
            except Exception as read_exc:
                raise _provider_error("memory_key_commit_uncertain", read_exc)
            if (
                isinstance(exc, MemoryKeyProviderError)
                and exc.code == "memory_key_create_failed"
                and exists_after is False
            ):
                raise
            # The exact key hash in ``pending`` is the restart-time authority;
            # a raised native write can never be reported as successful here.
            raise _provider_error("memory_key_commit_uncertain", exc)
        persisted = self._checked_key(version)
        if not secrets.compare_digest(persisted, key):
            raise _provider_error("memory_key_invalid")
        self._write_pending_exact(pending.at_phase("key_created"))
        self._write_state_exact(selected)
        self._write_ledger_exact(
            MemoryKeyLedgerV1(
                install_id=self.install_id,
                high_water=version,
                initialized=True,
            )
        )
        self._commit_identity_version(version)
        self._write_pending_exact(pending.at_phase("state_committed"))
        self._delete_pending_exact()
        return version, key

    def bootstrap(self) -> tuple[int, bytes]:
        self._ensure_backend(create_identity=True)
        with self._exclusive():
            if self._read_state_payload() is not None:
                state, _ = self._current_state()
                if state is None:
                    raise _provider_error("memory_key_state_invalid")
                return state.active_version, self._checked_key(state.active_version)
            ledger = self._prepare_bootstrap_ledger()
            # Preparing the bootstrap ledger also reconciles a durable pending
            # CREATE.  That reconciliation may have committed the first state;
            # re-read it before allocating so a retry is truly idempotent and
            # never advances an already-created installation to epoch v2.
            state, reconciled_ledger = self._current_state()
            if state is not None:
                return state.active_version, self._checked_key(state.active_version)
            if reconciled_ledger is None or reconciled_ledger != ledger:
                raise _provider_error("memory_key_state_invalid")
            return self._create_from(None, reconciled_ledger)

    def active_version(self) -> int:
        self._ensure_backend(create_identity=False)
        with self._exclusive():
            state, _ = self._current_state()
            if state is None:
                raise _provider_error("memory_key_not_initialized")
            return state.active_version

    def load(self, version: int) -> bytes:
        selected = _require_version(version)
        self._ensure_backend(create_identity=False)
        with self._exclusive():
            state, _ = self._current_state()
            if state is None:
                raise _provider_error("memory_key_not_initialized")
            if selected >= state.next_version:
                if self._key_exists(selected):
                    raise _provider_error("memory_key_state_invalid")
                raise _provider_error("memory_key_not_found")
            if not self._key_exists(selected):
                if selected == state.active_version:
                    raise _provider_error("memory_key_state_invalid")
                raise _provider_error("memory_key_not_found")
            return self._checked_key(selected)

    def create_next(self) -> tuple[int, bytes]:
        self._ensure_backend(create_identity=False)
        with self._exclusive():
            state, ledger = self._current_state()
            if state is None or ledger is None:
                raise _provider_error("memory_key_not_initialized")
            return self._create_from(state, ledger)

    def exists(self, version: int) -> bool:
        selected = _require_version(version)
        self._ensure_backend(create_identity=False)
        with self._exclusive():
            state, _ = self._current_state()
            if state is None:
                raise _provider_error("memory_key_not_initialized")
            exists = self._key_exists(selected)
            if selected >= state.next_version and exists:
                raise _provider_error("memory_key_state_invalid")
            return exists

    def _validate_destroy_candidate(self, selected: int) -> None:
        state, _ = self._current_state()
        if state is None:
            raise _provider_error("memory_key_not_initialized")
        if selected == state.active_version:
            raise _provider_error("memory_key_active_destroy_forbidden")
        if selected >= state.next_version:
            if self._key_exists(selected):
                raise _provider_error("memory_key_state_invalid")
            raise _provider_error("memory_key_not_found")
        if not self._key_exists(selected):
            raise _provider_error("memory_key_not_found")

    def destroy(self, version: int) -> None:
        selected = _require_version(version)
        self._ensure_backend(create_identity=False)
        with self._exclusive():
            self._validate_destroy_candidate(selected)
            try:
                deleted = self._backend.delete_key(selected)
                exists_after = self._backend.key_exists(selected)
            except Exception as delete_exc:
                try:
                    exists_after = self._backend.key_exists(selected)
                except Exception as verify_exc:
                    raise _provider_error("memory_key_destroy_uncertain", verify_exc)
                if exists_after is False:
                    raise _provider_error("memory_key_destroy_uncertain", delete_exc)
                if isinstance(delete_exc, MemoryKeyProviderError):
                    raise
                raise _provider_error("memory_key_destroy_failed", delete_exc)
            if deleted is not True or exists_after is not False:
                raise _provider_error("memory_key_destroy_failed")

    def attest(self) -> MemoryEpochProviderAttestationV1:
        self._ensure_backend(create_identity=False)
        with self._exclusive():
            state, ledger = self._current_state()
            if state is None or ledger is None:
                raise _provider_error("memory_key_not_initialized")
            return MemoryEpochProviderAttestationV1(
                install_id=self.install_id,
                active_version=state.active_version,
                high_water=ledger.high_water,
            )


class _AnchoredMemoryKeyProvider(AtomicMemoryKeyProvider):
    """Lazy provider whose random installation identity is bootstrap-only."""

    def __init__(
        self,
        *,
        anchor_file: _InstallationAnchorFile,
        backend_factory: Callable[[str], MemoryKeyBackend],
        lock: MemoryProviderLock,
    ) -> None:
        self._anchor_file = anchor_file
        self._backend_factory = backend_factory
        self._identity_lock = threading.RLock()
        super().__init__(None, lock, install_id=None)

    def _anchor(self) -> MemoryInstallationAnchorV1:
        anchor = self._anchor_file.read()
        if anchor is None or anchor.install_id != self.install_id:
            raise _provider_error("memory_key_state_invalid")
        return anchor

    def _validate_empty_identity(self) -> None:
        anchor = self._anchor()
        if anchor.initialized or anchor.high_water != 0:
            # An existing lifecycle anchor is independent evidence that this
            # namespace was used.  Missing backend metadata is evidence loss,
            # never authorization to reuse epoch v1.
            raise _provider_error("memory_key_state_invalid")

    def _validate_identity_ledger(self, ledger: MemoryKeyLedgerV1) -> None:
        if type(ledger) is not MemoryKeyLedgerV1:
            raise _provider_error("memory_key_ledger_invalid")
        anchor = self._anchor()
        if (
            anchor.high_water != ledger.high_water
            or anchor.initialized != ledger.initialized
        ):
            raise _provider_error("memory_key_state_invalid")

    def _reserve_identity_version(
        self,
        version: int,
        *,
        initialized: bool,
        allow_committed: bool = False,
    ) -> None:
        selected_version = _require_version(version)
        if type(initialized) is not bool or type(allow_committed) is not bool:
            raise _provider_error("memory_key_state_invalid")
        anchor = self._anchor()
        if anchor.high_water == selected_version:
            if anchor.initialized == initialized:
                return
            if allow_committed and anchor.initialized and not initialized:
                return
            raise _provider_error("memory_key_state_invalid")
        if (
            anchor.high_water != selected_version - 1
            or anchor.initialized != initialized
        ):
            raise _provider_error("memory_key_state_invalid")
        self._anchor_file.replace_exact(
            anchor,
            MemoryInstallationAnchorV1(
                install_id=self.install_id,
                high_water=selected_version,
                initialized=initialized,
            ),
        )

    def _commit_identity_version(self, version: int) -> None:
        selected_version = _require_version(version)
        anchor = self._anchor()
        if anchor.high_water != selected_version:
            raise _provider_error("memory_key_state_invalid")
        if anchor.initialized:
            return
        self._anchor_file.replace_exact(
            anchor,
            MemoryInstallationAnchorV1(
                install_id=self.install_id,
                high_water=selected_version,
                initialized=True,
            ),
        )

    def _ensure_backend(self, *, create_identity: bool) -> None:
        if self._backend is not None and self._install_id is not None:
            return
        with self._identity_lock:
            if self._backend is not None and self._install_id is not None:
                return
            try:
                with self._lock.hold():
                    anchor = self._anchor_file.read()
                    if anchor is None:
                        if not create_identity:
                            raise _provider_error("memory_key_not_initialized")
                        candidate = MemoryInstallationAnchorV1(
                            install_id=secrets.token_bytes(ROOT_KEY_BYTES).hex(),
                            high_water=0,
                            initialized=False,
                        )
                        try:
                            created = self._anchor_file.create(candidate)
                        except Exception as exc:
                            try:
                                observed = self._anchor_file.read()
                            except Exception as read_exc:
                                raise _provider_error(
                                    "memory_key_commit_uncertain", read_exc
                                )
                            if observed == candidate:
                                raise _provider_error(
                                    "memory_key_commit_uncertain", exc
                                )
                            raise _provider_error("memory_key_commit_uncertain", exc)
                        if created:
                            anchor = candidate
                        else:
                            anchor = self._anchor_file.read()
                            if anchor is None:
                                raise _provider_error("memory_key_commit_uncertain")
                    self._anchor_file.confirm_durable(anchor)
                    backend = self._backend_factory(anchor.install_id)
            except MemoryKeyProviderError:
                raise
            except Exception as exc:
                raise _provider_error("memory_key_store_unavailable", exc)
            self._install_id = anchor.install_id
            self._backend = backend


class KeychainSecretClient(Protocol):
    def read(self, account: str) -> bytes | None: ...

    def add(self, account: str, payload: bytes) -> bool: ...

    def replace(self, account: str, payload: bytes) -> None: ...

    def delete(self, account: str) -> bool: ...


class _KeychainBackend:
    def __init__(self, client: KeychainSecretClient, install_id: str) -> None:
        self.client = client
        self.install_id = _require_install_id(install_id)

    def _scoped(self, account: str) -> str:
        return f"install-{self.install_id}/{account}"

    def _account(self, version: int) -> str:
        _require_version(version)
        return self._scoped(f"epoch-v{version}")

    def read_state(self) -> bytes | None:
        return self.client.read(self._scoped(_STATE_ACCOUNT))

    def write_state(self, payload: bytes) -> None:
        MemoryKeyStateV1.decode(payload)
        account = self._scoped(_STATE_ACCOUNT)
        current = self.client.read(account)
        if current is None:
            if self.client.add(account, payload) is not True:
                raise _provider_error("memory_key_state_invalid")
        else:
            self.client.replace(account, payload)

    def read_ledger(self) -> bytes | None:
        return self.client.read(self._scoped(_LEDGER_ACCOUNT))

    def write_ledger(self, payload: bytes) -> None:
        MemoryKeyLedgerV1.decode(payload)
        account = self._scoped(_LEDGER_ACCOUNT)
        current = self.client.read(account)
        if current is None:
            if self.client.add(account, payload) is not True:
                raise _provider_error("memory_key_ledger_invalid")
        else:
            self.client.replace(account, payload)

    def read_pending(self) -> bytes | None:
        return self.client.read(self._scoped(_PENDING_ACCOUNT))

    def write_pending(self, payload: bytes) -> None:
        MemoryKeyPendingCreateV1.decode(payload)
        account = self._scoped(_PENDING_ACCOUNT)
        current = self.client.read(account)
        if current is None:
            if self.client.add(account, payload) is not True:
                raise _provider_error("memory_key_pending_invalid")
        else:
            self.client.replace(account, payload)

    def delete_pending(self) -> bool:
        return self.client.delete(self._scoped(_PENDING_ACCOUNT))

    def marker_exists(self) -> bool:
        payload = self.client.read(self._scoped(_MARKER_ACCOUNT))
        if payload is None:
            return False
        if payload != _marker_payload(self.install_id):
            raise _provider_error("memory_key_state_invalid")
        return True

    def create_marker(self) -> None:
        if (
            self.client.add(
                self._scoped(_MARKER_ACCOUNT), _marker_payload(self.install_id)
            )
            is not True
        ):
            raise _provider_error("memory_key_state_invalid")

    def delete_marker(self) -> bool:
        return self.client.delete(self._scoped(_MARKER_ACCOUNT))

    def key_exists(self, version: int) -> bool:
        return self.client.read(self._account(version)) is not None

    def create_key(self, version: int, key: bytes) -> None:
        if type(key) is not bytes or len(key) != ROOT_KEY_BYTES:
            raise _provider_error("memory_key_invalid")
        if self.client.add(self._account(version), key) is not True:
            raise _provider_error("memory_key_state_invalid")

    def load_key(self, version: int) -> bytes:
        payload = self.client.read(self._account(version))
        if payload is None:
            raise _provider_error("memory_key_not_found")
        return payload

    def delete_key(self, version: int) -> bool:
        return self.client.delete(self._account(version))


class MacOSKeychainSecretClient:
    """Exact service/account client for the current user's login Keychain.

    Source-run Python processes do not carry the application entitlement that
    macOS requires for ``kSecUseDataProtectionKeychain`` (OSStatus -34018).
    The dedicated service name keeps this explicit login-Keychain contract
    disjoint from any future signed Data Protection Keychain implementation.
    """

    _NOT_FOUND = -25300
    _DUPLICATE = -25299

    def __init__(self, service: str = MEMORY_KEYCHAIN_SERVICE) -> None:
        if service != MEMORY_KEYCHAIN_SERVICE:
            raise _provider_error("memory_key_namespace_invalid")
        self.service = service
        try:
            self._security = ctypes.CDLL(
                "/System/Library/Frameworks/Security.framework/Security"
            )
            self._cf = ctypes.CDLL(
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
            )
        except OSError as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        try:
            self._configure_ffi()
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)

    def _configure_ffi(self) -> None:
        self._cf.CFDictionaryCreateMutable.restype = ctypes.c_void_p
        self._cf.CFDictionaryCreateMutable.argtypes = [
            ctypes.c_void_p,
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._cf.CFDictionarySetValue.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        self._cf.CFStringCreateWithCString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        self._cf.CFDataCreate.restype = ctypes.c_void_p
        self._cf.CFDataCreate.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_long,
        ]
        self._cf.CFDataGetLength.restype = ctypes.c_long
        self._cf.CFDataGetLength.argtypes = [ctypes.c_void_p]
        self._cf.CFDataGetBytePtr.restype = ctypes.POINTER(ctypes.c_ubyte)
        self._cf.CFDataGetBytePtr.argtypes = [ctypes.c_void_p]
        self._cf.CFRelease.argtypes = [ctypes.c_void_p]
        self._security.SecItemAdd.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._security.SecItemAdd.restype = ctypes.c_int32
        self._security.SecItemCopyMatching.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._security.SecItemCopyMatching.restype = ctypes.c_int32
        self._security.SecItemUpdate.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._security.SecItemUpdate.restype = ctypes.c_int32
        self._security.SecItemDelete.argtypes = [ctypes.c_void_p]
        self._security.SecItemDelete.restype = ctypes.c_int32

    def _constant(self, name: str) -> ctypes.c_void_p:
        try:
            return ctypes.c_void_p.in_dll(self._security, name)
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)

    def _true(self) -> ctypes.c_void_p:
        try:
            return ctypes.c_void_p.in_dll(self._cf, "kCFBooleanTrue")
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)

    def _string(self, value: str) -> ctypes.c_void_p:
        try:
            result = self._cf.CFStringCreateWithCString(
                None, value.encode("utf-8"), 0x08000100
            )
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        if not result:
            raise _provider_error("memory_key_store_unavailable")
        return ctypes.c_void_p(result)

    def _data(self, value: bytes) -> ctypes.c_void_p:
        if type(value) is not bytes or not 1 <= len(value) <= MAX_KEYCHAIN_ITEM_BYTES:
            raise _provider_error("memory_key_blob_invalid")
        try:
            buffer = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
            result = self._cf.CFDataCreate(None, buffer, len(value))
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        if not result:
            raise _provider_error("memory_key_store_unavailable")
        return ctypes.c_void_p(result)

    def _dictionary(
        self, pairs: list[tuple[ctypes.c_void_p, ctypes.c_void_p]]
    ) -> ctypes.c_void_p:
        try:
            result = self._cf.CFDictionaryCreateMutable(None, 0, None, None)
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        if not result:
            raise _provider_error("memory_key_store_unavailable")
        dictionary = ctypes.c_void_p(result)
        try:
            for key, value in pairs:
                self._cf.CFDictionarySetValue(dictionary, key, value)
        except Exception as exc:
            try:
                self._cf.CFRelease(dictionary)
            except Exception:
                pass
            raise _provider_error("memory_key_store_unavailable", exc)
        return dictionary

    def _query(
        self, account: str, *, return_data: bool
    ) -> tuple[ctypes.c_void_p, list[ctypes.c_void_p]]:
        _require_keychain_account(account)
        service = self._string(self.service)
        try:
            account_value = self._string(account)
        except BaseException:
            self._release_all([service])
            raise
        owned = [service, account_value]
        try:
            pairs = [
                (
                    self._constant("kSecClass"),
                    self._constant("kSecClassGenericPassword"),
                ),
                (self._constant("kSecAttrService"), service),
                (self._constant("kSecAttrAccount"), account_value),
            ]
            if return_data:
                pairs.extend(
                    [
                        (self._constant("kSecReturnData"), self._true()),
                        (
                            self._constant("kSecMatchLimit"),
                            self._constant("kSecMatchLimitOne"),
                        ),
                    ]
                )
            return self._dictionary(pairs), owned
        except BaseException:
            self._release_all(owned)
            raise

    def _release_all(self, values: list[ctypes.c_void_p]) -> None:
        for value in reversed(values):
            if value and value.value:
                try:
                    self._cf.CFRelease(value)
                except Exception:
                    pass

    def read(self, account: str) -> bytes | None:
        query, owned = self._query(account, return_data=True)
        result = ctypes.c_void_p()
        try:
            try:
                status = self._security.SecItemCopyMatching(query, ctypes.byref(result))
            except Exception as exc:
                raise _provider_error("memory_key_store_unavailable", exc)
            if status == self._NOT_FOUND:
                return None
            if status != 0 or not result.value:
                raise _provider_error("memory_key_store_unavailable")
            try:
                length = self._cf.CFDataGetLength(result)
                pointer = self._cf.CFDataGetBytePtr(result)
            except Exception as exc:
                raise _provider_error("memory_key_store_unavailable", exc)
            if not 0 <= length <= MAX_KEYCHAIN_ITEM_BYTES or (length and not pointer):
                raise _provider_error("memory_key_store_unavailable")
            try:
                return bytes(pointer[index] for index in range(length))
            except Exception as exc:
                raise _provider_error("memory_key_store_unavailable", exc)
        finally:
            self._release_all([result, query, *owned])

    def add(self, account: str, payload: bytes) -> bool:
        _require_keychain_account(account)
        owned: list[ctypes.c_void_p] = []
        try:
            service = self._string(self.service)
            owned.append(service)
            account_value = self._string(account)
            owned.append(account_value)
            value = self._data(payload)
            owned.append(value)
            dictionary = self._dictionary(
                [
                    (
                        self._constant("kSecClass"),
                        self._constant("kSecClassGenericPassword"),
                    ),
                    (self._constant("kSecAttrService"), service),
                    (self._constant("kSecAttrAccount"), account_value),
                    (self._constant("kSecValueData"), value),
                ]
            )
            owned.append(dictionary)
            try:
                status = self._security.SecItemAdd(dictionary, None)
            except Exception as exc:
                raise _provider_error("memory_key_store_unavailable", exc)
            if status == self._DUPLICATE:
                return False
            if status != 0:
                raise _provider_error("memory_key_store_unavailable")
            return True
        finally:
            self._release_all(owned)

    def replace(self, account: str, payload: bytes) -> None:
        query, owned = self._query(account, return_data=False)
        try:
            value = self._data(payload)
            owned.append(value)
            additions = self._dictionary([(self._constant("kSecValueData"), value)])
            owned.append(additions)
            try:
                status = self._security.SecItemUpdate(query, additions)
            except Exception as exc:
                raise _provider_error("memory_key_store_unavailable", exc)
            if status == self._NOT_FOUND:
                raise _provider_error("memory_key_state_invalid")
            if status != 0:
                raise _provider_error("memory_key_store_unavailable")
        finally:
            self._release_all([query, *owned])

    def delete(self, account: str) -> bool:
        query, owned = self._query(account, return_data=False)
        try:
            try:
                status = self._security.SecItemDelete(query)
            except Exception as exc:
                raise _provider_error("memory_key_store_unavailable", exc)
            if status == self._NOT_FOUND:
                return False
            if status != 0:
                raise _provider_error("memory_key_store_unavailable")
            return True
        finally:
            self._release_all([query, *owned])


class MacOSMemoryKeyProvider(_AnchoredMemoryKeyProvider):
    def __init__(
        self,
        state_root: Path,
        *,
        client: KeychainSecretClient | None = None,
        lock: MemoryProviderLock | None = None,
    ) -> None:
        selected_state_root = absolute_path_no_resolve(state_root)
        keys_root = selected_state_root / "keys"
        key_root = keys_root / MEMORY_KEY_NAMESPACE
        try:
            ensure_private_directory(selected_state_root, create=True)
            ensure_private_directory(keys_root, create=True)
            ensure_private_directory(key_root, create=True)
        except PrivateStateError as exc:
            raise _provider_error("memory_key_private_state_invalid", exc)
        physical_root = _physical_key_root(key_root)
        selected_client = client if client is not None else MacOSKeychainSecretClient()
        selected_lock = lock or _InterprocessFileLock(
            physical_root / _PHYSICAL_LOCK_FILE_NAME
        )
        super().__init__(
            anchor_file=_InstallationAnchorFile(physical_root),
            backend_factory=lambda install_id: _KeychainBackend(
                selected_client, install_id
            ),
            lock=selected_lock,
        )


class DpapiProtector(Protocol):
    def protect(
        self, plaintext: bytes, *, entropy: bytes, description: str
    ) -> bytes: ...

    def unprotect(self, ciphertext: bytes, *, entropy: bytes) -> bytes: ...


class WindowsPrivateGuard(Protocol):
    def prepare_key_directory(self, state_root: Path, key_root: Path) -> None: ...

    def validate_directory(self, path: Path) -> None: ...

    def harden_file(self, path: Path) -> None: ...

    def validate_file(self, path: Path) -> None: ...

    def validate_descriptor(self, descriptor: int) -> None: ...


class NativeWindowsPrivateGuard:
    """Current-user DACL/reparse boundary for the DPAPI key directory."""

    def prepare_key_directory(self, state_root: Path, key_root: Path) -> None:
        try:
            from .windows_private_file import (
                harden_windows_private_path,
                validate_windows_private_path,
            )

            try:
                state_root.mkdir()
            except FileExistsError:
                pass
            else:
                harden_windows_private_path(state_root)
            validate_windows_private_path(state_root)
            current = state_root
            for name in ("keys", MEMORY_KEY_NAMESPACE):
                candidate = current / name
                try:
                    candidate.mkdir()
                except FileExistsError:
                    pass
                else:
                    harden_windows_private_path(candidate)
                validate_windows_private_path(candidate)
                current = candidate
            if current != key_root:
                raise RuntimeError
        except Exception as exc:
            raise _provider_error("memory_key_private_state_invalid", exc)

    def validate_directory(self, path: Path) -> None:
        try:
            from .windows_private_file import validate_windows_private_path

            validate_windows_private_path(path)
        except Exception as exc:
            raise _provider_error("memory_key_private_state_invalid", exc)

    def harden_file(self, path: Path) -> None:
        try:
            from .windows_private_file import harden_windows_private_path

            harden_windows_private_path(path)
        except Exception as exc:
            raise _provider_error("memory_key_private_state_invalid", exc)

    def validate_file(self, path: Path) -> None:
        try:
            from .windows_private_file import validate_windows_private_path

            validate_windows_private_path(path)
        except Exception as exc:
            raise _provider_error("memory_key_private_state_invalid", exc)

    def validate_descriptor(self, descriptor: int) -> None:
        try:
            from .windows_private_file import validate_windows_private_descriptor

            validate_windows_private_descriptor(descriptor)
        except Exception as exc:
            raise _provider_error("memory_key_private_state_invalid", exc)


class WindowsDpapiProtector:
    """Current-user DPAPI with UI forbidden and required optional entropy."""

    class _Blob(ctypes.Structure):
        _fields_ = [
            ("cbData", ctypes.c_ulong),
            ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
        ]

    _UI_FORBIDDEN = 0x1

    def __init__(self) -> None:
        if os.name != "nt":
            raise _provider_error("memory_key_platform_unsupported")
        try:
            self._crypt32 = ctypes.windll.crypt32
            self._kernel32 = ctypes.windll.kernel32
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        try:
            blob_pointer = ctypes.POINTER(self._Blob)
            self._crypt32.CryptProtectData.argtypes = [
                blob_pointer,
                ctypes.c_wchar_p,
                blob_pointer,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_ulong,
                blob_pointer,
            ]
            self._crypt32.CryptProtectData.restype = wintypes.BOOL
            self._crypt32.CryptUnprotectData.argtypes = [
                blob_pointer,
                ctypes.POINTER(ctypes.c_wchar_p),
                blob_pointer,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_ulong,
                blob_pointer,
            ]
            self._crypt32.CryptUnprotectData.restype = wintypes.BOOL
            self._kernel32.LocalFree.argtypes = [ctypes.c_void_p]
            self._kernel32.LocalFree.restype = ctypes.c_void_p
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)

    @classmethod
    def _blob(cls, value: bytes) -> tuple[WindowsDpapiProtector._Blob, object]:
        buffer = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
        return (
            cls._Blob(
                len(value),
                ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
            ),
            buffer,
        )

    def protect(self, plaintext: bytes, *, entropy: bytes, description: str) -> bytes:
        if (
            type(plaintext) is not bytes
            or len(plaintext) != ROOT_KEY_BYTES
            or type(entropy) is not bytes
            or len(entropy) != hashlib.sha256().digest_size
            or type(description) is not str
            or not description.isascii()
        ):
            raise _provider_error("memory_key_blob_invalid")
        source, source_keepalive = self._blob(plaintext)
        entropy_blob, entropy_keepalive = self._blob(entropy)
        target = self._Blob()
        try:
            try:
                ok = self._crypt32.CryptProtectData(
                    ctypes.byref(source),
                    description,
                    ctypes.byref(entropy_blob),
                    None,
                    None,
                    self._UI_FORBIDDEN,
                    ctypes.byref(target),
                )
            except Exception as exc:
                raise _provider_error("memory_key_store_unavailable", exc)
        finally:
            del source_keepalive, entropy_keepalive
        if not ok:
            raise _provider_error("memory_key_store_unavailable")
        if not target.pbData or not 1 <= target.cbData <= MAX_DPAPI_BLOB_BYTES // 2:
            if target.pbData:
                try:
                    self._kernel32.LocalFree(target.pbData)
                except Exception:
                    pass
            raise _provider_error("memory_key_blob_invalid")
        try:
            try:
                return ctypes.string_at(target.pbData, target.cbData)
            except Exception as exc:
                raise _provider_error("memory_key_store_unavailable", exc)
        finally:
            try:
                self._kernel32.LocalFree(target.pbData)
            except Exception:
                pass

    def unprotect(self, ciphertext: bytes, *, entropy: bytes) -> bytes:
        if (
            type(ciphertext) is not bytes
            or not ciphertext
            or len(ciphertext) > MAX_DPAPI_BLOB_BYTES // 2
            or type(entropy) is not bytes
            or len(entropy) != hashlib.sha256().digest_size
        ):
            raise _provider_error("memory_key_blob_invalid")
        source, source_keepalive = self._blob(ciphertext)
        entropy_blob, entropy_keepalive = self._blob(entropy)
        target = self._Blob()
        try:
            try:
                ok = self._crypt32.CryptUnprotectData(
                    ctypes.byref(source),
                    None,
                    ctypes.byref(entropy_blob),
                    None,
                    None,
                    self._UI_FORBIDDEN,
                    ctypes.byref(target),
                )
            except Exception as exc:
                raise _provider_error("memory_key_store_unavailable", exc)
        finally:
            del source_keepalive, entropy_keepalive
        if not ok:
            raise _provider_error("memory_key_store_unavailable")
        if not target.pbData or target.cbData != ROOT_KEY_BYTES:
            if target.pbData:
                try:
                    self._kernel32.LocalFree(target.pbData)
                except Exception:
                    pass
            raise _provider_error("memory_key_invalid")
        try:
            try:
                return ctypes.string_at(target.pbData, target.cbData)
            except Exception as exc:
                raise _provider_error("memory_key_store_unavailable", exc)
        finally:
            try:
                self._kernel32.LocalFree(target.pbData)
            except Exception:
                pass


def _dpapi_entropy(version: int, *, install_id: str) -> bytes:
    _require_version(version)
    _require_install_id(install_id)
    return hashlib.sha256(
        jcs_bytes(
            {
                "contract": MEMORY_DPAPI_ENTROPY_CONTRACT,
                "install_id": install_id,
                "namespace": MEMORY_KEY_NAMESPACE,
                "version": version,
            }
        )
    ).digest()


def _dpapi_document(version: int, protected: bytes, *, install_id: str) -> bytes:
    _require_version(version)
    _require_install_id(install_id)
    if (
        type(protected) is not bytes
        or not protected
        or len(protected) > MAX_DPAPI_BLOB_BYTES // 2
    ):
        raise _provider_error("memory_key_blob_invalid")
    payload = jcs_bytes(
        {
            "contract": MEMORY_DPAPI_BLOB_CONTRACT,
            "install_id": install_id,
            "namespace": MEMORY_KEY_NAMESPACE,
            "protected_b64": _b64url_encode(protected),
            "protected_sha256": hashlib.sha256(protected).hexdigest(),
            "version": version,
        }
    )
    if len(payload) > MAX_DPAPI_BLOB_BYTES:
        raise _provider_error("memory_key_blob_invalid")
    return payload


def _parse_dpapi_document(
    payload: object, *, expected_version: int, install_id: str
) -> bytes:
    _require_install_id(install_id)
    if type(payload) is not bytes or not 1 <= len(payload) <= MAX_DPAPI_BLOB_BYTES:
        raise _provider_error("memory_key_blob_invalid")
    try:
        document = json.loads(payload.decode("utf-8", errors="strict"))
    except (ValueError, RecursionError) as exc:
        raise _provider_error("memory_key_blob_invalid", exc)
    expected = {
        "contract",
        "install_id",
        "namespace",
        "protected_b64",
        "protected_sha256",
        "version",
    }
    if type(document) is not dict or set(document) != expected:
        raise _provider_error("memory_key_blob_invalid")
    if (
        document.get("contract") != MEMORY_DPAPI_BLOB_CONTRACT
        or document.get("install_id") != install_id
        or document.get("namespace") != MEMORY_KEY_NAMESPACE
        or type(document.get("version")) is not int
        or document.get("version") != expected_version
        or type(document.get("protected_sha256")) is not str
        or _HEX64.fullmatch(document["protected_sha256"]) is None
    ):
        raise _provider_error("memory_key_blob_invalid")
    protected = _b64url_decode(
        document.get("protected_b64"), maximum=MAX_DPAPI_BLOB_BYTES
    )
    if not secrets.compare_digest(
        hashlib.sha256(protected).hexdigest(), document["protected_sha256"]
    ):
        raise _provider_error("memory_key_blob_invalid")
    if _dpapi_document(expected_version, protected, install_id=install_id) != payload:
        raise _provider_error("memory_key_blob_invalid")
    return protected


class _WindowsFileBackend:
    def __init__(
        self,
        state_root: Path,
        protector: DpapiProtector,
        guard: WindowsPrivateGuard,
        install_id: str,
    ) -> None:
        self.state_root = absolute_path_no_resolve(state_root)
        self.root = self.state_root / "keys" / MEMORY_KEY_NAMESPACE
        self.protector = protector
        self.guard = guard
        self.install_id = _require_install_id(install_id)
        try:
            self.guard.prepare_key_directory(self.state_root, self.root)
            self.guard.validate_directory(self.root)
        except MemoryKeyProviderError:
            raise
        except Exception as exc:
            raise _provider_error("memory_key_private_state_invalid", exc)

    @staticmethod
    def _guard_call(function, /, *args) -> None:
        try:
            function(*args)
        except MemoryKeyProviderError:
            raise
        except Exception as exc:
            raise _provider_error("memory_key_private_state_invalid", exc)

    @property
    def lock_path(self) -> Path:
        return self.root / "provider.lock"

    @property
    def state_path(self) -> Path:
        return self.root / "active-state.json"

    @property
    def marker_path(self) -> Path:
        return self.root / "initialized-v1.json"

    @property
    def ledger_path(self) -> Path:
        return self.root / "used-versions-v1.json"

    @property
    def pending_path(self) -> Path:
        return self.root / "pending-create-v1.json"

    def key_path(self, version: int) -> Path:
        _require_version(version)
        return self.root / f"epoch-v{version}.dpapi.json"

    def _validate_posix_metadata(self, metadata: os.stat_result) -> None:
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise _provider_error("memory_key_private_state_invalid")
        if os.name == "posix" and (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != PRIVATE_FILE_MODE
        ):
            raise _provider_error("memory_key_private_state_invalid")

    def _read_file(self, path: Path, *, maximum: int) -> bytes | None:
        try:
            before = path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        if stat.S_ISLNK(before.st_mode):
            raise _provider_error("memory_key_private_state_invalid")
        before = _recover_staging_links(path, before)
        self._validate_posix_metadata(before)
        self._guard_call(self.guard.validate_file, path)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        if os.name == "posix":
            flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        try:
            self._guard_call(self.guard.validate_descriptor, descriptor)
            opened = os.fstat(descriptor)
            self._validate_posix_metadata(opened)
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise _provider_error("memory_key_private_state_invalid")
            if not 1 <= opened.st_size <= maximum:
                raise _provider_error("memory_key_blob_invalid")
            chunks: list[bytes] = []
            remaining = maximum + 1
            while remaining:
                try:
                    chunk = os.read(descriptor, min(4096, remaining))
                except OSError as exc:
                    raise _provider_error("memory_key_store_unavailable", exc)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) != opened.st_size or len(payload) > maximum:
                raise _provider_error("memory_key_blob_invalid")
            return payload
        finally:
            os.close(descriptor)

    def _write_all(self, descriptor: int, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            try:
                written = os.write(descriptor, view)
            except OSError as exc:
                raise _provider_error("memory_key_store_unavailable", exc)
            if written <= 0:
                raise _provider_error("memory_key_store_unavailable")
            view = view[written:]

    def _write_exclusive(self, path: Path, payload: bytes) -> None:
        self._guard_call(self.guard.validate_directory, self.root)
        temporary = self.root / f".{path.name}.{secrets.token_hex(16)}.stage"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        if os.name == "posix":
            flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(temporary, flags, PRIVATE_FILE_MODE)
        except OSError as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        closed = False
        installed = False
        try:
            try:
                if os.name == "posix":
                    os.fchmod(descriptor, PRIVATE_FILE_MODE)
                self._write_all(descriptor, payload)
                os.fsync(descriptor)
                self._guard_call(self.guard.harden_file, temporary)
                self._guard_call(self.guard.validate_descriptor, descriptor)
                self._validate_posix_metadata(os.fstat(descriptor))
                os.close(descriptor)
                closed = True
                self._guard_call(self.guard.validate_file, temporary)
                try:
                    path.lstat()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    raise _provider_error("memory_key_store_unavailable", exc)
                else:
                    raise _provider_error("memory_key_state_invalid")
                _atomic_move(temporary, path, replace=False)
                installed = True
                self._guard_call(self.guard.validate_file, path)
            except MemoryKeyProviderError:
                raise
            except Exception as exc:
                raise _provider_error("memory_key_store_unavailable", exc)
        except BaseException:
            if not closed:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if not installed:
                try:
                    temporary.unlink()
                except OSError:
                    pass
            raise

    def _replace_atomic(self, path: Path, payload: bytes, *, maximum: int) -> None:
        existing = self._read_file(path, maximum=maximum)
        if existing is None:
            self._write_exclusive(path, payload)
            return
        temporary = self.root / f".{path.name}.{secrets.token_hex(16)}.tmp"
        self._write_exclusive(temporary, payload)
        try:
            _atomic_move(temporary, path, replace=True)
            self._guard_call(self.guard.validate_file, path)
        except MemoryKeyProviderError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        except Exception as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise _provider_error("memory_key_store_unavailable", exc)

    def _delete_file(self, path: Path, *, maximum: int) -> bool:
        if self._read_file(path, maximum=maximum) is None:
            return False
        try:
            _durable_unlink(path)
        except MemoryKeyProviderError:
            raise
        except OSError as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        return True

    def _fsync_root(self) -> None:
        _fsync_directory(self.root)

    def read_state(self) -> bytes | None:
        return self._read_file(self.state_path, maximum=MAX_STATE_BYTES)

    def write_state(self, payload: bytes) -> None:
        state = MemoryKeyStateV1.decode(payload)
        if state.install_id != self.install_id:
            raise _provider_error("memory_key_state_invalid")
        self._replace_atomic(self.state_path, payload, maximum=MAX_STATE_BYTES)

    def read_ledger(self) -> bytes | None:
        return self._read_file(self.ledger_path, maximum=MAX_LEDGER_BYTES)

    def write_ledger(self, payload: bytes) -> None:
        ledger = MemoryKeyLedgerV1.decode(payload)
        if ledger.install_id != self.install_id:
            raise _provider_error("memory_key_ledger_invalid")
        self._replace_atomic(self.ledger_path, payload, maximum=MAX_LEDGER_BYTES)

    def read_pending(self) -> bytes | None:
        return self._read_file(self.pending_path, maximum=MAX_PENDING_BYTES)

    def write_pending(self, payload: bytes) -> None:
        pending = MemoryKeyPendingCreateV1.decode(payload)
        if pending.install_id != self.install_id:
            raise _provider_error("memory_key_pending_invalid")
        self._replace_atomic(self.pending_path, payload, maximum=MAX_PENDING_BYTES)

    def delete_pending(self) -> bool:
        return self._delete_file(self.pending_path, maximum=MAX_PENDING_BYTES)

    def marker_exists(self) -> bool:
        payload = self._read_file(self.marker_path, maximum=MAX_MARKER_BYTES)
        if payload is None:
            return False
        if payload != _marker_payload(self.install_id):
            raise _provider_error("memory_key_state_invalid")
        return True

    def create_marker(self) -> None:
        self._write_exclusive(self.marker_path, _marker_payload(self.install_id))

    def delete_marker(self) -> bool:
        return self._delete_file(self.marker_path, maximum=MAX_MARKER_BYTES)

    def key_exists(self, version: int) -> bool:
        payload = self._read_file(self.key_path(version), maximum=MAX_DPAPI_BLOB_BYTES)
        if payload is None:
            return False
        _parse_dpapi_document(
            payload, expected_version=version, install_id=self.install_id
        )
        return True

    def create_key(self, version: int, key: bytes) -> None:
        if type(key) is not bytes or len(key) != ROOT_KEY_BYTES:
            raise _provider_error("memory_key_invalid")
        try:
            protected = self.protector.protect(
                key,
                entropy=_dpapi_entropy(version, install_id=self.install_id),
                description=f"JARVIS {MEMORY_KEY_NAMESPACE} epoch v{version}",
            )
        except MemoryKeyProviderError:
            raise
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        self._write_exclusive(
            self.key_path(version),
            _dpapi_document(version, protected, install_id=self.install_id),
        )

    def load_key(self, version: int) -> bytes:
        payload = self._read_file(self.key_path(version), maximum=MAX_DPAPI_BLOB_BYTES)
        if payload is None:
            raise _provider_error("memory_key_not_found")
        protected = _parse_dpapi_document(
            payload, expected_version=version, install_id=self.install_id
        )
        try:
            key = self.protector.unprotect(
                protected,
                entropy=_dpapi_entropy(version, install_id=self.install_id),
            )
        except MemoryKeyProviderError:
            raise
        except Exception as exc:
            raise _provider_error("memory_key_store_unavailable", exc)
        if type(key) is not bytes or len(key) != ROOT_KEY_BYTES:
            raise _provider_error("memory_key_invalid")
        return key

    def delete_key(self, version: int) -> bool:
        return self._delete_file(self.key_path(version), maximum=MAX_DPAPI_BLOB_BYTES)


class WindowsMemoryKeyProvider(_AnchoredMemoryKeyProvider):
    def __init__(
        self,
        state_root: Path,
        *,
        protector: DpapiProtector | None = None,
        guard: WindowsPrivateGuard | None = None,
        lock: MemoryProviderLock | None = None,
    ) -> None:
        selected_guard = guard if guard is not None else NativeWindowsPrivateGuard()
        selected_protector = (
            protector if protector is not None else WindowsDpapiProtector()
        )
        selected_root = absolute_path_no_resolve(state_root)
        key_root = selected_root / "keys" / MEMORY_KEY_NAMESPACE
        try:
            selected_guard.prepare_key_directory(selected_root, key_root)
            selected_guard.validate_directory(key_root)
        except MemoryKeyProviderError:
            raise
        except Exception as exc:
            raise _provider_error("memory_key_private_state_invalid", exc)
        physical_root = _physical_key_root(key_root)
        selected_lock = lock or _InterprocessFileLock(
            physical_root / _PHYSICAL_LOCK_FILE_NAME,
            windows_guard=selected_guard,
        )
        super().__init__(
            anchor_file=_InstallationAnchorFile(
                physical_root,
                windows_guard=selected_guard,
            ),
            backend_factory=lambda install_id: _WindowsFileBackend(
                selected_root,
                selected_protector,
                selected_guard,
                install_id,
            ),
            lock=selected_lock,
        )

    def destroy(self, version: int) -> None:
        """Block verified-erasure claims until Windows deletion is durable."""

        selected = _require_version(version)
        self._ensure_backend(create_identity=False)
        with self._exclusive():
            self._validate_destroy_candidate(selected)
            raise _provider_error("memory_key_destroy_not_durable")


def platform_memory_key_provider(
    *,
    state_root: Path,
    platform: str | None = None,
    keychain_client: KeychainSecretClient | None = None,
    dpapi_protector: DpapiProtector | None = None,
    windows_guard: WindowsPrivateGuard | None = None,
    lock: MemoryProviderLock | None = None,
) -> AtomicMemoryKeyProvider:
    """Select one production provider; injection is exclusively a test seam."""

    selected = sys.platform if platform is None else platform
    if selected == "darwin":
        if dpapi_protector is not None or windows_guard is not None:
            raise _provider_error("memory_key_provider_arguments_invalid")
        return MacOSMemoryKeyProvider(
            state_root,
            client=keychain_client,
            lock=lock,
        )
    if selected == "win32":
        if keychain_client is not None:
            raise _provider_error("memory_key_provider_arguments_invalid")
        return WindowsMemoryKeyProvider(
            state_root,
            protector=dpapi_protector,
            guard=windows_guard,
            lock=lock,
        )
    raise _provider_error("memory_key_platform_unsupported")


__all__ = [
    "MEMORY_KEYCHAIN_SERVICE",
    "MEMORY_KEY_NAMESPACE",
    "AtomicMemoryKeyProvider",
    "DpapiProtector",
    "InMemoryMemoryProviderLock",
    "KeychainSecretClient",
    "MacOSKeychainSecretClient",
    "MacOSMemoryKeyProvider",
    "MemoryInstallationAnchorV1",
    "MemoryKeyBackend",
    "MemoryKeyLedgerV1",
    "MemoryKeyPendingCreateV1",
    "MemoryKeyProviderError",
    "MemoryKeyStateV1",
    "MemoryProviderLock",
    "NativeWindowsPrivateGuard",
    "WindowsDpapiProtector",
    "WindowsMemoryKeyProvider",
    "WindowsPrivateGuard",
    "platform_memory_key_provider",
]
