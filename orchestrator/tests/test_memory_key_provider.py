from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator.jcs import jcs_bytes
from jarvis_orchestrator.memory_key_provider import (
    MEMORY_KEY_NAMESPACE,
    AtomicMemoryKeyProvider,
    InMemoryMemoryProviderLock,
    MacOSMemoryKeyProvider,
    MemoryInstallationAnchorV1,
    MemoryKeyLedgerV1,
    MemoryKeyPendingCreateV1,
    MemoryKeyProviderError,
    MemoryKeyStateV1,
    NativeWindowsPrivateGuard,
    WindowsDpapiProtector,
    WindowsMemoryKeyProvider,
    _atomic_move,
    _dpapi_document,
    _InterprocessFileLock,
    _parse_dpapi_document,
    platform_memory_key_provider,
)

INSTALL_ID = "0" * 64


def state(active: int, next_version: int) -> MemoryKeyStateV1:
    return MemoryKeyStateV1(
        install_id=INSTALL_ID,
        active_version=active,
        next_version=next_version,
    )


class FakeBackend:
    def __init__(self) -> None:
        self.state: bytes | None = None
        self.ledger: bytes | None = None
        self.pending: bytes | None = None
        self.marker = False
        self.keys: dict[int, bytes] = {}
        self.locked = False
        self.write_failure: str | None = None
        self.ledger_write_failure: str | None = None
        self.pending_write_failure: str | None = None
        self.pending_delete_failure: str | None = None
        self.fail_delete_versions: set[int] = set()
        self.lie_delete_versions: set[int] = set()
        self.delete_after_error_versions: set[int] = set()
        self.corrupt_create_versions: set[int] = set()
        self.fail_create_versions: set[int] = set()
        self.create_after_error_versions: set[int] = set()

    def check(self) -> None:
        if self.locked:
            raise MemoryKeyProviderError("memory_key_store_unavailable")

    def read_state(self) -> bytes | None:
        self.check()
        return self.state

    def write_state(self, payload: bytes) -> None:
        self.check()
        if self.write_failure == "before":
            raise MemoryKeyProviderError("memory_key_store_unavailable")
        self.state = bytes(payload)
        if self.write_failure == "after":
            raise MemoryKeyProviderError("memory_key_store_unavailable")

    def read_ledger(self) -> bytes | None:
        self.check()
        return self.ledger

    def write_ledger(self, payload: bytes) -> None:
        self.check()
        if self.ledger_write_failure == "before":
            raise MemoryKeyProviderError("memory_key_durability_unavailable")
        self.ledger = bytes(payload)
        if self.ledger_write_failure == "after":
            raise MemoryKeyProviderError("memory_key_durability_unavailable")

    def read_pending(self) -> bytes | None:
        self.check()
        return self.pending

    def write_pending(self, payload: bytes) -> None:
        self.check()
        if self.pending_write_failure == "before":
            raise MemoryKeyProviderError("memory_key_durability_unavailable")
        self.pending = bytes(payload)
        if self.pending_write_failure == "after":
            raise MemoryKeyProviderError("memory_key_durability_unavailable")

    def delete_pending(self) -> bool:
        self.check()
        if self.pending_delete_failure == "before":
            raise MemoryKeyProviderError("memory_key_durability_unavailable")
        if self.pending is None:
            return False
        self.pending = None
        if self.pending_delete_failure == "after":
            raise MemoryKeyProviderError("memory_key_durability_unavailable")
        return True

    def marker_exists(self) -> bool:
        self.check()
        return self.marker

    def create_marker(self) -> None:
        self.check()
        if self.marker:
            raise MemoryKeyProviderError("memory_key_state_invalid")
        self.marker = True

    def delete_marker(self) -> bool:
        self.check()
        if not self.marker:
            return False
        self.marker = False
        return True

    def key_exists(self, version: int) -> bool:
        self.check()
        return version in self.keys

    def create_key(self, version: int, key: bytes) -> None:
        self.check()
        if version in self.fail_create_versions:
            raise MemoryKeyProviderError("memory_key_create_failed")
        if version in self.keys:
            raise MemoryKeyProviderError("memory_key_state_invalid")
        self.keys[version] = (
            b"z" * 32 if version in self.corrupt_create_versions else bytes(key)
        )
        if version in self.create_after_error_versions:
            raise MemoryKeyProviderError("memory_key_durability_unavailable")

    def load_key(self, version: int) -> bytes:
        self.check()
        try:
            return self.keys[version]
        except KeyError as exc:
            raise MemoryKeyProviderError("memory_key_not_found") from exc

    def delete_key(self, version: int) -> bool:
        self.check()
        if version in self.delete_after_error_versions:
            self.keys.pop(version, None)
            raise MemoryKeyProviderError("memory_key_store_unavailable")
        if version in self.fail_delete_versions:
            raise MemoryKeyProviderError("memory_key_store_unavailable")
        if version in self.lie_delete_versions:
            return True
        return self.keys.pop(version, None) is not None


class FakeKeychainClient:
    def __init__(self) -> None:
        self.items: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.locked = False
        self._lock = threading.RLock()

    def check(self) -> None:
        if self.locked:
            raise MemoryKeyProviderError("memory_key_store_unavailable")

    def read(self, account: str) -> bytes | None:
        with self._lock:
            self.check()
            value = self.items.get(account)
            return None if value is None else bytes(value)

    def add(self, account: str, payload: bytes) -> bool:
        with self._lock:
            self.check()
            if account in self.items:
                return False
            self.items[account] = bytes(payload)
            return True

    def replace(self, account: str, payload: bytes) -> None:
        with self._lock:
            self.check()
            if account not in self.items:
                raise MemoryKeyProviderError("memory_key_state_invalid")
            self.items[account] = bytes(payload)

    def delete(self, account: str) -> bool:
        with self._lock:
            self.check()
            self.deleted.append(account)
            return self.items.pop(account, None) is not None


class FakeDpapiProtector:
    def __init__(self) -> None:
        self.locked = False
        self.descriptions: list[str] = []

    def check(self) -> None:
        if self.locked:
            raise MemoryKeyProviderError("memory_key_store_unavailable")

    def protect(self, plaintext: bytes, *, entropy: bytes, description: str) -> bytes:
        self.check()
        self.descriptions.append(description)
        masked = bytes(
            value ^ entropy[index % len(entropy)]
            for index, value in enumerate(plaintext)
        )
        tag = hmac.new(entropy, plaintext, hashlib.sha256).digest()[:16]
        return b"FAKE-DPAPI-1\x00" + tag + masked

    def unprotect(self, ciphertext: bytes, *, entropy: bytes) -> bytes:
        self.check()
        prefix = b"FAKE-DPAPI-1\x00"
        if not ciphertext.startswith(prefix) or len(ciphertext) != len(prefix) + 48:
            raise MemoryKeyProviderError("memory_key_store_unavailable")
        tag = ciphertext[len(prefix) : len(prefix) + 16]
        masked = ciphertext[len(prefix) + 16 :]
        plaintext = bytes(
            value ^ entropy[index % len(entropy)] for index, value in enumerate(masked)
        )
        expected = hmac.new(entropy, plaintext, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(tag, expected):
            raise MemoryKeyProviderError("memory_key_store_unavailable")
        return plaintext


class FakeWindowsPrivateGuard:
    def prepare_key_directory(self, state_root: Path, key_root: Path) -> None:
        state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(state_root, 0o700)
        current = state_root
        for name in ("keys", MEMORY_KEY_NAMESPACE):
            current /= name
            current.mkdir(mode=0o700, exist_ok=True)
            os.chmod(current, 0o700)
        if current != key_root:
            raise MemoryKeyProviderError("memory_key_private_state_invalid")

    def validate_directory(self, path: Path) -> None:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or (
                os.name == "posix"
                and stat.S_IMODE(metadata.st_mode) != 0o700
            )
        ):
            raise MemoryKeyProviderError("memory_key_private_state_invalid")

    def harden_file(self, path: Path) -> None:
        os.chmod(path, 0o600, follow_symlinks=False)

    def validate_file(self, path: Path) -> None:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or (
                os.name == "posix"
                and stat.S_IMODE(metadata.st_mode) != 0o600
            )
        ):
            raise MemoryKeyProviderError("memory_key_private_state_invalid")

    def validate_descriptor(self, descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (
                os.name == "posix"
                and stat.S_IMODE(metadata.st_mode) != 0o600
            )
        ):
            raise MemoryKeyProviderError("memory_key_private_state_invalid")


class MemoryKeyProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.state_root = Path(self.temporary.name) / "state"
        self.state_root.mkdir(mode=0o700)
        os.chmod(self.state_root, 0o700)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_code(self, code: str, function, /, *args, **kwargs) -> None:
        with self.assertRaises(MemoryKeyProviderError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)
        self.assertIsNone(caught.exception.__cause__)
        self.assertTrue(caught.exception.__suppress_context__)

    def generic(self, backend: FakeBackend | None = None):
        selected = backend or FakeBackend()
        return (
            AtomicMemoryKeyProvider(selected, InMemoryMemoryProviderLock()),
            selected,
        )

    def windows(self):
        protector = FakeDpapiProtector()
        guard = FakeWindowsPrivateGuard()
        provider = WindowsMemoryKeyProvider(
            self.state_root,
            protector=protector,
            guard=guard,
        )
        return provider, protector, guard

    def test_fresh_provider_initializes_once_and_persists_exact_pointer(self) -> None:
        provider, backend = self.generic()
        self.assert_code("memory_key_not_initialized", provider.active_version)
        self.assert_code("memory_key_not_initialized", provider.create_next)
        version, first = provider.bootstrap()
        self.assertEqual(version, 1)
        first = provider.load(1)
        self.assertEqual(len(first), 32)
        self.assertTrue(backend.marker)
        self.assertEqual(
            MemoryKeyStateV1.decode(backend.state),
            state(1, 2),
        )
        self.assertEqual(
            MemoryKeyLedgerV1.decode(backend.ledger),
            MemoryKeyLedgerV1(
                install_id=INSTALL_ID,
                high_water=1,
                initialized=True,
            ),
        )
        self.assertIsNone(backend.pending)
        self.assertEqual(provider.active_version(), 1)
        self.assertEqual(provider.bootstrap(), (1, first))
        self.assertEqual(provider.load(1), first)

    def test_attestation_exposes_only_reconciled_identity_and_versions(self) -> None:
        provider, backend = self.generic()
        self.assert_code("memory_key_not_initialized", provider.attest)
        provider.bootstrap()
        provider.create_next()
        attestation = provider.attest()
        self.assertEqual(attestation.install_id, INSTALL_ID)
        self.assertEqual(attestation.active_version, 2)
        self.assertEqual(attestation.high_water, 2)
        self.assertEqual(attestation.reconciliation_state, "clean")
        self.assertNotIn(backend.keys[2].hex(), repr(attestation))

    def test_bootstrap_repairs_only_ordered_prekey_ledger_phase(self) -> None:
        marker_backend = FakeBackend()
        marker_backend.marker = True
        marker_provider, _ = self.generic(marker_backend)
        self.assert_code("memory_key_state_invalid", marker_provider.bootstrap)

        ledger_backend = FakeBackend()
        ledger_backend.ledger = MemoryKeyLedgerV1(
            install_id=INSTALL_ID,
            high_water=0,
            initialized=False,
        ).encode()
        ledger_provider, _ = self.generic(ledger_backend)
        self.assertEqual(ledger_provider.bootstrap()[0], 1)
        self.assertTrue(ledger_backend.marker)

    def test_ledger_and_pending_documents_are_exact_closed_types(self) -> None:
        pending = MemoryKeyPendingCreateV1(
            install_id=INSTALL_ID,
            phase="reserved",
            version=2,
            key_sha256="a" * 64,
            previous_state=state(1, 2),
            selected_state=state(2, 3),
        )
        self.assertEqual(MemoryKeyPendingCreateV1.decode(pending.encode()), pending)
        malformed_pending = pending.document()
        malformed_pending["version"] = True
        self.assert_code(
            "memory_key_pending_invalid",
            MemoryKeyPendingCreateV1.decode,
            jcs_bytes(malformed_pending),
        )
        malformed_ledger = MemoryKeyLedgerV1(
            install_id=INSTALL_ID,
            high_water=1,
            initialized=True,
        ).document()
        malformed_ledger["initialized"] = 1
        self.assert_code(
            "memory_key_ledger_invalid",
            MemoryKeyLedgerV1.decode,
            jcs_bytes(malformed_ledger),
        )

    def test_create_next_is_unique_under_shared_lock_race(self) -> None:
        backend = FakeBackend()
        shared_lock = InMemoryMemoryProviderLock()
        providers = [AtomicMemoryKeyProvider(backend, shared_lock) for _ in range(24)]
        self.assertEqual(providers[0].bootstrap()[0], 1)
        barrier = threading.Barrier(len(providers))

        def rotate(provider: AtomicMemoryKeyProvider) -> int:
            barrier.wait()
            return provider.create_next()[0]

        with ThreadPoolExecutor(max_workers=len(providers)) as pool:
            versions = list(pool.map(rotate, providers))
        self.assertEqual(sorted(versions), list(range(2, 26)))
        self.assertEqual(providers[0].active_version(), 25)
        self.assertEqual(len(backend.keys), 25)

    def test_file_lock_serializes_version_allocation_across_processes(self) -> None:
        provider, _, _ = self.windows()
        self.assertEqual(provider.bootstrap()[0], 1)
        script = """
import sys
from pathlib import Path
from test_memory_key_provider import FakeDpapiProtector, FakeWindowsPrivateGuard
from jarvis_orchestrator.memory_key_provider import WindowsMemoryKeyProvider
provider = WindowsMemoryKeyProvider(
    Path(sys.argv[1]),
    protector=FakeDpapiProtector(),
    guard=FakeWindowsPrivateGuard(),
)
print(provider.create_next()[0], flush=True)
"""
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(self.state_root)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path(__file__).parent,
            )
            for _ in range(8)
        ]
        versions: list[int] = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, stderr)
            versions.append(int(stdout.strip()))
        self.assertEqual(sorted(versions), list(range(2, 10)))
        self.assertEqual(provider.active_version(), 9)

    def test_file_lock_does_not_relabel_protected_operation_failures(self) -> None:
        lock_root = self.state_root / "lock-domain"
        lock_root.mkdir(mode=0o700)
        lock = _InterprocessFileLock(lock_root / "provider.lock")
        with self.assertRaisesRegex(ValueError, "body_failure"), lock.hold():
            raise ValueError("body_failure")

    def test_windows_lock_closes_descriptor_on_validation_failure(self) -> None:
        class RejectingGuard(FakeWindowsPrivateGuard):
            rejected_descriptor: int | None = None

            def validate_descriptor(self, descriptor: int) -> None:
                super().validate_descriptor(descriptor)
                self.rejected_descriptor = descriptor
                raise MemoryKeyProviderError("memory_key_private_state_invalid")

        lock_root = self.state_root / "windows-lock-validation"
        lock_root.mkdir(mode=0o700)
        os.chmod(lock_root, 0o700)
        guard = RejectingGuard()
        lock = _InterprocessFileLock(
            lock_root / "provider.lock",
            windows_guard=guard,
        )
        with self.assertRaisesRegex(
            MemoryKeyProviderError, "memory_key_private_state_invalid"
        ):
            lock._open_windows()
        assert guard.rejected_descriptor is not None
        with self.assertRaises(OSError):
            os.fstat(guard.rejected_descriptor)

    def test_file_lock_closes_descriptor_when_acquisition_fails(self) -> None:
        lock_root = self.state_root / "lock-acquisition"
        lock_root.mkdir(mode=0o700)
        os.chmod(lock_root, 0o700)
        lock = _InterprocessFileLock(lock_root / "provider.lock")
        descriptor = os.open(lock_root / "held.lock", os.O_CREAT | os.O_RDWR, 0o600)
        fake_msvcrt = SimpleNamespace(
            LK_LOCK=1,
            LK_UNLCK=2,
            locking=lambda *_args: (_ for _ in ()).throw(OSError("busy")),
        )
        with (
            patch("jarvis_orchestrator.memory_key_provider.os.name", "nt"),
            patch.object(lock, "_open_windows", return_value=descriptor),
            patch.dict(sys.modules, {"msvcrt": fake_msvcrt}),
            self.assertRaisesRegex(MemoryKeyProviderError, "memory_key_lock_failed"),
            lock.hold(),
        ):
            pass
        with self.assertRaises(OSError):
            os.fstat(descriptor)

    def test_precommit_pointer_failure_is_reconciled_without_version_reuse(
        self,
    ) -> None:
        provider, backend = self.generic()
        provider.bootstrap()
        backend.write_failure = "before"
        self.assert_code("memory_key_commit_uncertain", provider.create_next)
        self.assertIn(2, backend.keys)
        self.assertIsNotNone(backend.pending)
        self.assertEqual(MemoryKeyStateV1.decode(backend.state).active_version, 1)
        backend.write_failure = None
        self.assertEqual(provider.active_version(), 2)
        self.assertIsNone(backend.pending)
        self.assertEqual(provider.create_next()[0], 3)

    def test_postcommit_pointer_failure_reconciles_exact_committed_write(
        self,
    ) -> None:
        provider, backend = self.generic()
        provider.bootstrap()
        backend.write_failure = "after"
        self.assert_code("memory_key_commit_uncertain", provider.create_next)
        self.assertIn(2, backend.keys)
        self.assertEqual(MemoryKeyStateV1.decode(backend.state).active_version, 2)
        self.assertIsNotNone(backend.pending)
        backend.write_failure = None
        self.assertEqual(provider.active_version(), 2)
        self.assertIsNone(backend.pending)

    def test_pending_write_after_commit_is_uncertain_and_journal_recovers(self) -> None:
        provider, backend = self.generic()
        provider.bootstrap()
        backend.pending_write_failure = "after"
        self.assert_code("memory_key_commit_uncertain", provider.create_next)
        self.assertIsNotNone(backend.pending)
        self.assertNotIn(2, backend.keys)
        backend.pending_write_failure = None
        self.assertEqual(provider.active_version(), 1)
        self.assertEqual(MemoryKeyStateV1.decode(backend.state).next_version, 3)
        self.assertEqual(provider.create_next()[0], 3)

    def test_ledger_write_after_commit_is_uncertain_and_journal_recovers(self) -> None:
        provider, backend = self.generic()
        provider.bootstrap()
        backend.ledger_write_failure = "after"
        self.assert_code("memory_key_commit_uncertain", provider.create_next)
        self.assertIsNotNone(backend.pending)
        self.assertEqual(MemoryKeyLedgerV1.decode(backend.ledger).high_water, 2)
        backend.ledger_write_failure = None
        self.assertEqual(provider.active_version(), 1)
        self.assertEqual(provider.create_next()[0], 3)

    def test_pending_delete_after_commit_never_reports_false_success(self) -> None:
        provider, backend = self.generic()
        provider.bootstrap()
        backend.pending_delete_failure = "after"
        self.assert_code("memory_key_commit_uncertain", provider.create_next)
        self.assertIsNone(backend.pending)
        self.assertEqual(MemoryKeyStateV1.decode(backend.state).active_version, 2)
        backend.pending_delete_failure = None
        self.assertEqual(provider.active_version(), 2)

    def test_failed_bootstrap_is_journaled_and_reconciled(self) -> None:
        provider, backend = self.generic()
        backend.write_failure = "before"
        self.assert_code("memory_key_commit_uncertain", provider.bootstrap)
        self.assertTrue(backend.marker)
        self.assertIn(1, backend.keys)
        self.assertIsNone(backend.state)
        self.assertIsNotNone(backend.pending)
        backend.write_failure = None
        version, key = provider.bootstrap()
        self.assertEqual(version, 1)
        self.assertEqual(key, backend.keys[1])
        self.assertEqual(set(backend.keys), {1})
        self.assertEqual(provider.active_version(), 1)
        self.assertIsNone(backend.pending)

    def test_generated_and_persisted_key_are_verified_before_pointer_commit(
        self,
    ) -> None:
        provider, backend = self.generic()
        backend.corrupt_create_versions.add(1)
        self.assert_code("memory_key_invalid", provider.bootstrap)
        self.assertTrue(backend.marker)
        self.assertIn(1, backend.keys)
        self.assertIsNone(backend.state)

        clean_provider, clean_backend = self.generic()
        with patch(
            "jarvis_orchestrator.memory_key_provider.os.urandom",
            return_value=b"short",
        ):
            self.assert_code("memory_key_create_failed", clean_provider.bootstrap)
        self.assertTrue(clean_backend.marker)
        self.assertEqual(clean_backend.keys, {})
        self.assertEqual(
            MemoryKeyLedgerV1.decode(clean_backend.ledger).high_water,
            0,
        )

    def test_crash_orphan_is_recovered_from_pending_create(self) -> None:
        provider, backend = self.generic()
        provider.bootstrap()
        backend.write_failure = "before"
        self.assert_code("memory_key_commit_uncertain", provider.create_next)
        self.assertIn(2, backend.keys)
        self.assertIsNotNone(backend.pending)
        backend.write_failure = None
        self.assertEqual(provider.active_version(), 2)
        self.assertTrue(provider.exists(2))
        self.assertIsNone(backend.pending)

    def test_reserved_version_without_key_is_burned_and_never_reused(self) -> None:
        provider, backend = self.generic()
        provider.bootstrap()
        backend.fail_create_versions.add(2)
        self.assert_code("memory_key_create_failed", provider.create_next)
        self.assertIsNotNone(backend.pending)
        self.assertNotIn(2, backend.keys)
        backend.fail_create_versions.clear()
        self.assertEqual(provider.active_version(), 1)
        self.assertEqual(MemoryKeyStateV1.decode(backend.state).next_version, 3)
        self.assertEqual(MemoryKeyLedgerV1.decode(backend.ledger).high_water, 2)
        self.assertEqual(provider.create_next()[0], 3)

    def test_key_create_after_commit_is_uncertain_and_journal_recovers(self) -> None:
        provider, backend = self.generic()
        provider.bootstrap()
        backend.create_after_error_versions.add(2)
        self.assert_code("memory_key_commit_uncertain", provider.create_next)
        self.assertIn(2, backend.keys)
        self.assertIsNotNone(backend.pending)
        backend.create_after_error_versions.clear()
        self.assertEqual(provider.active_version(), 2)
        self.assertIsNone(backend.pending)

    def test_restart_finalizes_exact_state_committed_pending_create(self) -> None:
        provider, backend = self.generic()
        provider.bootstrap()
        key2 = b"k" * 32
        selected = state(2, 3)
        pending = MemoryKeyPendingCreateV1(
            install_id=INSTALL_ID,
            phase="key_created",
            version=2,
            key_sha256=hashlib.sha256(key2).hexdigest(),
            previous_state=state(1, 2),
            selected_state=selected,
        )
        backend.keys[2] = key2
        backend.state = selected.encode()
        backend.ledger = MemoryKeyLedgerV1(
            install_id=INSTALL_ID,
            high_water=2,
            initialized=True,
        ).encode()
        backend.pending = pending.encode()
        self.assertEqual(provider.active_version(), 2)
        self.assertIsNone(backend.pending)
        self.assertEqual(provider.load(2), key2)

    def test_high_water_prevents_reuse_when_state_and_marker_are_lost(self) -> None:
        provider, backend = self.generic()
        provider.bootstrap()
        provider.create_next()
        provider.destroy(1)
        backend.state = None
        backend.marker = False
        self.assert_code("memory_key_state_invalid", provider.active_version)
        self.assert_code("memory_key_state_invalid", provider.bootstrap)
        self.assertNotIn(1, backend.keys)
        self.assertIn(2, backend.keys)
        self.assertEqual(MemoryKeyLedgerV1.decode(backend.ledger).high_water, 2)

    def test_marker_without_ledger_never_authorizes_rebootstrap(self) -> None:
        provider, backend = self.generic()
        provider.bootstrap()
        provider.create_next()
        backend.state = None
        backend.ledger = None
        self.assertTrue(backend.marker)
        self.assert_code("memory_key_state_invalid", provider.bootstrap)
        self.assertEqual(set(backend.keys), {1, 2})

    def test_malformed_pointer_marker_and_active_key_fail_closed(self) -> None:
        provider, backend = self.generic()
        provider.bootstrap()
        backend.state = b"{}"
        self.assert_code("memory_key_state_invalid", provider.active_version)

        backend.state = state(1, 2).encode()
        backend.marker = False
        self.assert_code("memory_key_state_invalid", provider.active_version)

        backend.marker = True
        backend.keys[1] = b"short"
        self.assert_code("memory_key_invalid", provider.active_version)

        backend.keys[1] = b"x" * 32
        backend.ledger = MemoryKeyLedgerV1(
            install_id=INSTALL_ID,
            high_water=2,
            initialized=True,
        ).encode()
        self.assert_code("memory_key_state_invalid", provider.active_version)

    def test_locked_store_is_not_misreported_as_missing(self) -> None:
        provider, backend = self.generic()
        backend.locked = True
        self.assert_code("memory_key_store_unavailable", provider.active_version)
        self.assert_code("memory_key_store_unavailable", provider.exists, 1)

    def test_destroy_forbids_active_rejects_absent_and_verifies_delete(self) -> None:
        provider, backend = self.generic()
        provider.bootstrap()
        self.assert_code("memory_key_active_destroy_forbidden", provider.destroy, 1)
        provider.create_next()
        provider.destroy(1)
        self.assertFalse(provider.exists(1))
        self.assert_code("memory_key_not_found", provider.destroy, 1)
        self.assert_code("memory_key_active_destroy_forbidden", provider.destroy, 2)

        backend.keys[1] = b"x" * 32
        backend.lie_delete_versions.add(1)
        self.assert_code("memory_key_destroy_failed", provider.destroy, 1)

    def test_delete_that_commits_then_errors_is_truthfully_uncertain(self) -> None:
        provider, backend = self.generic()
        provider.bootstrap()
        provider.create_next()
        backend.delete_after_error_versions.add(1)
        self.assert_code("memory_key_destroy_uncertain", provider.destroy, 1)
        self.assertNotIn(1, backend.keys)

    def test_mac_keychain_accounts_are_exact_and_active_pointer_is_not_inferred(
        self,
    ) -> None:
        client = FakeKeychainClient()
        provider = MacOSMemoryKeyProvider(
            self.state_root,
            client=client,
            lock=InMemoryMemoryProviderLock(),
        )
        version1, key1 = provider.bootstrap()
        version2, key2 = provider.create_next()
        self.assertEqual((version1, version2), (1, 2))
        self.assertEqual(provider.load(1), key1)
        self.assertEqual(provider.load(2), key2)
        prefix = f"install-{provider.install_id}/"
        self.assertEqual(
            set(client.items),
            {
                prefix + "active-state",
                prefix + "initialized-v1",
                prefix + "used-versions-v1",
                prefix + "epoch-v1",
                prefix + "epoch-v2",
            },
        )
        client.items[prefix + "epoch-v3"] = b"o" * 32
        self.assertEqual(provider.active_version(), 2)
        self.assert_code("memory_key_state_invalid", provider.exists, 3)
        client.items.pop(prefix + "epoch-v3")
        provider.destroy(1)
        self.assertEqual(client.deleted[-1], prefix + "epoch-v1")
        self.assertNotIn(prefix + "epoch-v1", client.items)

    def test_anchor_durability_error_is_uncertain_and_retry_reuses_identity(
        self,
    ) -> None:
        client = FakeKeychainClient()
        provider = MacOSMemoryKeyProvider(
            self.state_root,
            client=client,
            lock=InMemoryMemoryProviderLock(),
        )
        with patch.object(
            provider._anchor_file,
            "_fsync_root",
            side_effect=MemoryKeyProviderError("memory_key_durability_unavailable"),
        ):
            self.assert_code("memory_key_commit_uncertain", provider.bootstrap)
        anchor = MemoryInstallationAnchorV1.decode(
            provider._anchor_file.path.read_bytes()
        )
        self.assertEqual(client.items, {})
        self.assertEqual(provider.bootstrap()[0], 1)
        self.assertEqual(provider.install_id, anchor.install_id)

    def test_partial_anchor_write_is_removed_and_bootstrap_can_retry(self) -> None:
        client = FakeKeychainClient()
        provider = MacOSMemoryKeyProvider(
            self.state_root,
            client=client,
            lock=InMemoryMemoryProviderLock(),
        )
        real_write = os.write
        calls = 0

        def partial_then_fail(descriptor: int, payload: object) -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                return real_write(descriptor, bytes(payload)[:8])
            raise OSError("injected short anchor write")

        with patch(
            "jarvis_orchestrator.memory_key_provider.os.write",
            side_effect=partial_then_fail,
        ):
            self.assert_code("memory_key_commit_uncertain", provider.bootstrap)
        self.assertFalse(provider._anchor_file.path.exists())
        self.assertEqual(client.items, {})
        self.assertEqual(provider.bootstrap()[0], 1)

    def test_anchor_lifecycle_durability_error_is_journaled_and_burned(
        self,
    ) -> None:
        client = FakeKeychainClient()
        provider = MacOSMemoryKeyProvider(
            self.state_root,
            client=client,
            lock=InMemoryMemoryProviderLock(),
        )
        provider.bootstrap()

        def commit_then_report_uncertain(*args: object, **kwargs: object) -> None:
            _atomic_move(*args, **kwargs)  # type: ignore[arg-type]
            raise MemoryKeyProviderError("memory_key_durability_unavailable")

        with patch(
            "jarvis_orchestrator.memory_key_provider._atomic_move",
            side_effect=commit_then_report_uncertain,
        ):
            self.assert_code("memory_key_commit_uncertain", provider.create_next)
        pending_account = f"install-{provider.install_id}/pending-create-v1"
        self.assertIn(pending_account, client.items)
        anchor = MemoryInstallationAnchorV1.decode(
            provider._anchor_file.path.read_bytes()
        )
        self.assertEqual(anchor.high_water, 2)
        self.assertTrue(anchor.initialized)

        version3, _ = provider.create_next()
        self.assertEqual(version3, 3)
        self.assertNotIn(pending_account, client.items)
        self.assertNotIn(f"install-{provider.install_id}/epoch-v2", client.items)

    def test_mac_locked_keychain_and_malformed_items_fail_content_free(self) -> None:
        client = FakeKeychainClient()
        provider = MacOSMemoryKeyProvider(
            self.state_root,
            client=client,
            lock=InMemoryMemoryProviderLock(),
        )
        provider.bootstrap()
        prefix = f"install-{provider.install_id}/"
        client.locked = True
        self.assert_code("memory_key_store_unavailable", provider.load, 1)
        client.locked = False
        client.items[prefix + "epoch-v1"] = b"wrong"
        self.assert_code("memory_key_invalid", provider.load, 1)
        client.items[prefix + "epoch-v1"] = b"x" * 32
        client.items[prefix + "active-state"] += b"\n"
        self.assert_code("memory_key_state_invalid", provider.active_version)

    def test_mac_keychain_namespace_uses_random_per_root_anchor(self) -> None:
        client = FakeKeychainClient()
        other_root = Path(self.temporary.name) / "other-state"
        other_root.mkdir(mode=0o700)
        os.chmod(other_root, 0o700)
        first = MacOSMemoryKeyProvider(
            self.state_root,
            client=client,
        )
        second = MacOSMemoryKeyProvider(
            other_root,
            client=client,
        )
        same_root = MacOSMemoryKeyProvider(self.state_root, client=client)
        self.assertEqual(first._lock.path, same_root._lock.path)
        self.assertNotEqual(first._lock.path, second._lock.path)
        self.assertEqual(first.bootstrap()[0], 1)
        self.assertEqual(second.bootstrap()[0], 1)
        self.assertNotEqual(first.install_id, second.install_id)
        self.assertEqual(same_root.install_id, first.install_id)
        first_prefix = f"install-{first.install_id}/"
        second_prefix = f"install-{second.install_id}/"
        self.assertTrue(any(item.startswith(first_prefix) for item in client.items))
        self.assertTrue(any(item.startswith(second_prefix) for item in client.items))
        first_accounts = {
            item for item in client.items if item.startswith(first_prefix)
        }
        second_accounts = {
            item for item in client.items if item.startswith(second_prefix)
        }
        self.assertTrue(first_accounts.isdisjoint(second_accounts))

    def test_physical_root_alias_shares_anchor_lock_and_keychain_namespace(
        self,
    ) -> None:
        client = FakeKeychainClient()
        alias_parent = Path(self.temporary.name) / "STATE-CASE-ALIAS"
        alias_parent.symlink_to(Path(self.temporary.name), target_is_directory=True)
        alias_root = alias_parent / self.state_root.name
        first = MacOSMemoryKeyProvider(self.state_root, client=client)
        alias = MacOSMemoryKeyProvider(alias_root, client=client)
        self.assertEqual(first._lock.path, alias._lock.path)
        self.assertEqual(first.bootstrap()[0], 1)
        self.assertEqual(alias.active_version(), 1)
        self.assertEqual(first.install_id, alias.install_id)
        self.assertEqual(first._anchor_file.path, alias._anchor_file.path)
        self.assertEqual(
            first._anchor_file.path.stat().st_ino, alias._anchor_file.path.stat().st_ino
        )

    def test_all_metadata_loss_creates_new_namespace_only_on_explicit_bootstrap(
        self,
    ) -> None:
        client = FakeKeychainClient()
        first = MacOSMemoryKeyProvider(self.state_root, client=client)
        first.bootstrap()
        first.create_next()
        old_install_id = first.install_id
        old_prefix = f"install-{old_install_id}/"
        old_orphan = old_prefix + "epoch-v2"
        old_key = client.items[old_orphan]
        for account in tuple(client.items):
            if account.startswith(old_prefix) and account != old_orphan:
                del client.items[account]
        first._anchor_file.path.unlink()

        replacement = MacOSMemoryKeyProvider(self.state_root, client=client)
        self.assert_code(
            "memory_key_not_initialized", getattr, replacement, "install_id"
        )
        self.assertEqual(replacement.bootstrap()[0], 1)
        self.assertNotEqual(replacement.install_id, old_install_id)
        self.assertEqual(client.items[old_orphan], old_key)
        new_prefix = f"install-{replacement.install_id}/"
        self.assertIn(new_prefix + "epoch-v1", client.items)
        self.assertNotIn(new_prefix + "epoch-v2", client.items)

    def test_retained_anchor_blocks_rebootstrap_after_backend_metadata_loss(
        self,
    ) -> None:
        client = FakeKeychainClient()
        first = MacOSMemoryKeyProvider(self.state_root, client=client)
        first.bootstrap()
        first.create_next()
        install_id = first.install_id
        prefix = f"install-{install_id}/"
        orphan_account = prefix + "epoch-v2"
        orphan_key = client.items[orphan_account]
        anchor = MemoryInstallationAnchorV1.decode(first._anchor_file.path.read_bytes())
        self.assertEqual(anchor.high_water, 2)
        self.assertTrue(anchor.initialized)
        for account in tuple(client.items):
            if account.startswith(prefix) and account != orphan_account:
                del client.items[account]
        before = dict(client.items)

        replacement = MacOSMemoryKeyProvider(self.state_root, client=client)
        self.assertEqual(replacement.install_id, install_id)
        self.assert_code("memory_key_state_invalid", replacement.bootstrap)
        self.assert_code("memory_key_state_invalid", replacement.active_version)
        self.assertEqual(client.items, before)
        self.assertEqual(client.items[orphan_account], orphan_key)

    def test_mac_lock_state_directory_must_remain_owner_private(self) -> None:
        os.chmod(self.state_root, 0o755)
        self.assert_code(
            "memory_key_private_state_invalid",
            MacOSMemoryKeyProvider,
            self.state_root,
            client=FakeKeychainClient(),
        )

    def test_windows_dpapi_files_are_private_bound_and_restartable(self) -> None:
        provider, protector, guard = self.windows()
        version1, key1 = provider.bootstrap()
        version2, key2 = provider.create_next()
        self.assertEqual((version1, version2), (1, 2))
        self.assertEqual(provider.load(1), key1)
        self.assertEqual(provider.load(2), key2)

        root = self.state_root / "keys" / MEMORY_KEY_NAMESPACE
        for path in root.iterdir():
            if os.name == "posix" and path.is_file():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        blob = (root / "epoch-v1.dpapi.json").read_bytes()
        self.assertNotIn(key1, blob)
        self.assertIn(b'"contract":"jarvis.memory-dpapi-key/1"', blob)
        self.assertEqual(
            protector.descriptions,
            [
                "JARVIS memory-records epoch v1",
                "JARVIS memory-records epoch v2",
            ],
        )

        restarted = WindowsMemoryKeyProvider(
            self.state_root,
            protector=protector,
            guard=guard,
        )
        self.assertEqual(restarted.active_version(), 2)
        self.assertEqual(restarted.load(1), key1)

    def test_windows_blob_schema_and_dpapi_entropy_prevent_row_swap(self) -> None:
        provider, _protector, _ = self.windows()
        provider.bootstrap()
        provider.create_next()
        root = self.state_root / "keys" / MEMORY_KEY_NAMESPACE
        path1 = root / "epoch-v1.dpapi.json"
        path2 = root / "epoch-v2.dpapi.json"
        original2 = path2.read_bytes()
        protected1 = _parse_dpapi_document(
            path1.read_bytes(),
            expected_version=1,
            install_id=provider.install_id,
        )
        path2.write_bytes(
            _dpapi_document(2, protected1, install_id=provider.install_id)
        )
        os.chmod(path2, 0o600)
        self.assert_code("memory_key_store_unavailable", provider.load, 2)
        path2.write_bytes(original2)
        os.chmod(path2, 0o600)

        document = json.loads(path1.read_text(encoding="utf-8"))
        document["extra"] = True
        path1.write_bytes(jcs_bytes(document))
        os.chmod(path1, 0o600)
        self.assert_code("memory_key_blob_invalid", provider.exists, 1)

    @unittest.skipUnless(os.name == "posix", "fake POSIX mode seam")
    def test_windows_private_metadata_and_symlink_fail_closed(self) -> None:
        provider, _, _ = self.windows()
        provider.bootstrap()
        root = self.state_root / "keys" / MEMORY_KEY_NAMESPACE
        state = root / "active-state.json"
        os.chmod(state, 0o644)
        self.assert_code("memory_key_private_state_invalid", provider.active_version)
        os.chmod(state, 0o600)

        provider.create_next()
        key1 = root / "epoch-v1.dpapi.json"
        target = root / "target.json"
        key1.replace(target)
        key1.symlink_to(target.name)
        self.assert_code("memory_key_private_state_invalid", provider.exists, 1)

    def test_windows_missing_locked_and_double_destroy_semantics(self) -> None:
        provider, protector, _ = self.windows()
        self.assert_code("memory_key_not_initialized", provider.exists, 1)
        provider.bootstrap()
        protector.locked = True
        self.assert_code("memory_key_store_unavailable", provider.load, 1)
        protector.locked = False
        provider.create_next()
        self.assert_code("memory_key_destroy_not_durable", provider.destroy, 1)
        self.assertTrue(provider.exists(1))
        self.assert_code("memory_key_active_destroy_forbidden", provider.destroy, 2)

    def test_windows_file_and_directory_fsyncs_are_attempted(self) -> None:
        protector = FakeDpapiProtector()
        guard = FakeWindowsPrivateGuard()
        with patch(
            "jarvis_orchestrator.memory_key_provider.os.fsync",
            wraps=os.fsync,
        ) as fsync:
            provider = WindowsMemoryKeyProvider(
                self.state_root,
                protector=protector,
                guard=guard,
            )
            provider.bootstrap()
        self.assertGreaterEqual(fsync.call_count, 6)

    def test_directory_fsync_failure_is_not_silently_ignored(self) -> None:
        provider, _, _ = self.windows()
        provider.bootstrap()
        with patch(
            "jarvis_orchestrator.memory_key_provider.os.fsync",
            side_effect=OSError("injected"),
        ):
            self.assert_code(
                "memory_key_durability_unavailable",
                provider._backend._fsync_root,
            )

    def test_windows_atomic_move_uses_write_through(self) -> None:
        calls: list[tuple[str, object]] = []

        class Function:
            argtypes = None
            restype = None

            def __init__(self, name: str, result: object) -> None:
                self.name = name
                self.result = result

            def __call__(self, *args: object) -> object:
                calls.append((self.name, args))
                return self.result

        kernel32 = SimpleNamespace(
            MoveFileExW=Function("move", 1),
        )
        with (
            patch("jarvis_orchestrator.memory_key_provider.os.name", "nt"),
            patch(
                "jarvis_orchestrator.memory_key_provider.ctypes.windll",
                SimpleNamespace(kernel32=kernel32),
                create=True,
            ),
        ):
            _atomic_move(
                self.state_root / "source",
                self.state_root / "destination",
                replace=True,
            )
        self.assertEqual([name for name, _ in calls], ["move"])
        move_args = calls[0][1]
        assert isinstance(move_args, tuple)
        self.assertEqual(move_args[0], str(self.state_root / "source"))
        self.assertEqual(move_args[1], str(self.state_root / "destination"))
        self.assertTrue(move_args[2] & 0x00000008)
        self.assertTrue(move_args[2] & 0x00000001)

    def test_dpapi_ffi_uses_windows_bool_abi(self) -> None:
        class Function:
            argtypes = None
            restype = None

        crypt32 = SimpleNamespace(
            CryptProtectData=Function(),
            CryptUnprotectData=Function(),
        )
        kernel32 = SimpleNamespace(LocalFree=Function())
        with (
            patch("jarvis_orchestrator.memory_key_provider.os.name", "nt"),
            patch(
                "jarvis_orchestrator.memory_key_provider.ctypes.windll",
                SimpleNamespace(crypt32=crypt32, kernel32=kernel32),
                create=True,
            ),
        ):
            WindowsDpapiProtector()
        self.assertIs(crypt32.CryptProtectData.restype, wintypes.BOOL)
        self.assertIs(crypt32.CryptUnprotectData.restype, wintypes.BOOL)

    def test_windows_pointer_replace_failure_burns_reserved_version(self) -> None:
        provider, _, _ = self.windows()
        self.assertEqual(provider.bootstrap()[0], 1)
        root = self.state_root / "keys" / MEMORY_KEY_NAMESPACE
        with patch(
            "jarvis_orchestrator.memory_key_provider.os.replace",
            side_effect=OSError("injected"),
        ):
            self.assert_code("memory_key_commit_uncertain", provider.create_next)
        self.assertEqual(provider.active_version(), 1)
        self.assertFalse((root / "epoch-v2.dpapi.json").exists())
        self.assertEqual(tuple(root.glob(".*.tmp")), ())
        self.assertEqual(provider.create_next()[0], 3)

    def test_factory_uses_only_injected_native_seams_on_non_native_host(self) -> None:
        client = FakeKeychainClient()
        mac = platform_memory_key_provider(
            state_root=self.state_root,
            platform="darwin",
            keychain_client=client,
            lock=InMemoryMemoryProviderLock(),
        )
        self.assert_code("memory_key_not_initialized", mac.active_version)
        self.assertEqual(mac.bootstrap()[0], 1)

        other_root = Path(self.temporary.name) / "windows-state"
        other_root.mkdir(mode=0o700)
        windows = platform_memory_key_provider(
            state_root=other_root,
            platform="win32",
            dpapi_protector=FakeDpapiProtector(),
            windows_guard=FakeWindowsPrivateGuard(),
            lock=InMemoryMemoryProviderLock(),
        )
        self.assert_code("memory_key_not_initialized", windows.active_version)
        self.assertEqual(windows.bootstrap()[0], 1)
        self.assert_code(
            "memory_key_platform_unsupported",
            platform_memory_key_provider,
            state_root=self.state_root,
            platform="linux",
        )
        self.assert_code(
            "memory_key_provider_arguments_invalid",
            platform_memory_key_provider,
            state_root=self.state_root,
            platform="darwin",
            keychain_client=client,
            dpapi_protector=FakeDpapiProtector(),
        )

    def test_native_windows_guard_creates_and_hardens_dedicated_state_root(
        self,
    ) -> None:
        parent = Path(self.temporary.name) / "operations"
        parent.mkdir(mode=0o700)
        state_root = parent / "memory-keys"
        key_root = state_root / "keys" / MEMORY_KEY_NAMESPACE
        hardened: list[Path] = []
        validated: list[Path] = []

        def harden(path: str | Path) -> None:
            hardened.append(Path(path))

        def validate(path: str | Path) -> None:
            selected = Path(path)
            self.assertTrue(selected.is_dir())
            validated.append(selected)

        with (
            patch(
                "jarvis_orchestrator.windows_private_file.harden_windows_private_path",
                side_effect=harden,
            ),
            patch(
                "jarvis_orchestrator.windows_private_file.validate_windows_private_path",
                side_effect=validate,
            ),
        ):
            NativeWindowsPrivateGuard().prepare_key_directory(
                state_root,
                key_root,
            )

        self.assertEqual(
            hardened,
            [state_root, state_root / "keys", key_root],
        )
        self.assertEqual(validated, hardened)


if __name__ == "__main__":
    unittest.main()
