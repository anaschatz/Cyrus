from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from test_memory_crypto import binding
from test_memory_key_provider import FakeWindowsPrivateGuard, state
from jarvis_orchestrator import memory_key_provider as provider
from jarvis_orchestrator.memory_crypto import (
    MemoryEnvelopeInvalid,
    MemoryEnvelopeCipherV1,
    InMemoryMemoryEpochKeyProvider,
    MemoryCryptoError,
)


class MemoryKeyBoundaryTests(unittest.TestCase):
    def test_cached_and_provider_token_limits_match(self):
        cipher = MemoryEnvelopeCipherV1(
            InMemoryMemoryEpochKeyProvider.initialized(b"k" * 32)
        )
        principal = binding().principal_id
        session = cipher.open_epoch_session(principal_id=principal, key_version=1)
        for target, extra in (
            (cipher, dict(principal_id=principal, key_version=1)),
            (session, {}),
        ):
            terms = tuple(f"term-{index:03d}" for index in range(129))
            self.assertEqual(
                len(
                    target.blind_tokens(
                        content="text",
                        normalized_content="text",
                        terms=terms[:128],
                        **extra,
                    ).term_hmacs
                ),
                128,
            )
            self.assertEqual(
                len(target.blind_term_tokens(terms=terms[:32], **extra)), 32
            )
            with self.assertRaises(MemoryCryptoError):
                target.blind_tokens(
                    content="text", normalized_content="text", terms=terms, **extra
                )
            with self.assertRaises(MemoryCryptoError):
                target.blind_term_tokens(terms=terms[:33], **extra)

    def test_binding_rejects_fractional_schema_version_at_construction(self):
        with self.assertRaises(MemoryEnvelopeInvalid):
            replace(binding(), record_schema_version=10.0)

    def test_pending_constructor_rejects_invalid_nested_state_types(self):
        valid = provider.MemoryKeyPendingCreateV1(
            install_id="0" * 64,
            phase="reserved",
            version=2,
            key_sha256="a" * 64,
            previous_state=state(1, 2),
            selected_state=state(2, 3),
        )
        for field, value in (
            ("selected_state", None),
            ("selected_state", {}),
            ("previous_state", False),
            ("previous_state", []),
        ):
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(provider.MemoryKeyProviderError),
            ):
                replace(valid, **{field: value})

    def test_provider_json_decoders_preserve_content_free_errors(self):
        for raw, decode in (
            (
                b"[" * 1000 + b"0" + b"]" * 1000,
                provider.MemoryKeyPendingCreateV1.decode,
            ),
            (
                b"[" * 1500 + b"0" + b"]" * 1500,
                lambda raw: provider._parse_dpapi_document(
                    raw, expected_version=1, install_id="0" * 64
                ),
            ),
            (
                b'{"x":' + b"9" * 5000 + b"}",
                lambda raw: provider._parse_dpapi_document(
                    raw, expected_version=1, install_id="0" * 64
                ),
            ),
        ):
            with (
                self.subTest(size=len(raw)),
                self.assertRaises(provider.MemoryKeyProviderError) as raised,
            ):
                decode(raw)
            self.assertIsNone(raised.exception.__cause__)
            self.assertTrue(raised.exception.__suppress_context__)

    def test_windows_lock_open_failure_keeps_original_typed_error(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = provider._InterprocessFileLock(
                Path(directory) / "lock", windows_guard=FakeWindowsPrivateGuard()
            )
            with patch.object(
                provider.os, "open", side_effect=OSError("synthetic open failure")
            ):
                with self.assertRaisesRegex(
                    provider.MemoryKeyProviderError, "memory_key_lock_failed"
                ):
                    lock._open_windows()

    @unittest.skipUnless(os.name == "posix", "POSIX exclusive publication")
    def test_exclusive_move_preserves_competing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = (
                Path(directory) / "source",
                Path(directory) / "destination",
            )
            source.write_bytes(b"new identity")
            destination.write_bytes(b"existing identity")
            with self.assertRaises(provider.MemoryKeyProviderError):
                provider._atomic_move(source, destination, replace=False)
            self.assertEqual(destination.read_bytes(), b"existing identity")
            self.assertEqual(source.read_bytes(), b"new identity")
            destination.unlink()
            provider._atomic_move(source, destination, replace=False)
            self.assertFalse(source.exists())
            self.assertEqual(destination.read_bytes(), b"new identity")
            self.assertEqual(destination.stat().st_nlink, 1)

    @unittest.skipUnless(os.name == "posix", "POSIX anchor publication race")
    def test_anchor_create_never_overwrites_a_competing_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            anchor_file = provider._InstallationAnchorFile(Path(directory))
            selected = provider.MemoryInstallationAnchorV1("a" * 64)
            competing = provider.MemoryInstallationAnchorV1("b" * 64)
            move = provider._atomic_move

            def race(source, destination, *, replace):
                destination.write_bytes(competing.encode())
                destination.chmod(0o600)
                move(source, destination, replace=replace)

            with patch.object(provider, "_atomic_move", side_effect=race):
                with self.assertRaises(provider.MemoryKeyProviderError):
                    anchor_file.create(selected)
            self.assertEqual(anchor_file.read(), competing)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    @unittest.skipUnless(os.name == "posix", "POSIX staged hard links")
    def test_readers_recover_only_known_staging_aliases_after_interruption(self):
        from test_memory_key_provider import FakeDpapiProtector

        for mode in ("anchor", "dpapi"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                if mode == "anchor":
                    reader = provider._InstallationAnchorFile(root)
                    path = reader.path
                    value = provider.MemoryInstallationAnchorV1("a" * 64)
                    raw, expected, read = value.encode(), value, reader.read
                    suffix = "tmp"
                else:
                    reader = provider._WindowsFileBackend(
                        root, FakeDpapiProtector(), FakeWindowsPrivateGuard(), "a" * 64
                    )
                    path = reader.state_path
                    raw = provider.MemoryKeyStateV1("a" * 64, 1, 2).encode()
                    expected, read, suffix = raw, reader.read_state, "stage"
                path.write_bytes(raw)
                path.chmod(0o600)
                staged = path.parent / f".{path.name}.{'f' * 32}.{suffix}"
                os.link(path, staged)
                unrelated = path.parent / f".{path.name}.{'e' * 32}.{suffix}"
                unrelated.write_bytes(b"unrelated staging file")
                linked = path.parent / f".{path.name}.{'d' * 32}.{suffix}"
                linked.symlink_to(path)
                self.assertEqual(read(), expected)
                self.assertFalse(staged.exists())
                self.assertEqual(unrelated.read_bytes(), b"unrelated staging file")
                self.assertTrue(linked.is_symlink())
                self.assertEqual(path.stat().st_nlink, 1)
                other = path.parent / "unrecognized-link"
                os.link(path, other)
                with self.assertRaises(provider.MemoryKeyProviderError):
                    read()
                self.assertTrue(other.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX FIFO")
    def test_anchor_and_dpapi_readers_reject_fifo_swaps_without_blocking(self):
        script = """import os, sys
from pathlib import Path
from unittest.mock import patch
from test_memory_key_provider import FakeWindowsPrivateGuard, FakeDpapiProtector
from jarvis_orchestrator import memory_key_provider as m
root=Path(sys.argv[1]); mode=sys.argv[2]
if mode=="anchor":
    reader=m._InstallationAnchorFile(root); path=reader.path
    path.write_bytes(m.MemoryInstallationAnchorV1("a"*64).encode())
    action=reader.read
else:
    reader=m._WindowsFileBackend(root, FakeDpapiProtector(), FakeWindowsPrivateGuard(), "a"*64)
    path=reader.state_path; path.write_bytes(b"fixture")
    action=reader.read_state
path.chmod(0o600); real_open=os.open
def swap(name, flags, *args, **kwargs):
    if Path(name)==path:
        path.unlink(); os.mkfifo(path, 0o600)
    return real_open(name, flags, *args, **kwargs)
with patch.object(os, "open", side_effect=swap):
    try: action()
    except m.MemoryKeyProviderError: sys.exit(0)
sys.exit(1)
"""
        for mode in ("anchor", "dpapi"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, "-c", script, directory, mode],
                    capture_output=True,
                    timeout=2,
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
