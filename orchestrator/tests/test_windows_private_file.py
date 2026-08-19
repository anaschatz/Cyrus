from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from jarvis_orchestrator.windows_private_file import (
    WindowsPrivateFileError,
    _validate_security_descriptor,
    harden_windows_private_path,
    validate_windows_private_descriptor,
)


class _Acl:
    def __init__(self, *entries: str | tuple[int, str]) -> None:
        self._aces = tuple(
            (
                (entry[0], 0),
                0xFFFF,
                entry[1],
            )
            if isinstance(entry, tuple)
            else ((0, 0), 0xFFFF, entry)
            for entry in entries
        )

    def GetAceCount(self) -> int:
        return len(self._aces)

    def GetAce(self, index: int):
        return self._aces[index]


class _Security:
    def __init__(self, owner: str, dacl: _Acl | None) -> None:
        self._owner = owner
        self._dacl = dacl

    def GetSecurityDescriptorOwner(self):
        return self._owner

    def GetSecurityDescriptorDacl(self):
        return self._dacl


class _Win32Security:
    ACCESS_ALLOWED_ACE_TYPE = 0

    @staticmethod
    def EqualSid(left: str, right: str) -> bool:
        return left == right


class WindowsPrivateFileTests(unittest.TestCase):
    def test_descriptor_policy_accepts_only_user_system_and_administrators(
        self,
    ) -> None:
        _validate_security_descriptor(
            _Win32Security,
            _Security("user", _Acl("user", "system", "administrators")),
            "user",
            "system",
            "administrators",
        )

        for security in (
            _Security("other", _Acl("user")),
            _Security("user", None),
            _Security("user", _Acl("system")),
            _Security("user", _Acl("user", "everyone")),
            _Security("user", _Acl("user", (9, "everyone"))),
        ):
            with (
                self.subTest(owner=security.GetSecurityDescriptorOwner()),
                self.assertRaises(WindowsPrivateFileError),
            ):
                _validate_security_descriptor(
                    _Win32Security,
                    security,
                    "user",
                    "system",
                    "administrators",
                )

    @unittest.skipUnless(os.name == "nt", "native Windows DACL contract")
    def test_native_hardened_file_descriptor_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harden_windows_private_path(root)
            path = root / "private.json"
            path.write_bytes(b"{}")
            harden_windows_private_path(path)
            descriptor = os.open(path, os.O_RDONLY)
            try:
                validate_windows_private_descriptor(descriptor)
                self.assertEqual(os.read(descriptor, 2), b"{}")
            finally:
                os.close(descriptor)


if __name__ == "__main__":
    unittest.main()
