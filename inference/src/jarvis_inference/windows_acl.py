from __future__ import annotations

import ctypes
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ACCESS_ALLOWED_ACE_TYPE = 0
_OBJECT_INHERIT_ACE = 0x01
_CONTAINER_INHERIT_ACE = 0x02
_PRIVATE_ACE_FLAGS = _OBJECT_INHERIT_ACE | _CONTAINER_INHERIT_ACE
_FILE_ALL_ACCESS = 0x001F01FF
_SE_DACL_PROTECTED = 0x1000
_SYSTEM_SID = "S-1-5-18"
_BUILTIN_ADMINISTRATORS_SID = "S-1-5-32-544"
_SID_PATTERN = re.compile(r"S-\d+(?:-\d+)+", re.IGNORECASE)


class WindowsAclFailure(RuntimeError):
    """A Windows private-directory ACL operation failed closed."""


@dataclass(frozen=True)
class WindowsAce:
    sid: str
    mask: int
    ace_type: int
    flags: int


@dataclass(frozen=True)
class WindowsAclSnapshot:
    owner_sid: str
    dacl_present: bool
    dacl_defaulted: bool
    protected: bool
    entries: tuple[WindowsAce, ...]


def _normalise_sid(value: str) -> str:
    if not isinstance(value, str) or _SID_PATTERN.fullmatch(value) is None:
        raise WindowsAclFailure("private directory identity is invalid")
    return value.upper()


def private_acl_principals(current_user_sid: str) -> tuple[str, ...]:
    principals = (
        _normalise_sid(current_user_sid),
        _SYSTEM_SID,
        _BUILTIN_ADMINISTRATORS_SID,
    )
    return tuple(dict.fromkeys(principals))


def validate_acl_snapshot(
    snapshot: WindowsAclSnapshot,
    *,
    current_user_sid: str,
) -> None:
    expected = set(private_acl_principals(current_user_sid))
    owner = _normalise_sid(snapshot.owner_sid)
    if owner != _normalise_sid(current_user_sid):
        raise WindowsAclFailure("private directory owner is not trusted")
    if (
        not snapshot.dacl_present
        or snapshot.dacl_defaulted
        or not snapshot.protected
    ):
        raise WindowsAclFailure("private directory ACL is not protected")
    if len(snapshot.entries) != len(expected):
        raise WindowsAclFailure("private directory ACL is permissive")

    observed: set[str] = set()
    for entry in snapshot.entries:
        if (
            entry.ace_type != _ACCESS_ALLOWED_ACE_TYPE
            or entry.mask != _FILE_ALL_ACCESS
            or entry.flags != _PRIVATE_ACE_FLAGS
        ):
            raise WindowsAclFailure("private directory ACL is permissive")
        sid = _normalise_sid(entry.sid)
        if sid not in expected or sid in observed:
            raise WindowsAclFailure("private directory ACL is permissive")
        observed.add(sid)
    if observed != expected:
        raise WindowsAclFailure("private directory ACL is permissive")


class WindowsAclApi:
    """Small ctypes boundary for atomic protected directory creation."""

    def __init__(self) -> None:  # pragma: no cover - native Windows
        if os.name != "nt":
            raise WindowsAclFailure("Windows ACLs require Windows")
        try:
            from ctypes import wintypes

            self.wintypes = wintypes
            self.advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
            self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._configure_signatures()
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise WindowsAclFailure("Windows ACL support is unavailable") from exc

    def _configure_signatures(self) -> None:  # pragma: no cover - native Windows
        wintypes = self.wintypes
        advapi32 = self.advapi32
        kernel32 = self.kernel32

        advapi32.OpenProcessToken.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        advapi32.GetTokenInformation.restype = wintypes.BOOL
        advapi32.ConvertSidToStringSidW.argtypes = [
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.LPWSTR),
        ]
        advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
        advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.DWORD),
        ]
        advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = (
            wintypes.BOOL
        )
        advapi32.GetSecurityDescriptorDacl.argtypes = [
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.BOOL),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.BOOL),
        ]
        advapi32.GetSecurityDescriptorDacl.restype = wintypes.BOOL
        advapi32.SetNamedSecurityInfoW.argtypes = [
            wintypes.LPWSTR,
            ctypes.c_int,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.LPVOID,
        ]
        advapi32.SetNamedSecurityInfoW.restype = wintypes.DWORD
        advapi32.GetNamedSecurityInfoW.argtypes = [
            wintypes.LPWSTR,
            ctypes.c_int,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
        ]
        advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
        advapi32.GetSecurityDescriptorControl.argtypes = [
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.WORD),
            ctypes.POINTER(wintypes.DWORD),
        ]
        advapi32.GetSecurityDescriptorControl.restype = wintypes.BOOL
        advapi32.GetAclInformation.argtypes = [
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.c_int,
        ]
        advapi32.GetAclInformation.restype = wintypes.BOOL
        advapi32.GetAce.argtypes = [
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
        ]
        advapi32.GetAce.restype = wintypes.BOOL
        advapi32.IsValidSid.argtypes = [wintypes.LPVOID]
        advapi32.IsValidSid.restype = wintypes.BOOL
        advapi32.GetLengthSid.argtypes = [wintypes.LPVOID]
        advapi32.GetLengthSid.restype = wintypes.DWORD

        class _SecurityAttributes(ctypes.Structure):
            _fields_ = (
                ("length", wintypes.DWORD),
                ("security_descriptor", wintypes.LPVOID),
                ("inherit_handle", wintypes.BOOL),
            )

        self.security_attributes_type = _SecurityAttributes
        kernel32.CreateDirectoryW.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(_SecurityAttributes),
        ]
        kernel32.CreateDirectoryW.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.LocalFree.argtypes = [wintypes.LPVOID]
        kernel32.LocalFree.restype = wintypes.LPVOID

    def _sid_string(self, sid: Any) -> str:  # pragma: no cover - native Windows
        wintypes = self.wintypes
        if not sid or not self.advapi32.IsValidSid(sid):
            raise WindowsAclFailure("ACL contains an invalid identity")
        text = wintypes.LPWSTR()
        if not self.advapi32.ConvertSidToStringSidW(sid, ctypes.byref(text)):
            raise WindowsAclFailure("ACL identity could not be verified")
        try:
            return _normalise_sid(text.value)
        finally:
            self.kernel32.LocalFree(ctypes.cast(text, wintypes.LPVOID))

    def current_user_sid(self) -> str:  # pragma: no cover - native Windows
        wintypes = self.wintypes

        class _SidAndAttributes(ctypes.Structure):
            _fields_ = (
                ("sid", wintypes.LPVOID),
                ("attributes", wintypes.DWORD),
            )

        class _TokenUser(ctypes.Structure):
            _fields_ = (("user", _SidAndAttributes),)

        token = wintypes.HANDLE()
        if not self.advapi32.OpenProcessToken(
            self.kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)
        ):
            raise WindowsAclFailure("current owner could not be derived")
        try:
            required = wintypes.DWORD()
            self.advapi32.GetTokenInformation(
                token, 1, None, 0, ctypes.byref(required)
            )
            if required.value <= 0:
                raise WindowsAclFailure("current owner could not be derived")
            buffer = ctypes.create_string_buffer(required.value)
            if not self.advapi32.GetTokenInformation(
                token,
                1,
                buffer,
                required.value,
                ctypes.byref(required),
            ):
                raise WindowsAclFailure("current owner could not be derived")
            return self._sid_string(_TokenUser.from_buffer(buffer).user.sid)
        finally:
            self.kernel32.CloseHandle(token)

    def _descriptor(self, sddl: str) -> Any:  # pragma: no cover - native Windows
        wintypes = self.wintypes
        descriptor = wintypes.LPVOID()
        descriptor_size = wintypes.DWORD()
        if not self.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl,
            1,
            ctypes.byref(descriptor),
            ctypes.byref(descriptor_size),
        ):
            raise WindowsAclFailure("private ACL could not be constructed")
        return descriptor

    def create_private_directory(
        self,
        path: Path,
        current_user_sid: str,
        principals: tuple[str, ...],
    ) -> None:  # pragma: no cover - native Windows
        entries = "".join(f"(A;OICI;FA;;;{sid})" for sid in principals)
        descriptor = self._descriptor(
            f"O:{_normalise_sid(current_user_sid)}D:P{entries}"
        )
        try:
            attributes = self.security_attributes_type(
                ctypes.sizeof(self.security_attributes_type),
                descriptor,
                False,
            )
            if not self.kernel32.CreateDirectoryW(
                str(path), ctypes.byref(attributes)
            ):
                if ctypes.get_last_error() == 183:
                    raise FileExistsError
                raise WindowsAclFailure("private directory could not be created")
        finally:
            self.kernel32.LocalFree(descriptor)

    def _set_dacl_from_sddl(
        self,
        path: Path,
        sddl: str,
        *,
        protected: bool,
    ) -> None:  # pragma: no cover - native Windows test support
        wintypes = self.wintypes
        descriptor = self._descriptor(sddl)
        try:
            present = wintypes.BOOL()
            defaulted = wintypes.BOOL()
            dacl = wintypes.LPVOID()
            if not self.advapi32.GetSecurityDescriptorDacl(
                descriptor,
                ctypes.byref(present),
                ctypes.byref(dacl),
                ctypes.byref(defaulted),
            ) or not present.value or not dacl:
                raise WindowsAclFailure("private ACL could not be constructed")
            security_information = 0x00000004
            security_information |= 0x80000000 if protected else 0x20000000
            if (
                self.advapi32.SetNamedSecurityInfoW(
                    str(path),
                    1,
                    security_information,
                    None,
                    None,
                    dacl,
                    None,
                )
                != 0
            ):
                raise WindowsAclFailure("private ACL could not be applied")
        finally:
            self.kernel32.LocalFree(descriptor)

    def read_private_dacl(
        self, path: Path
    ) -> WindowsAclSnapshot:  # pragma: no cover - native Windows
        wintypes = self.wintypes

        class _AclSizeInformation(ctypes.Structure):
            _fields_ = (
                ("ace_count", wintypes.DWORD),
                ("bytes_in_use", wintypes.DWORD),
                ("bytes_free", wintypes.DWORD),
            )

        class _AceHeader(ctypes.Structure):
            _fields_ = (
                ("ace_type", wintypes.BYTE),
                ("ace_flags", wintypes.BYTE),
                ("ace_size", wintypes.WORD),
            )

        owner = wintypes.LPVOID()
        dacl = wintypes.LPVOID()
        descriptor = wintypes.LPVOID()
        result = self.advapi32.GetNamedSecurityInfoW(
            str(path),
            1,
            0x00000001 | 0x00000004,
            ctypes.byref(owner),
            None,
            ctypes.byref(dacl),
            None,
            ctypes.byref(descriptor),
        )
        if result != 0 or not descriptor:
            raise WindowsAclFailure("private ACL could not be read")
        try:
            present = wintypes.BOOL()
            defaulted = wintypes.BOOL()
            descriptor_dacl = wintypes.LPVOID()
            if not self.advapi32.GetSecurityDescriptorDacl(
                descriptor,
                ctypes.byref(present),
                ctypes.byref(descriptor_dacl),
                ctypes.byref(defaulted),
            ):
                raise WindowsAclFailure("private ACL could not be read")
            control = wintypes.WORD()
            revision = wintypes.DWORD()
            if not self.advapi32.GetSecurityDescriptorControl(
                descriptor,
                ctypes.byref(control),
                ctypes.byref(revision),
            ):
                raise WindowsAclFailure("private ACL could not be read")
            if not present.value or not descriptor_dacl:
                return WindowsAclSnapshot(
                    owner_sid=self._sid_string(owner),
                    dacl_present=bool(present.value),
                    dacl_defaulted=bool(defaulted.value),
                    protected=bool(control.value & _SE_DACL_PROTECTED),
                    entries=(),
                )

            information = _AclSizeInformation()
            if not self.advapi32.GetAclInformation(
                descriptor_dacl,
                ctypes.byref(information),
                ctypes.sizeof(information),
                2,
            ):
                raise WindowsAclFailure("private ACL could not be read")
            entries: list[WindowsAce] = []
            for index in range(information.ace_count):
                ace = wintypes.LPVOID()
                if not self.advapi32.GetAce(
                    descriptor_dacl, index, ctypes.byref(ace)
                ) or not ace:
                    raise WindowsAclFailure("private ACL could not be read")
                address = ctypes.cast(ace, ctypes.c_void_p).value
                if address is None:
                    raise WindowsAclFailure("private ACL could not be read")
                header = _AceHeader.from_address(address)
                if header.ace_type != _ACCESS_ALLOWED_ACE_TYPE:
                    entries.append(
                        WindowsAce(
                            sid="",
                            mask=0,
                            ace_type=int(header.ace_type),
                            flags=int(header.ace_flags),
                        )
                    )
                    continue
                if header.ace_size < 16:
                    raise WindowsAclFailure("private ACL could not be read")
                mask = ctypes.c_uint32.from_address(address + 4).value
                sid = wintypes.LPVOID(address + 8)
                if not self.advapi32.IsValidSid(sid):
                    raise WindowsAclFailure("private ACL could not be read")
                sid_length = int(self.advapi32.GetLengthSid(sid))
                if sid_length <= 0 or 8 + sid_length > header.ace_size:
                    raise WindowsAclFailure("private ACL could not be read")
                entries.append(
                    WindowsAce(
                        sid=self._sid_string(sid),
                        mask=int(mask),
                        ace_type=int(header.ace_type),
                        flags=int(header.ace_flags),
                    )
                )
            return WindowsAclSnapshot(
                owner_sid=self._sid_string(owner),
                dacl_present=bool(present.value),
                dacl_defaulted=bool(defaulted.value),
                protected=bool(control.value & _SE_DACL_PROTECTED),
                entries=tuple(entries),
            )
        finally:
            self.kernel32.LocalFree(descriptor)


def validate_private_directory_acl(
    path: Path,
    *,
    api: Any | None = None,
) -> None:
    boundary = api or WindowsAclApi()
    try:
        current_user_sid = boundary.current_user_sid()
        snapshot = boundary.read_private_dacl(Path(path))
        validate_acl_snapshot(snapshot, current_user_sid=current_user_sid)
    except WindowsAclFailure:
        raise
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise WindowsAclFailure("private ACL could not be verified") from exc


def create_private_directory(
    parent: Path,
    *,
    prefix: str,
    api: Any | None = None,
) -> Path:
    if (
        not isinstance(prefix, str)
        or not prefix
        or "\x00" in prefix
        or Path(prefix).name != prefix
        or prefix in {".", ".."}
    ):
        raise WindowsAclFailure("private directory prefix is invalid")
    boundary = api or WindowsAclApi()
    try:
        current_user_sid = boundary.current_user_sid()
        principals = private_acl_principals(current_user_sid)
        for _ in range(100):
            candidate = Path(parent) / f"{prefix}{secrets.token_hex(16)}"
            try:
                boundary.create_private_directory(
                    candidate, current_user_sid, principals
                )
            except FileExistsError:
                continue
            snapshot = boundary.read_private_dacl(candidate)
            validate_acl_snapshot(snapshot, current_user_sid=current_user_sid)
            return candidate
    except WindowsAclFailure:
        raise
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise WindowsAclFailure("private directory could not be created") from exc
    raise WindowsAclFailure("private directory name allocation failed")
