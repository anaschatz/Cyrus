"""Current-user Windows DACL checks for private runtime files.

The pywin32 imports are deliberately lazy so non-Windows source runs do not
pretend to validate a Windows ACL and do not require native modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class WindowsPrivateFileError(RuntimeError):
    """A stable, path-free Windows private-file boundary failure."""

    def __init__(self) -> None:
        super().__init__("windows_private_file_invalid")


def _windows_security_context() -> tuple[Any, Any, Any, Any]:
    if os.name != "nt":
        raise WindowsPrivateFileError
    try:
        import win32api  # type: ignore[import-not-found]
        import win32con  # type: ignore[import-not-found]
        import win32security  # type: ignore[import-not-found]

        token = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(), win32con.TOKEN_QUERY
        )
        user_sid = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        system_sid = win32security.CreateWellKnownSid(
            win32security.WinLocalSystemSid, None
        )
        administrators_sid = win32security.CreateWellKnownSid(
            win32security.WinBuiltinAdministratorsSid, None
        )
        return win32security, user_sid, system_sid, administrators_sid
    except Exception:  # noqa: BLE001 - pywin32 errors vary by API and version
        raise WindowsPrivateFileError from None


def _validate_security_descriptor(
    win32security: Any,
    security: Any,
    user_sid: Any,
    system_sid: Any,
    administrators_sid: Any,
) -> None:
    try:
        owner = security.GetSecurityDescriptorOwner()
        if not win32security.EqualSid(owner, user_sid):
            raise WindowsPrivateFileError
        dacl = security.GetSecurityDescriptorDacl()
        if dacl is None:
            raise WindowsPrivateFileError
        allowed_types = {
            win32security.ACCESS_ALLOWED_ACE_TYPE,
            getattr(win32security, "ACCESS_ALLOWED_OBJECT_ACE_TYPE", 5),
            getattr(win32security, "ACCESS_ALLOWED_CALLBACK_ACE_TYPE", 9),
            getattr(
                win32security,
                "ACCESS_ALLOWED_CALLBACK_OBJECT_ACE_TYPE",
                11,
            ),
        }
        compound_allow_type = getattr(
            win32security,
            "ACCESS_ALLOWED_COMPOUND_ACE_TYPE",
            4,
        )
        allowed_sids = (user_sid, system_sid, administrators_sid)
        user_has_access = False
        for index in range(dacl.GetAceCount()):
            ace = dacl.GetAce(index)
            ace_type = ace[0][0]
            if ace_type == compound_allow_type:
                # Compound allow ACEs have multiple principals and are not
                # needed by the private-file policy, so reject them outright.
                raise WindowsPrivateFileError
            if ace_type not in allowed_types:
                continue
            sid = ace[-1]
            if not any(
                win32security.EqualSid(sid, allowed) for allowed in allowed_sids
            ):
                raise WindowsPrivateFileError
            if win32security.EqualSid(sid, user_sid):
                user_has_access = True
        if not user_has_access:
            raise WindowsPrivateFileError
    except WindowsPrivateFileError:
        raise
    except Exception:  # noqa: BLE001 - normalize pywin32 descriptor failures
        raise WindowsPrivateFileError from None


def validate_windows_private_descriptor(descriptor: int) -> None:
    """Validate the owner and DACL attached to the exact opened file handle."""

    if os.name != "nt":
        raise WindowsPrivateFileError
    try:
        import msvcrt  # type: ignore[import-not-found]

        win32security, user_sid, system_sid, administrators_sid = (
            _windows_security_context()
        )
        # Pass the borrowed CRT handle value directly. Wrapping it in a
        # pywintypes.HANDLE would transfer ownership and could close the file
        # descriptor before the caller completes its exact-size read.
        handle = msvcrt.get_osfhandle(descriptor)
        security = win32security.GetSecurityInfo(
            handle,
            win32security.SE_FILE_OBJECT,
            win32security.OWNER_SECURITY_INFORMATION
            | win32security.DACL_SECURITY_INFORMATION,
        )
        _validate_security_descriptor(
            win32security,
            security,
            user_sid,
            system_sid,
            administrators_sid,
        )
    except WindowsPrivateFileError:
        raise
    except Exception:  # noqa: BLE001 - normalize pywin32 handle failures
        raise WindowsPrivateFileError from None


def _validate_windows_private_path(path: Path) -> None:
    if os.name != "nt":
        raise WindowsPrivateFileError
    try:
        import win32con  # type: ignore[import-not-found]
        import win32file  # type: ignore[import-not-found]

        win32security, user_sid, system_sid, administrators_sid = (
            _windows_security_context()
        )
        attributes = win32file.GetFileAttributes(str(path))
        if attributes & win32con.FILE_ATTRIBUTE_REPARSE_POINT:
            raise WindowsPrivateFileError
        security = win32security.GetNamedSecurityInfo(
            str(path),
            win32security.SE_FILE_OBJECT,
            win32security.OWNER_SECURITY_INFORMATION
            | win32security.DACL_SECURITY_INFORMATION,
        )
        _validate_security_descriptor(
            win32security,
            security,
            user_sid,
            system_sid,
            administrators_sid,
        )
    except WindowsPrivateFileError:
        raise
    except Exception:  # noqa: BLE001 - normalize pywin32 path failures
        raise WindowsPrivateFileError from None


def validate_windows_private_path(path: str | Path) -> None:
    """Validate a path before creating a child inside a private directory."""

    _validate_windows_private_path(Path(path))


def harden_windows_private_path(path: str | Path) -> None:
    """Replace one Windows path DACL with the current-user private policy."""

    if os.name != "nt":
        raise WindowsPrivateFileError
    try:
        import ntsecuritycon  # type: ignore[import-not-found]

        win32security, user_sid, system_sid, administrators_sid = (
            _windows_security_context()
        )
        dacl = win32security.ACL()
        for sid in (user_sid, system_sid, administrators_sid):
            dacl.AddAccessAllowedAce(
                win32security.ACL_REVISION,
                ntsecuritycon.FILE_ALL_ACCESS,
                sid,
            )
        win32security.SetNamedSecurityInfo(
            str(Path(path)),
            win32security.SE_FILE_OBJECT,
            win32security.OWNER_SECURITY_INFORMATION
            | win32security.DACL_SECURITY_INFORMATION
            | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
            user_sid,
            None,
            dacl,
            None,
        )
        _validate_windows_private_path(Path(path))
    except WindowsPrivateFileError:
        raise
    except Exception:  # noqa: BLE001 - normalize pywin32 hardening failures
        raise WindowsPrivateFileError from None


__all__ = [
    "WindowsPrivateFileError",
    "harden_windows_private_path",
    "validate_windows_private_descriptor",
    "validate_windows_private_path",
]
