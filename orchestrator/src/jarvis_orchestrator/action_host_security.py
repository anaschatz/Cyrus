"""Local Action Host filesystem/peer boundary for the v1 Unix-socket IPC."""

from __future__ import annotations

import os
import socket
import stat
from pathlib import Path

from .action_host_protocol import ActionHostProtocolError


V1_THREAT_MODEL = (
    "Protects against other local users and accidental replay; it does not "
    "protect a fully compromised process running as the same user."
)


def validate_action_host_paths(socket_path: Path, token_path: Path) -> None:
    """Fail closed on symlinks or group/world-readable IPC material."""

    socket_path = Path(socket_path)
    token_path = Path(token_path)
    parent = socket_path.parent
    parent_stat = parent.lstat()
    if stat.S_ISLNK(parent_stat.st_mode) or stat.S_IMODE(parent_stat.st_mode) != 0o700:
        raise ActionHostProtocolError("host.auth_failed")
    token_stat = token_path.lstat()
    if stat.S_ISLNK(token_stat.st_mode) or not stat.S_ISREG(token_stat.st_mode):
        raise ActionHostProtocolError("host.auth_failed")
    if stat.S_IMODE(token_stat.st_mode) != 0o600 or token_stat.st_uid != os.getuid():
        raise ActionHostProtocolError("host.auth_failed")
    if socket_path.exists() or socket_path.is_symlink():
        socket_stat = socket_path.lstat()
        if stat.S_ISLNK(socket_stat.st_mode) or not stat.S_ISSOCK(socket_stat.st_mode):
            raise ActionHostProtocolError("host.auth_failed")
        if socket_stat.st_uid != os.getuid():
            raise ActionHostProtocolError("host.auth_failed")


def load_instance_token(token_path: Path) -> bytes:
    descriptor = os.open(token_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        token = os.read(descriptor, 33)
    finally:
        os.close(descriptor)
    if len(token) != 32:
        raise ActionHostProtocolError("host.auth_failed")
    return token


def assert_peer_uid(connection: socket.socket, expected_uid: int | None = None) -> None:
    expected = os.getuid() if expected_uid is None else expected_uid
    getter = getattr(connection, "getpeereid", None)
    if getter is None:
        raise ActionHostProtocolError("host.auth_failed")
    peer_uid, _peer_gid = getter()
    if peer_uid != expected:
        raise ActionHostProtocolError("host.auth_failed")
