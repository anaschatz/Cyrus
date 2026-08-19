"""Strict pre-protocol mutual authentication for the Voice WebSocket.

``jarvis.voice-auth/1`` runs before the first ``jarvis.voice/1`` envelope.  It
authenticates both peers with a per-install key, commits the client to the
exact hello frame it will send, and lets the server authenticate the complete
challenge/response/hello transcript.  This module is deliberately transport
agnostic; gateway and runtime state machines live elsewhere.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


AUTH_PROTOCOL = "jarvis.voice-auth/1"
KEY_FILE_SCHEMA = "jarvis.voice-auth-key/1"

KEY_BYTES = 32
NONCE_BYTES = 32
MAC_BYTES = 32
CHALLENGE_LIFETIME_MS = 5_000
MAX_AUTH_FRAME_BYTES = 4 * 1024
MAX_HELLO_FRAME_BYTES = 64 * 1024
MAX_KEY_FILE_BYTES = 1_024
MAX_KEY_VERSION = 2**31 - 1

_B64URL_32 = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_CHALLENGE_TYPE = "server.challenge"
_RESPONSE_TYPE = "client.response"
_READY_TYPE = "server.ready"

_CHALLENGE_DOMAIN = b"jarvis.voice-auth/1\x00server.challenge\x00"
_RESPONSE_DOMAIN = b"jarvis.voice-auth/1\x00client.response\x00"
_READY_DOMAIN = b"jarvis.voice-auth/1\x00server.ready\x00"


class VoiceAuthError(RuntimeError):
    """A content-free authentication or key-store failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _DuplicateKey(ValueError):
    pass


def _now_ms(value: int | None) -> int:
    selected = time.time_ns() // 1_000_000 if value is None else value
    if not isinstance(selected, int) or isinstance(selected, bool) or selected < 0:
        raise VoiceAuthError("voice_auth.time_invalid")
    return selected


def _key_version(value: Any, *, code: str = "voice_auth.binding_invalid") -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= MAX_KEY_VERSION
    ):
        raise VoiceAuthError(code)
    return value


def _uuid4(value: Any) -> str:
    if not isinstance(value, str):
        raise VoiceAuthError("voice_auth.binding_invalid")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError):
        raise VoiceAuthError("voice_auth.binding_invalid") from None
    if parsed.version != 4 or str(parsed) != value:
        raise VoiceAuthError("voice_auth.binding_invalid")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise _DuplicateKey
        result[key] = value
    return result


def _raw_bytes(
    raw: str | bytes,
    *,
    maximum: int,
    empty_code: str = "voice_auth.frame_invalid",
    size_code: str = "voice_auth.frame_too_large",
) -> bytes:
    if isinstance(raw, str):
        try:
            encoded = raw.encode("utf-8", errors="strict")
        except UnicodeError:
            raise VoiceAuthError(empty_code) from None
    elif isinstance(raw, bytes):
        encoded = raw
    else:
        raise VoiceAuthError(empty_code)
    if not encoded:
        raise VoiceAuthError(empty_code)
    if len(encoded) > maximum:
        raise VoiceAuthError(size_code)
    return encoded


def _json_object(raw: str | bytes, *, maximum: int = MAX_AUTH_FRAME_BYTES) -> tuple[dict[str, Any], bytes]:
    encoded = _raw_bytes(raw, maximum=maximum)
    try:
        text = encoded.decode("utf-8", errors="strict")
        if text.startswith("\ufeff"):
            raise ValueError
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, _DuplicateKey):
        raise VoiceAuthError("voice_auth.frame_invalid") from None
    if not isinstance(value, dict):
        raise VoiceAuthError("voice_auth.frame_invalid")
    return value, encoded


def _exact(value: Mapping[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise VoiceAuthError("voice_auth.frame_invalid")


def _protocol_and_type(value: Mapping[str, Any], expected_type: str) -> None:
    if value.get("protocol") != AUTH_PROTOCOL:
        raise VoiceAuthError("voice_auth.protocol_mismatch")
    if value.get("type") != expected_type:
        raise VoiceAuthError("voice_auth.type_invalid")


def _encode_32(value: bytes, *, code: str = "voice_auth.base64_invalid") -> str:
    if not isinstance(value, bytes) or len(value) != 32:
        raise VoiceAuthError(code)
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_32(value: Any, *, code: str = "voice_auth.base64_invalid") -> bytes:
    if not isinstance(value, str) or _B64URL_32.fullmatch(value) is None:
        raise VoiceAuthError(code)
    try:
        decoded = base64.b64decode(value + "=", altchars=b"-_", validate=True)
    except (ValueError, TypeError):
        raise VoiceAuthError(code) from None
    if len(decoded) != 32 or _encode_32(decoded, code=code) != value:
        raise VoiceAuthError(code)
    return decoded


def _key_bytes(value: bytes) -> bytes:
    if not isinstance(value, bytes) or len(value) != KEY_BYTES:
        raise VoiceAuthError("voice_auth.key_format_invalid")
    return value


def _mac(key: bytes, domain: bytes, message: bytes) -> bytes:
    return hmac.new(_key_bytes(key), domain + message, hashlib.sha256).digest()


def _framed(value: bytes) -> bytes:
    if len(value) > 2**32 - 1:
        raise VoiceAuthError("voice_auth.frame_too_large")
    return len(value).to_bytes(4, "big") + value


def _auth_frame(value: Any) -> bytes:
    method = getattr(value, "to_bytes", None)
    if method is None or not callable(method):
        raise VoiceAuthError("voice_auth.frame_invalid")
    encoded = method()
    return _raw_bytes(encoded, maximum=MAX_AUTH_FRAME_BYTES)


def _hello_frame(value: str | bytes) -> bytes:
    return _raw_bytes(value, maximum=MAX_HELLO_FRAME_BYTES)


@dataclass(frozen=True, slots=True)
class VoiceAuthKey:
    key_version: int
    key: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _key_version(self.key_version, code="voice_auth.key_format_invalid")
        _key_bytes(self.key)


@dataclass(frozen=True, slots=True)
class ServerChallenge:
    connection_id: str
    key_version: int
    issued_at_ms: int
    expires_at_ms: int
    server_nonce: bytes
    server_mac: bytes

    def __post_init__(self) -> None:
        _uuid4(self.connection_id)
        _key_version(self.key_version)
        if (
            not isinstance(self.issued_at_ms, int)
            or isinstance(self.issued_at_ms, bool)
            or not isinstance(self.expires_at_ms, int)
            or isinstance(self.expires_at_ms, bool)
            or self.issued_at_ms < 0
            or self.expires_at_ms - self.issued_at_ms != CHALLENGE_LIFETIME_MS
        ):
            raise VoiceAuthError("voice_auth.challenge_time_invalid")
        if not isinstance(self.server_nonce, bytes) or len(self.server_nonce) != NONCE_BYTES:
            raise VoiceAuthError("voice_auth.nonce_invalid")
        if not isinstance(self.server_mac, bytes) or len(self.server_mac) != MAC_BYTES:
            raise VoiceAuthError("voice_auth.authentication_failed")

    def _unsigned(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "expires_at_ms": self.expires_at_ms,
            "issued_at_ms": self.issued_at_ms,
            "key_version": self.key_version,
            "protocol": AUTH_PROTOCOL,
            "server_nonce": _encode_32(self.server_nonce),
            "type": _CHALLENGE_TYPE,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "server_mac": _encode_32(self.server_mac)}

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_dict())

    def to_json(self) -> str:
        return self.to_bytes().decode("ascii")


@dataclass(frozen=True, slots=True)
class ClientResponse:
    connection_id: str
    key_version: int
    server_nonce: bytes
    client_nonce: bytes
    hello_sha256: str
    client_mac: bytes

    def __post_init__(self) -> None:
        _uuid4(self.connection_id)
        _key_version(self.key_version)
        if not isinstance(self.server_nonce, bytes) or len(self.server_nonce) != NONCE_BYTES:
            raise VoiceAuthError("voice_auth.nonce_invalid")
        if not isinstance(self.client_nonce, bytes) or len(self.client_nonce) != NONCE_BYTES:
            raise VoiceAuthError("voice_auth.nonce_invalid")
        if not isinstance(self.hello_sha256, str) or _SHA256.fullmatch(self.hello_sha256) is None:
            raise VoiceAuthError("voice_auth.hello_digest_invalid")
        if not isinstance(self.client_mac, bytes) or len(self.client_mac) != MAC_BYTES:
            raise VoiceAuthError("voice_auth.authentication_failed")

    def _unsigned(self) -> dict[str, Any]:
        return {
            "client_nonce": _encode_32(self.client_nonce),
            "connection_id": self.connection_id,
            "hello_sha256": self.hello_sha256,
            "key_version": self.key_version,
            "protocol": AUTH_PROTOCOL,
            "server_nonce": _encode_32(self.server_nonce),
            "type": _RESPONSE_TYPE,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "client_mac": _encode_32(self.client_mac)}

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_dict())

    def to_json(self) -> str:
        return self.to_bytes().decode("ascii")


@dataclass(frozen=True, slots=True)
class ServerReady:
    connection_id: str
    key_version: int
    transcript_mac: bytes

    def __post_init__(self) -> None:
        _uuid4(self.connection_id)
        _key_version(self.key_version)
        if not isinstance(self.transcript_mac, bytes) or len(self.transcript_mac) != MAC_BYTES:
            raise VoiceAuthError("voice_auth.authentication_failed")

    def _unsigned(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "key_version": self.key_version,
            "protocol": AUTH_PROTOCOL,
            "type": _READY_TYPE,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "transcript_mac": _encode_32(self.transcript_mac)}

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_dict())

    def to_json(self) -> str:
        return self.to_bytes().decode("ascii")


def _validate_challenge_time(challenge: ServerChallenge, now_ms: int | None) -> None:
    now = _now_ms(now_ms)
    if (
        challenge.issued_at_ms < 0
        or challenge.expires_at_ms - challenge.issued_at_ms != CHALLENGE_LIFETIME_MS
    ):
        raise VoiceAuthError("voice_auth.challenge_time_invalid")
    if now < challenge.issued_at_ms:
        raise VoiceAuthError("voice_auth.challenge_not_yet_valid")
    if now > challenge.expires_at_ms:
        raise VoiceAuthError("voice_auth.challenge_expired")


def create_server_challenge(
    auth_key: VoiceAuthKey,
    *,
    connection_id: str | None = None,
    now_ms: int | None = None,
    server_nonce: bytes | None = None,
) -> ServerChallenge:
    issued = _now_ms(now_ms)
    selected_id = _uuid4(connection_id or str(uuid.uuid4()))
    nonce = secrets.token_bytes(NONCE_BYTES) if server_nonce is None else server_nonce
    if not isinstance(nonce, bytes) or len(nonce) != NONCE_BYTES:
        raise VoiceAuthError("voice_auth.nonce_invalid")
    unsigned = ServerChallenge(
        connection_id=selected_id,
        key_version=auth_key.key_version,
        issued_at_ms=issued,
        expires_at_ms=issued + CHALLENGE_LIFETIME_MS,
        server_nonce=nonce,
        server_mac=bytes(MAC_BYTES),
    )
    return ServerChallenge(
        connection_id=unsigned.connection_id,
        key_version=unsigned.key_version,
        issued_at_ms=unsigned.issued_at_ms,
        expires_at_ms=unsigned.expires_at_ms,
        server_nonce=unsigned.server_nonce,
        server_mac=_mac(auth_key.key, _CHALLENGE_DOMAIN, _canonical_json(unsigned._unsigned())),
    )


def parse_server_challenge(
    raw: str | bytes,
    auth_key: VoiceAuthKey,
    *,
    now_ms: int | None = None,
) -> ServerChallenge:
    value, _encoded = _json_object(raw)
    _exact(value, {
        "protocol", "type", "connection_id", "key_version", "issued_at_ms",
        "expires_at_ms", "server_nonce", "server_mac",
    })
    _protocol_and_type(value, _CHALLENGE_TYPE)
    issued = value["issued_at_ms"]
    expires = value["expires_at_ms"]
    if not isinstance(issued, int) or isinstance(issued, bool):
        raise VoiceAuthError("voice_auth.challenge_time_invalid")
    if not isinstance(expires, int) or isinstance(expires, bool):
        raise VoiceAuthError("voice_auth.challenge_time_invalid")
    challenge = ServerChallenge(
        connection_id=_uuid4(value["connection_id"]),
        key_version=_key_version(value["key_version"]),
        issued_at_ms=issued,
        expires_at_ms=expires,
        server_nonce=_decode_32(value["server_nonce"]),
        server_mac=_decode_32(value["server_mac"]),
    )
    if challenge.key_version != auth_key.key_version:
        raise VoiceAuthError("voice_auth.authentication_failed")
    expected = _mac(
        auth_key.key,
        _CHALLENGE_DOMAIN,
        _canonical_json(challenge._unsigned()),
    )
    if not hmac.compare_digest(challenge.server_mac, expected):
        raise VoiceAuthError("voice_auth.authentication_failed")
    _validate_challenge_time(challenge, now_ms)
    return challenge


def create_client_response(
    auth_key: VoiceAuthKey,
    challenge_frame: str | bytes,
    hello_frame: str | bytes,
    *,
    now_ms: int | None = None,
    client_nonce: bytes | None = None,
) -> ClientResponse:
    challenge_raw = _raw_bytes(challenge_frame, maximum=MAX_AUTH_FRAME_BYTES)
    challenge = parse_server_challenge(challenge_raw, auth_key, now_ms=now_ms)
    hello = _hello_frame(hello_frame)
    nonce = secrets.token_bytes(NONCE_BYTES) if client_nonce is None else client_nonce
    if not isinstance(nonce, bytes) or len(nonce) != NONCE_BYTES:
        raise VoiceAuthError("voice_auth.nonce_invalid")
    unsigned = ClientResponse(
        connection_id=challenge.connection_id,
        key_version=challenge.key_version,
        server_nonce=challenge.server_nonce,
        client_nonce=nonce,
        hello_sha256=hashlib.sha256(hello).hexdigest(),
        client_mac=bytes(MAC_BYTES),
    )
    message = _framed(challenge_raw) + _canonical_json(unsigned._unsigned())
    return ClientResponse(
        connection_id=unsigned.connection_id,
        key_version=unsigned.key_version,
        server_nonce=unsigned.server_nonce,
        client_nonce=unsigned.client_nonce,
        hello_sha256=unsigned.hello_sha256,
        client_mac=_mac(auth_key.key, _RESPONSE_DOMAIN, message),
    )


def parse_client_response(
    raw: str | bytes,
    auth_key: VoiceAuthKey,
    challenge_frame: str | bytes,
    *,
    now_ms: int | None = None,
) -> ClientResponse:
    challenge_raw = _raw_bytes(challenge_frame, maximum=MAX_AUTH_FRAME_BYTES)
    challenge = parse_server_challenge(challenge_raw, auth_key, now_ms=now_ms)
    value, _encoded = _json_object(raw)
    _exact(value, {
        "protocol", "type", "connection_id", "key_version", "server_nonce",
        "client_nonce", "hello_sha256", "client_mac",
    })
    _protocol_and_type(value, _RESPONSE_TYPE)
    hello_sha256 = value["hello_sha256"]
    if not isinstance(hello_sha256, str) or _SHA256.fullmatch(hello_sha256) is None:
        raise VoiceAuthError("voice_auth.hello_digest_invalid")
    response = ClientResponse(
        connection_id=_uuid4(value["connection_id"]),
        key_version=_key_version(value["key_version"]),
        server_nonce=_decode_32(value["server_nonce"]),
        client_nonce=_decode_32(value["client_nonce"]),
        hello_sha256=hello_sha256,
        client_mac=_decode_32(value["client_mac"]),
    )
    if (
        response.connection_id != challenge.connection_id
        or response.key_version != challenge.key_version
        or not hmac.compare_digest(response.server_nonce, challenge.server_nonce)
    ):
        raise VoiceAuthError("voice_auth.binding_invalid")
    message = _framed(challenge_raw) + _canonical_json(response._unsigned())
    expected = _mac(auth_key.key, _RESPONSE_DOMAIN, message)
    if not hmac.compare_digest(response.client_mac, expected):
        raise VoiceAuthError("voice_auth.authentication_failed")
    return response


def verify_hello_binding(response: ClientResponse, hello_frame: str | bytes) -> bytes:
    hello = _hello_frame(hello_frame)
    expected = hashlib.sha256(hello).hexdigest()
    if not hmac.compare_digest(response.hello_sha256, expected):
        raise VoiceAuthError("voice_auth.hello_mismatch")
    return hello


def _transcript_message(
    challenge_frame: bytes,
    response_frame: bytes,
    hello_frame: bytes,
    ready_unsigned: Mapping[str, Any],
) -> bytes:
    return (
        _framed(challenge_frame)
        + _framed(response_frame)
        + _framed(hello_frame)
        + _canonical_json(ready_unsigned)
    )


def create_server_ready(
    auth_key: VoiceAuthKey,
    challenge_frame: str | bytes,
    response_frame: str | bytes,
    hello_frame: str | bytes,
    *,
    now_ms: int | None = None,
) -> ServerReady:
    challenge_raw = _raw_bytes(challenge_frame, maximum=MAX_AUTH_FRAME_BYTES)
    response_raw = _raw_bytes(response_frame, maximum=MAX_AUTH_FRAME_BYTES)
    challenge = parse_server_challenge(challenge_raw, auth_key, now_ms=now_ms)
    response = parse_client_response(
        response_raw, auth_key, challenge_raw, now_ms=now_ms
    )
    hello = verify_hello_binding(response, hello_frame)
    unsigned = ServerReady(
        connection_id=challenge.connection_id,
        key_version=challenge.key_version,
        transcript_mac=bytes(MAC_BYTES),
    )
    transcript_mac = _mac(
        auth_key.key,
        _READY_DOMAIN,
        _transcript_message(
            challenge_raw, response_raw, hello, unsigned._unsigned()
        ),
    )
    return ServerReady(
        connection_id=unsigned.connection_id,
        key_version=unsigned.key_version,
        transcript_mac=transcript_mac,
    )


def parse_server_ready(
    raw: str | bytes,
    auth_key: VoiceAuthKey,
    challenge_frame: str | bytes,
    response_frame: str | bytes,
    hello_frame: str | bytes,
    *,
    now_ms: int | None = None,
) -> ServerReady:
    challenge_raw = _raw_bytes(challenge_frame, maximum=MAX_AUTH_FRAME_BYTES)
    response_raw = _raw_bytes(response_frame, maximum=MAX_AUTH_FRAME_BYTES)
    challenge = parse_server_challenge(challenge_raw, auth_key, now_ms=now_ms)
    response = parse_client_response(
        response_raw, auth_key, challenge_raw, now_ms=now_ms
    )
    hello = verify_hello_binding(response, hello_frame)
    value, _encoded = _json_object(raw)
    _exact(value, {"protocol", "type", "connection_id", "key_version", "transcript_mac"})
    _protocol_and_type(value, _READY_TYPE)
    ready = ServerReady(
        connection_id=_uuid4(value["connection_id"]),
        key_version=_key_version(value["key_version"]),
        transcript_mac=_decode_32(value["transcript_mac"]),
    )
    if ready.connection_id != challenge.connection_id or ready.key_version != challenge.key_version:
        raise VoiceAuthError("voice_auth.binding_invalid")
    expected = _mac(
        auth_key.key,
        _READY_DOMAIN,
        _transcript_message(challenge_raw, response_raw, hello, ready._unsigned()),
    )
    if not hmac.compare_digest(ready.transcript_mac, expected):
        raise VoiceAuthError("voice_auth.authentication_failed")
    return ready


def _key_path(path: str | os.PathLike[str]) -> Path:
    try:
        selected = Path(path)
    except (TypeError, ValueError):
        raise VoiceAuthError("voice_auth.key_path_invalid") from None
    if not selected.is_absolute():
        raise VoiceAuthError("voice_auth.key_path_invalid")
    return selected


def _validate_key_stat(value: os.stat_result) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise VoiceAuthError("voice_auth.key_file_invalid")
    if os.name == "posix" and (
        value.st_uid != os.getuid() or stat.S_IMODE(value.st_mode) != 0o600
    ):
        raise VoiceAuthError("voice_auth.key_permissions_invalid")


def _windows_key_sids() -> tuple[Any, Any, Any, Any]:
    try:
        import win32api  # type: ignore[import-not-found]
        import win32con  # type: ignore[import-not-found]
        import win32security  # type: ignore[import-not-found]

        token = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(), win32con.TOKEN_QUERY
        )
        user_sid = win32security.GetTokenInformation(
            token, win32security.TokenUser
        )[0]
        system_sid = win32security.CreateWellKnownSid(
            win32security.WinLocalSystemSid, None
        )
        administrators_sid = win32security.CreateWellKnownSid(
            win32security.WinBuiltinAdministratorsSid, None
        )
        return win32security, user_sid, system_sid, administrators_sid
    except Exception:
        raise VoiceAuthError("voice_auth.key_permissions_invalid") from None


def _validate_windows_key_permissions(path: Path) -> None:
    if os.name != "nt":
        return
    try:
        import win32con  # type: ignore[import-not-found]
        import win32file  # type: ignore[import-not-found]

        win32security, user_sid, system_sid, administrators_sid = _windows_key_sids()
        attributes = win32file.GetFileAttributes(str(path))
        if attributes & win32con.FILE_ATTRIBUTE_REPARSE_POINT:
            raise VoiceAuthError("voice_auth.key_file_invalid")
        security = win32security.GetNamedSecurityInfo(
            str(path),
            win32security.SE_FILE_OBJECT,
            win32security.OWNER_SECURITY_INFORMATION
            | win32security.DACL_SECURITY_INFORMATION,
        )
        owner = security.GetSecurityDescriptorOwner()
        if not win32security.EqualSid(owner, user_sid):
            raise VoiceAuthError("voice_auth.key_permissions_invalid")
        dacl = security.GetSecurityDescriptorDacl()
        if dacl is None:
            raise VoiceAuthError("voice_auth.key_permissions_invalid")
        allowed_sids = (user_sid, system_sid, administrators_sid)
        allowed_types = {
            win32security.ACCESS_ALLOWED_ACE_TYPE,
            getattr(win32security, "ACCESS_ALLOWED_OBJECT_ACE_TYPE", -1),
        }
        for index in range(dacl.GetAceCount()):
            ace = dacl.GetAce(index)
            if ace[0][0] not in allowed_types:
                continue
            sid = ace[-1]
            if not any(
                win32security.EqualSid(sid, allowed) for allowed in allowed_sids
            ):
                raise VoiceAuthError("voice_auth.key_permissions_invalid")
    except VoiceAuthError:
        raise
    except Exception:
        raise VoiceAuthError("voice_auth.key_permissions_invalid") from None


def _harden_windows_key_file(path: Path) -> None:
    if os.name != "nt":
        return
    try:
        import ntsecuritycon  # type: ignore[import-not-found]

        win32security, user_sid, system_sid, administrators_sid = _windows_key_sids()
        dacl = win32security.ACL()
        for sid in (user_sid, system_sid, administrators_sid):
            dacl.AddAccessAllowedAce(
                win32security.ACL_REVISION,
                ntsecuritycon.FILE_ALL_ACCESS,
                sid,
            )
        win32security.SetNamedSecurityInfo(
            str(path),
            win32security.SE_FILE_OBJECT,
            win32security.OWNER_SECURITY_INFORMATION
            | win32security.DACL_SECURITY_INFORMATION
            | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
            user_sid,
            None,
            dacl,
            None,
        )
        _validate_windows_key_permissions(path)
    except VoiceAuthError:
        raise
    except Exception:
        raise VoiceAuthError("voice_auth.key_permissions_invalid") from None


def _validate_key_parent_security(
    path: Path, *, platform_name: str | None = None
) -> None:
    selected_platform = os.name if platform_name is None else platform_name
    if selected_platform == "nt":
        # The stdlib-compatible creation path writes before the file DACL can
        # be hardened. A private parent closes that otherwise observable
        # creation window.
        _validate_windows_key_permissions(path)


def load_voice_auth_key(path: str | os.PathLike[str]) -> VoiceAuthKey:
    """Load one exact, non-symlink per-install key file without path leakage."""

    selected = _key_path(path)
    try:
        before = selected.lstat()
    except OSError:
        raise VoiceAuthError("voice_auth.key_unavailable") from None
    if stat.S_ISLNK(before.st_mode):
        raise VoiceAuthError("voice_auth.key_file_invalid")
    _validate_key_stat(before)
    _validate_windows_key_permissions(selected)
    if before.st_size < 1 or before.st_size > MAX_KEY_FILE_BYTES:
        raise VoiceAuthError("voice_auth.key_format_invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(selected, flags)
    except OSError:
        raise VoiceAuthError("voice_auth.key_unavailable") from None
    try:
        opened = os.fstat(descriptor)
        _validate_key_stat(opened)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise VoiceAuthError("voice_auth.key_file_invalid")
        chunks: list[bytes] = []
        remaining = MAX_KEY_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
    except VoiceAuthError:
        raise
    except OSError:
        raise VoiceAuthError("voice_auth.key_unavailable") from None
    finally:
        os.close(descriptor)
    if not encoded or len(encoded) > MAX_KEY_FILE_BYTES:
        raise VoiceAuthError("voice_auth.key_format_invalid")
    try:
        text = encoded.decode("utf-8", errors="strict")
        if text.startswith("\ufeff"):
            raise ValueError
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, _DuplicateKey):
        raise VoiceAuthError("voice_auth.key_format_invalid") from None
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "key_version", "key_b64"
    }:
        raise VoiceAuthError("voice_auth.key_format_invalid")
    if value.get("schema_version") != KEY_FILE_SCHEMA:
        raise VoiceAuthError("voice_auth.key_format_invalid")
    version = _key_version(value.get("key_version"), code="voice_auth.key_format_invalid")
    key = _decode_32(value.get("key_b64"), code="voice_auth.key_format_invalid")
    return VoiceAuthKey(version, key)


def create_voice_auth_key_file(
    path: str | os.PathLike[str],
    *,
    key_version: int = 1,
    key: bytes | None = None,
) -> VoiceAuthKey:
    """Exclusively create one mode-0600 key file; existing paths are untouched."""

    selected = _key_path(path)
    version = _key_version(key_version, code="voice_auth.key_format_invalid")
    secret = secrets.token_bytes(KEY_BYTES) if key is None else _key_bytes(key)
    document = {
        "key_b64": _encode_32(secret, code="voice_auth.key_format_invalid"),
        "key_version": version,
        "schema_version": KEY_FILE_SCHEMA,
    }
    encoded = _canonical_json(document) + b"\n"
    try:
        parent = selected.parent.lstat()
    except OSError:
        raise VoiceAuthError("voice_auth.key_path_invalid") from None
    if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
        raise VoiceAuthError("voice_auth.key_path_invalid")
    _validate_key_parent_security(selected.parent)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    created = False
    failure: VoiceAuthError | None = None
    try:
        descriptor = os.open(selected, flags, 0o600)
        created = True
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
        _validate_key_stat(os.fstat(descriptor))
    except VoiceAuthError as exc:
        failure = exc
    except OSError:
        failure = VoiceAuthError("voice_auth.key_create_failed")
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                failure = VoiceAuthError("voice_auth.key_create_failed")
    if failure is not None:
        if created:
            try:
                selected.unlink()
            except OSError:
                pass
        raise failure from None
    try:
        _harden_windows_key_file(selected)
    except VoiceAuthError:
        selected.unlink(missing_ok=True)
        raise
    return VoiceAuthKey(version, secret)


__all__ = [
    "AUTH_PROTOCOL",
    "CHALLENGE_LIFETIME_MS",
    "ClientResponse",
    "KEY_FILE_SCHEMA",
    "MAX_AUTH_FRAME_BYTES",
    "MAX_HELLO_FRAME_BYTES",
    "ServerChallenge",
    "ServerReady",
    "VoiceAuthError",
    "VoiceAuthKey",
    "create_client_response",
    "create_server_challenge",
    "create_server_ready",
    "create_voice_auth_key_file",
    "load_voice_auth_key",
    "parse_client_response",
    "parse_server_challenge",
    "parse_server_ready",
    "verify_hello_binding",
]
