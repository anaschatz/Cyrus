"""Transport adapters for the pre-``jarvis.voice/1`` authentication exchange."""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Protocol

from .voice_auth import (
    VoiceAuthError,
    VoiceAuthKey,
    create_client_response,
    create_server_challenge,
    create_server_ready,
    parse_client_response,
    parse_server_ready,
    verify_hello_binding,
)


AUTH_TIMEOUT_SECONDS = 2.0
AUTH_PROBE_PATH = "/v1/voice-auth-probe"
AUTH_PROBE_FRAME = '{"protocol":"jarvis.voice-auth-probe/1"}'
AUTH_PROBE_SUCCESS_FRAME = (
    '{"protocol":"jarvis.voice-auth-probe/1","status":"ready"}'
)


class SyncAuthConnection(Protocol):
    def send(self, data: str) -> None: ...
    def recv(self, timeout: float | None = None) -> str | bytes: ...


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise VoiceAuthError("voice_auth.timeout")
    return remaining


def _deadline(timeout_seconds: float) -> float:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise VoiceAuthError("voice_auth.timeout_invalid")
    try:
        deadline = time.monotonic() + timeout_seconds
    except OverflowError:
        raise VoiceAuthError("voice_auth.timeout_invalid") from None
    if timeout_seconds <= 0 or not math.isfinite(deadline):
        raise VoiceAuthError("voice_auth.timeout_invalid")
    return deadline


async def _auth_io(deadline: float, operation: Any, *args: Any) -> Any:
    remaining = _remaining(deadline)
    result = await asyncio.wait_for(operation(*args), timeout=remaining)
    _remaining(deadline)
    return result


def authenticate_sync_client(
    connection: SyncAuthConnection,
    auth_key: VoiceAuthKey,
    hello_frame: str,
    *,
    timeout_seconds: float = AUTH_TIMEOUT_SECONDS,
) -> None:
    """Authenticate both peers and transmit the exact first Voice hello frame."""

    deadline = _deadline(timeout_seconds)
    try:
        challenge_raw = connection.recv(timeout=_remaining(deadline))
        response = create_client_response(auth_key, challenge_raw, hello_frame)
        response_raw = response.to_json()
        _remaining(deadline)
        connection.send(response_raw)
        _remaining(deadline)
        connection.send(hello_frame)
        ready_raw = connection.recv(timeout=_remaining(deadline))
        parse_server_ready(
            ready_raw,
            auth_key,
            challenge_raw,
            response_raw,
            hello_frame,
        )
        _remaining(deadline)
    except VoiceAuthError:
        raise
    except TimeoutError:
        raise VoiceAuthError("voice_auth.timeout") from None
    except Exception:
        raise VoiceAuthError("voice_auth.transport_failed") from None


async def authenticate_async_client(
    websocket: Any,
    auth_key: VoiceAuthKey,
    hello_frame: str,
    *,
    timeout_seconds: float = AUTH_TIMEOUT_SECONDS,
) -> None:
    """Async equivalent used by operational probes and integration tests."""

    deadline = _deadline(timeout_seconds)
    try:
        challenge_raw = await _auth_io(deadline, websocket.recv)
        response = create_client_response(auth_key, challenge_raw, hello_frame)
        response_raw = response.to_json()
        await _auth_io(deadline, websocket.send, response_raw)
        await _auth_io(deadline, websocket.send, hello_frame)
        ready_raw = await _auth_io(deadline, websocket.recv)
        parse_server_ready(
            ready_raw,
            auth_key,
            challenge_raw,
            response_raw,
            hello_frame,
        )
        _remaining(deadline)
    except VoiceAuthError:
        raise
    except TimeoutError:
        raise VoiceAuthError("voice_auth.timeout") from None
    except Exception:
        raise VoiceAuthError("voice_auth.transport_failed") from None


async def authenticate_async_server(
    websocket: Any,
    auth_key: VoiceAuthKey,
    *,
    connection_id: str,
    timeout_seconds: float = AUTH_TIMEOUT_SECONDS,
) -> str | bytes:
    """Authenticate one connection and return its bound raw Voice hello frame."""

    deadline = _deadline(timeout_seconds)
    try:
        challenge = create_server_challenge(
            auth_key,
            connection_id=connection_id,
        )
        challenge_raw = challenge.to_json()
        await _auth_io(deadline, websocket.send, challenge_raw)
        response_raw = await _auth_io(deadline, websocket.recv)
        response = parse_client_response(response_raw, auth_key, challenge_raw)
        hello_raw = await _auth_io(deadline, websocket.recv)
        verify_hello_binding(response, hello_raw)
        ready = create_server_ready(
            auth_key,
            challenge_raw,
            response_raw,
            hello_raw,
        )
        await _auth_io(deadline, websocket.send, ready.to_json())
        return hello_raw
    except VoiceAuthError:
        raise
    except TimeoutError:
        raise VoiceAuthError("voice_auth.timeout") from None
    except Exception:
        raise VoiceAuthError("voice_auth.transport_failed") from None


__all__ = [
    "AUTH_PROBE_FRAME",
    "AUTH_PROBE_PATH",
    "AUTH_PROBE_SUCCESS_FRAME",
    "AUTH_TIMEOUT_SECONDS",
    "authenticate_async_client",
    "authenticate_async_server",
    "authenticate_sync_client",
]
