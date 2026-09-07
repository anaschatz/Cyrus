from __future__ import annotations

import socket
import time
import urllib.error
import urllib.request
from typing import Any

from .engine import EngineProtocolFailure, EngineTimeout
from .limits import BACKEND_EVENT_BYTES
from .errors import ApplianceError
from .strict_json import canonical_bytes, parse_text


class BackendHTTP:
    def __init__(self, port: int, secret: str):
        self.base = f"http://127.0.0.1:{port}"
        self.secret = secret

    def request(self, method: str, path: str, value: Any | None, deadline: float) -> Any:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EngineTimeout("backend deadline expired")
        body = None if value is None else canonical_bytes(value)
        request = urllib.request.Request(
            self.base + path,
            data=body,
            method=method,
            headers={"Authorization": "Bearer " + self.secret, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=remaining) as response:
                raw = response.read(BACKEND_EVENT_BYTES + 1)
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise EngineTimeout("managed backend request failed") from exc
        if len(raw) > BACKEND_EVENT_BYTES:
            raise EngineProtocolFailure("oversized backend event")
        try:
            if raw.startswith(b"\xef\xbb\xbf"):
                raise ValueError
            return parse_text(raw.decode("utf-8", errors="strict"))
        except (ApplianceError, ValueError, RecursionError) as exc:
            raise EngineProtocolFailure("invalid backend JSON") from exc
