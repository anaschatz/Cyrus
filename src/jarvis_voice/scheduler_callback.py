"""Authenticated, replay-safe ingress for native scheduler notification actions."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping


CALLBACK_CONTRACT = "jarvis.scheduler-native-callback/1"
MAX_CALLBACK_BYTES = 4096
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _utc(value: Any) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("callback.invalid_timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("callback.invalid_timestamp") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("callback.invalid_timestamp")
    return value


@dataclass(frozen=True, slots=True)
class NativeSchedulerCallback:
    callback_id: str
    notification_id: str
    action: str
    requested_due_at_utc: str | None
    occurred_at_utc: str

    @classmethod
    def verify(cls, raw: bytes | str, *, key: bytes) -> "NativeSchedulerCallback":
        if len(key) != 32:
            raise ValueError("callback.invalid_key")
        encoded = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        if not encoded or len(encoded) > MAX_CALLBACK_BYTES or encoded.startswith(b"\xef\xbb\xbf"):
            raise ValueError("callback.invalid_frame")
        try:
            value = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("callback.invalid_frame") from exc
        fields = {"contract", "callback_id", "notification_id", "action",
                  "requested_due_at_utc", "occurred_at_utc", "mac"}
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("callback.invalid_fields")
        mac = value.pop("mac")
        if not isinstance(mac, str) or not re.fullmatch(r"[0-9a-f]{64}", mac):
            raise ValueError("callback.invalid_mac")
        expected = hmac.new(key, _canonical(value), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, expected):
            raise ValueError("callback.authentication_failed")
        if value["contract"] != CALLBACK_CONTRACT:
            raise ValueError("callback.invalid_contract")
        try:
            callback_id = str(uuid.UUID(value["callback_id"], version=4))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("callback.invalid_id") from exc
        if callback_id != value["callback_id"]:
            raise ValueError("callback.invalid_id")
        notification_id = value["notification_id"]
        if not isinstance(notification_id, str) or _SHA256.fullmatch(notification_id) is None:
            raise ValueError("callback.invalid_notification")
        action = value["action"]
        requested = value["requested_due_at_utc"]
        if action == "SNOOZE":
            requested = _utc(requested)
        elif action == "CANCEL":
            if requested is not None:
                raise ValueError("callback.invalid_due_time")
        else:
            raise ValueError("callback.invalid_action")
        return cls(callback_id, notification_id, action, requested, _utc(value["occurred_at_utc"]))

    def payload(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "callback_id": self.callback_id,
            "notification_id": self.notification_id,
            "occurred_at_utc": self.occurred_at_utc,
            "requested_due_at_utc": self.requested_due_at_utc,
        }


def sign_callback(payload: Mapping[str, Any], *, key: bytes) -> bytes:
    """Qualification/test helper; production signing belongs to the native helper."""

    core = {"contract": CALLBACK_CONTRACT, **dict(payload)}
    core["mac"] = hmac.new(key, _canonical(core), hashlib.sha256).hexdigest()
    return _canonical(core)
