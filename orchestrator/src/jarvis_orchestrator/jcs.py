"""Frozen CYRUS canonical JSON profile for durable artifacts.

The profile uses ASCII object keys, NFC strings and integer-only numbers.
Its NFC normalization and full integer encoding are persisted-hash behavior,
not a general RFC 8785 implementation. Monetary decimals remain strings.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


class CanonicalJsonError(ValueError):
    """The value cannot be represented by the frozen JARVIS JCS subset."""


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJsonError("non-finite JSON number")
        raise CanonicalJsonError("binary floating point is forbidden")
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFC", value)
        return normalized
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.isascii():
                raise CanonicalJsonError("object keys must be ASCII strings")
            if key in result:
                raise CanonicalJsonError("duplicate object key")
            result[key] = _normalize(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_normalize(item) for item in value]
    raise CanonicalJsonError(f"unsupported canonical JSON type: {type(value).__name__}")


def jcs_bytes(value: Any) -> bytes:
    try:
        normalized = _normalize(value)
        return json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except CanonicalJsonError:
        raise
    except (ValueError, RecursionError) as exc:
        raise CanonicalJsonError("canonical JSON value is invalid") from exc


def jcs_text(value: Any) -> str:
    return jcs_bytes(value).decode("utf-8")


def domain_sha256(schema_id: str, value: Any) -> str:
    if not isinstance(schema_id, str) or not schema_id or not schema_id.isascii() or "\x00" in schema_id:
        raise CanonicalJsonError("schema_id must be non-empty ASCII without NUL")
    prefix = b"JARVIS\x00" + schema_id.encode("ascii") + b"\x00"
    return hashlib.sha256(prefix + jcs_bytes(value)).hexdigest()
