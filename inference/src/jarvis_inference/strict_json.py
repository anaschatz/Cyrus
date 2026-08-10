from __future__ import annotations

import json
import math
from decimal import Decimal
from typing import Any

from .errors import ApplianceError
from .limits import JSON_DEPTH

_MAX_SAFE_INTEGER = 2**53 - 1


class DuplicateKey(ValueError):
    pass


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKey
        result[key] = value
    return result


def _nonfinite(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _integer(text: str) -> int:
    value = int(text, 10)
    if abs(value) > _MAX_SAFE_INTEGER:
        raise ValueError("integer is outside the supported I-JSON domain")
    return value


def _decimal(text: str) -> Decimal:
    value = Decimal(text)
    if not value.is_finite() or not math.isfinite(float(value)):
        raise ValueError("number is outside the supported binary64 domain")
    return value


def _depth(value: Any, current: int = 1) -> int:
    if current > JSON_DEPTH:
        raise ValueError("JSON nesting limit exceeded")
    if isinstance(value, dict):
        for key, child in value.items():
            key.encode("utf-8")
            _depth(child, current + 1)
    elif isinstance(value, list):
        for child in value:
            _depth(child, current + 1)
    elif isinstance(value, str):
        value.encode("utf-8")
    return current


def parse_text(text: str, *, nested: bool = False) -> Any:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_nonfinite,
            parse_int=_integer,
            parse_float=_decimal,
        )
        _depth(value)
        return value
    except DuplicateKey as exc:
        raise ApplianceError(400, "duplicate_json_key", "JSON object keys must be unique.") from exc
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        code = "invalid_tool_arguments" if nested else "invalid_json"
        raise ApplianceError(400, code, "The JSON payload is invalid.") from exc


def parse_body(raw: bytes) -> dict[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ApplianceError(400, "invalid_json_encoding", "The request body must be strict UTF-8 without a BOM.")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ApplianceError(400, "invalid_json_encoding", "The request body must be strict UTF-8 without a BOM.") from exc
    value = parse_text(text)
    if not isinstance(value, dict):
        raise ApplianceError(400, "invalid_json", "The top-level JSON value must be an object.")
    return value


def _utf16_sort_key(value: str) -> bytes:
    """RFC 8785 sorts object names by their raw UTF-16 code units."""
    try:
        return value.encode("utf-16-be")
    except UnicodeEncodeError as exc:
        raise ValueError("lone surrogate is outside the supported JCS domain") from exc


def _string(value: str) -> str:
    # ensure_ascii=False gives the RFC 8785 escaping rules for valid Unicode:
    # control characters, backslash and quote are escaped; other code points
    # are emitted verbatim.  Encoding below also rejects lone surrogates.
    value.encode("utf-8")
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _number(value: int | float | Decimal) -> str:
    if isinstance(value, bool):
        raise TypeError("boolean is not a JSON number")
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise ValueError("integer is outside the exact IEEE-754 JCS domain")
        return str(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite JSON number")
        value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if value == 0:
        return "0"

    # CPython's repr implements shortest round-trip binary64 formatting.  JCS
    # uses the same shortest representation but ECMAScript switches between
    # decimal and exponent notation at 1e-6 and 1e21.
    rendered = repr(value).lower()
    absolute = abs(value)
    if 1e-6 <= absolute < 1e21:
        if "e" in rendered:
            rendered = format(Decimal(rendered), "f")
        elif rendered.endswith(".0"):
            rendered = rendered[:-2]
        return rendered

    if "e" not in rendered:
        rendered = format(value, ".17e")
    mantissa, exponent_text = rendered.split("e", 1)
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = int(exponent_text)
    return f"{mantissa}e{'+' if exponent >= 0 else ''}{exponent}"


def _canonical(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float, Decimal)):
        return _number(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, list):
        return "[" + ",".join(_canonical(item) for item in value) + "]"
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object names must be strings")
        keys = sorted(value, key=_utf16_sort_key)
        return "{" + ",".join(_string(key) + ":" + _canonical(value[key]) for key in keys) + "}"
    raise TypeError("value is outside the supported JSON domain")


def canonical_bytes(value: Any) -> bytes:
    """RFC 8785 canonical JSON for the appliance's I-JSON/binary64 domain."""
    return _canonical(value).encode("utf-8")


def canonical_text(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")
