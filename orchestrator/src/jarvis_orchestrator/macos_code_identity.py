"""Canonical macOS designated-requirement identity helpers."""

from __future__ import annotations

import hashlib


def canonical_designated_requirement(raw: bytes) -> bytes:
    """Remove path-bearing codesign diagnostics from the stable identity."""

    try:
        lines = raw.decode("utf-8", errors="strict").replace("\r\n", "\n").split("\n")
    except UnicodeDecodeError as exc:
        raise ValueError("code signing requirement is not UTF-8") from exc
    selected = [line.strip() for line in lines if line.strip().startswith("designated =>")]
    if len(selected) != 1 or len(selected[0].encode("utf-8")) > 8192:
        raise ValueError("designated requirement is missing or ambiguous")
    return selected[0].encode("utf-8")


def code_identity_sha256(raw: bytes) -> str:
    return hashlib.sha256(canonical_designated_requirement(raw)).hexdigest()

