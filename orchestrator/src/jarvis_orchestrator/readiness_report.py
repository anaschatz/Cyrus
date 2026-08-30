"""Atomic persistence for content-free readiness reports."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .operations import OperationalError
from .readiness_evidence import (
    ReadinessEvidenceV2,
    ReadinessEvidenceVerifier,
    ReadinessIdentityV2,
    write_readiness_evidence,
)

MAXIMUM_EVIDENCE_BYTES = 1024 * 1024


def write_readiness_report(path: str | Path, report: dict[str, Any]) -> Path:
    """Persist a legacy operational report that is never gate-eligible."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAXIMUM_EVIDENCE_BYTES:
        raise OperationalError("readiness_report_too_large")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_signed_readiness_report(
    path: str | Path,
    evidence: ReadinessEvidenceV2,
    *,
    verifier: ReadinessEvidenceVerifier,
    cohort_state_directory: str | Path,
    current_state_provider: (
        Callable[[], tuple[ReadinessIdentityV2, Any]] | None
    ) = None,
    final_time_provider: Callable[[], datetime] | None = None,
) -> Path:
    """Persist only a structurally closed envelope trusted by the verifier."""

    return write_readiness_evidence(
        path,
        evidence,
        verifier=verifier,
        cohort_state_directory=cohort_state_directory,
        current_state_provider=current_state_provider,
        final_time_provider=final_time_provider,
    )


__all__ = ["write_readiness_report", "write_signed_readiness_report"]
