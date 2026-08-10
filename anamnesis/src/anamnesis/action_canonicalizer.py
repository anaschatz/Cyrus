"""Conservative, source-grounded canonicalization of immediate-action payloads."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import ConfigDict

from anamnesis.schema import Decision, ObservableEvent, ProposedAction, StrictModel

ACTION_CANONICALIZER_VERSION = "immediate-action-canonicalizer.v1"
_ADDRESS = re.compile(
    r"^\d+[A-Za-z]?(?:[- ]\d+)?\s+.+\s"
    r"(?:avenue|boulevard|crescent|drive|lane|quay|road|street)$",
    re.IGNORECASE,
)
_UPLOAD_REPORT = re.compile(
    r"^act now:\s*upload (?:the )?(?:latest )?report for "
    r"(?:my |the )?(.+?) study[.!]?$",
    re.IGNORECASE,
)


class CanonicalizationChange(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["move_slot", "drop_redundant", "rewrite_subject"]
    source: str
    target: str | None = None


class CanonicalizationResult(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Decision
    changes: tuple[CanonicalizationChange, ...] = ()


def canonicalize_immediate_decision(
    *,
    event: ObservableEvent,
    retrospective_recall: tuple[str, ...] | None,
    decision: Decision,
) -> CanonicalizationResult:
    """Normalize only values already grounded in the event or retrieved text."""

    if len(decision.actions) != 1:
        return CanonicalizationResult(decision=decision)
    action = decision.actions[0]
    if action.action_key != event.id or tuple(action.evidence_event_ids) != (event.id,):
        return CanonicalizationResult(decision=decision)

    sources = "\n".join((event.text, *(retrospective_recall or ())))
    source_folded = sources.casefold()
    payload = dict(action.payload)
    changes: list[CanonicalizationChange] = []

    # An address-like, source-grounded value has one canonical slot regardless
    # of which optional string field the model selected.
    for key, value in tuple(payload.items()):
        if key in {"subject", "address"} or not isinstance(value, str):
            continue
        if (
            _ADDRESS.fullmatch(value)
            and value.casefold() in source_folded
            and "address" not in payload
        ):
            payload["address"] = value
            del payload[key]
            changes.append(
                CanonicalizationChange(kind="move_slot", source=key, target="address")
            )

    subject = str(payload["subject"])
    subject_folded = subject.casefold()
    # Drop optional values that merely duplicate the canonical direct object,
    # or ungrounded possessive filler. Never drop a distinct sourced argument.
    for key, value in tuple(payload.items()):
        if key == "subject" or not isinstance(value, str):
            continue
        normalized = value.strip().casefold()
        if normalized in {"my", "the", "usual", "regular"} or (
            normalized
            and normalized in subject_folded
            and key in {"item", "shipment", "project"}
        ):
            del payload[key]
            changes.append(CanonicalizationChange(kind="drop_redundant", source=key))

    # Two common immediate-action shapes have an unambiguous canonical object
    # recoverable from the current event plus already-grounded typed slots.
    item = payload.get("item")
    if (
        subject == "track shipment"
        and isinstance(item, str)
        and item.casefold() in event.text.casefold()
    ):
        payload["subject"] = f"track {item.casefold()}"
        del payload["item"]
        changes.extend(
            (
                CanonicalizationChange(
                    kind="rewrite_subject", source="subject", target="subject"
                ),
                CanonicalizationChange(kind="drop_redundant", source="item"),
            )
        )
    match = _UPLOAD_REPORT.fullmatch(event.text.strip())
    if subject == "upload report" and match:
        topic = match.group(1).strip().casefold()
        payload["subject"] = f"upload {topic} report"
        changes.append(
            CanonicalizationChange(
                kind="rewrite_subject", source="subject", target="subject"
            )
        )

    normalized = ProposedAction(
        kind=action.kind,
        action_key=action.action_key,
        payload=payload,
        summary=action.summary,
        evidence_event_ids=action.evidence_event_ids,
    )
    return CanonicalizationResult(
        decision=Decision(actions=[normalized]), changes=tuple(changes)
    )


__all__ = [
    "ACTION_CANONICALIZER_VERSION",
    "CanonicalizationChange",
    "CanonicalizationResult",
    "canonicalize_immediate_decision",
]
