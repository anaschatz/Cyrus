"""One device-global F10/F11 presentation controller shared by extensions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .errors import ProtocolError


class LocalApprovalState(StrEnum):
    STAGED = "STAGED"
    ARMED = "ARMED"
    DISARMED_CAPTURE = "DISARMED_CAPTURE"
    DECISION_PENDING = "DECISION_PENDING"
    CLOSED = "CLOSED"


@dataclass(slots=True)
class LocalPresentation:
    protocol: str
    approval_id: str
    presentation_id: str
    requested_event_id: str
    state: LocalApprovalState
    arm_counter: int


class GlobalApprovalController:
    def __init__(self, journal: object | None = None) -> None:
        self.journal = journal
        self.presentation: LocalPresentation | None = None
        self._awaiting_both_keys_up = False
        if journal is not None:
            row = journal.load_arbiter_presentation()
            if row is not None:
                state = LocalApprovalState(row["state"])
                if state is LocalApprovalState.ARMED:
                    state = LocalApprovalState.STAGED
                self.presentation = LocalPresentation(
                    row["protocol"], row["approval_id"], row["presentation_id"],
                    row["requested_event_id"], state, int(row["arm_counter"]),
                )
                self._persist()

    def _persist(self) -> None:
        if self.journal is None or self.presentation is None:
            return
        value = self.presentation
        self.journal.store_arbiter_presentation(
            protocol=value.protocol, approval_id=value.approval_id,
            presentation_id=value.presentation_id, requested_event_id=value.requested_event_id,
            state=value.state.value, arm_counter=value.arm_counter,
        )

    def stage(self, *, protocol: str, approval_id: str, presentation_id: str,
              requested_event_id: str, key_counter: int, both_keys_up: bool) -> bool:
        current = self.presentation
        if current is not None and current.state is not LocalApprovalState.CLOSED:
            if (current.protocol, current.approval_id, current.presentation_id, current.requested_event_id) == (
                protocol, approval_id, presentation_id, requested_event_id
            ):
                return current.state is LocalApprovalState.ARMED
            raise ProtocolError("protocol.approval_arbiter_violation",
                                "a second local approval presentation was requested")
        state = LocalApprovalState.ARMED if both_keys_up else LocalApprovalState.STAGED
        self.presentation = LocalPresentation(protocol, approval_id, presentation_id,
                                              requested_event_id, state, key_counter)
        self._awaiting_both_keys_up = not both_keys_up
        self._persist()
        return state is LocalApprovalState.ARMED

    def disarm_for_capture(self) -> None:
        current = self.presentation
        if current is not None and current.state in {LocalApprovalState.STAGED, LocalApprovalState.ARMED}:
            current.state = LocalApprovalState.DISARMED_CAPTURE
            self._awaiting_both_keys_up = True
            self._persist()

    def capture_ended(self, *, key_counter: int, both_keys_up: bool) -> bool:
        current = self.presentation
        if current is None or current.state not in {
            LocalApprovalState.DISARMED_CAPTURE, LocalApprovalState.STAGED
        } or not both_keys_up:
            return False
        current.state = LocalApprovalState.ARMED
        current.arm_counter = key_counter
        self._awaiting_both_keys_up = False
        self._persist()
        return True

    def begin_decision(self, *, protocol: str, key_counter: int, capture_idle: bool,
                       autorepeat: bool) -> bool:
        current = self.presentation
        if (current is None or current.protocol != protocol or current.state is not LocalApprovalState.ARMED
                or not capture_idle or autorepeat or key_counter <= current.arm_counter):
            return False
        current.state = LocalApprovalState.DECISION_PENDING
        self._persist()
        return True

    def close(self, *, protocol: str, approval_id: str) -> None:
        current = self.presentation
        if current is not None and current.protocol == protocol and current.approval_id == approval_id:
            current.state = LocalApprovalState.CLOSED
            if self.journal is not None:
                self.journal.close_arbiter_presentation()
            self.presentation = None
