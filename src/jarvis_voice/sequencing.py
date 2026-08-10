"""Per-connection ordering and duplicate detection."""

from __future__ import annotations

from enum import Enum

from .errors import ProtocolError
from .protocol import Envelope


class SequenceResult(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"


class SequenceTracker:
    def __init__(self) -> None:
        self._next_seq = 1
        self._seen: dict[int, str] = {}

    @property
    def next_seq(self) -> int:
        return self._next_seq

    def observe(self, envelope: Envelope) -> SequenceResult:
        if envelope.seq == self._next_seq:
            self._seen[envelope.seq] = envelope.fingerprint()
            self._next_seq += 1
            return SequenceResult.ACCEPTED
        if envelope.seq < self._next_seq:
            if self._seen.get(envelope.seq) == envelope.fingerprint():
                return SequenceResult.DUPLICATE
            raise ProtocolError("protocol.conflicting_duplicate", "Sequence number was reused with different content", safe_context={"seq": envelope.seq})
        raise ProtocolError("protocol.sequence_gap", "Sequence number is ahead of the expected value", safe_context={"expected_seq": self._next_seq, "received_seq": envelope.seq})

