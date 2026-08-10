"""Deterministic in-memory adapters for Milestone 0."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping

from .backends import (
    AsrScore,
    LanguageMetadata,
    SpeechUnit,
    TranscriptResult,
)
from .protocol import Envelope, EventFactory, new_uuid4


class FakeClock:
    def __init__(self, initial: float = 0.0) -> None:
        self.value = initial

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("Cannot move a monotonic clock backwards")
        self.value += seconds


class FakeAsrBackend:
    def __init__(self, result: TranscriptResult | None = None) -> None:
        self.result = result or TranscriptResult(
            text="Άνοιξε το ημερολόγιο.",
            language=LanguageMetadata(code="el", score=0.97),
            asr_score=AsrScore(value=0.82, raw_avg_logprob=-0.20),
        )
        self.calls = 0

    def transcribe(self, audio: object) -> TranscriptResult:
        del audio
        self.calls += 1
        return self.result


class FakeTtsBackend:
    def __init__(self) -> None:
        self.pending: deque[SpeechUnit] = deque()
        self.active: SpeechUnit | None = None
        self.completed: list[SpeechUnit] = []
        self.cancelled_epochs: list[int] = []
        self.cancelled_through_epoch = 0
        self.ignored_stale: list[SpeechUnit] = []

    def enqueue(self, unit: SpeechUnit) -> None:
        if unit.epoch <= self.cancelled_through_epoch:
            self.ignored_stale.append(unit)
            return
        self.pending.append(unit)

    def start_next(self) -> SpeechUnit:
        if self.active is not None:
            raise RuntimeError("A speech unit is already active")
        if not self.pending:
            raise RuntimeError("No pending speech unit")
        self.active = self.pending.popleft()
        return self.active

    def complete_active(self) -> SpeechUnit:
        if self.active is None:
            raise RuntimeError("No active speech unit")
        completed = self.active
        self.completed.append(completed)
        self.active = None
        return completed

    def cancel(self, new_epoch: int, reason: str) -> None:
        del reason
        self.cancelled_epochs.append(new_epoch)
        self.cancelled_through_epoch = max(self.cancelled_through_epoch, new_epoch)
        self.pending.clear()
        self.active = None

    @property
    def is_idle(self) -> bool:
        return not self.pending and self.active is None


@dataclass(frozen=True, slots=True)
class ScriptedResponse:
    response_id: str
    events: tuple[Envelope, ...]


class FakeOrchestrator:
    """Scriptable server with deterministic ACK/deduplication behavior."""

    def __init__(self, session_id: str | None = None) -> None:
        self.factory = EventFactory(session_id=session_id)
        self.acknowledged_event_ids: set[str] = set()
        self.received: list[Envelope] = []
        self._ready: deque[Envelope] = deque()
        self._scheduled: list[tuple[int, Envelope]] = []

    def send(self, event: Envelope) -> None:
        """Transport-compatible send; generated replies become immediately readable."""

        self._ready.extend(self.accept(event))

    def receive(self) -> Envelope | None:
        return self._ready.popleft() if self._ready else None

    def queue_event(self, event: Envelope, *, delay_steps: int = 0) -> None:
        """Queue a server event now or after deterministic manual clock steps."""

        if delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        if delay_steps == 0:
            self._ready.append(event)
        else:
            self._scheduled.append((delay_steps, event))

    def advance(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        for _ in range(steps):
            remaining: list[tuple[int, Envelope]] = []
            for delay, event in self._scheduled:
                if delay <= 1:
                    self._ready.append(event)
                else:
                    remaining.append((delay - 1, event))
            self._scheduled = remaining

    def accept(self, event: Envelope) -> list[Envelope]:
        self.received.append(event)
        if event.type == "client.hello":
            return [self.factory.create("server.ready", {"protocol": "jarvis.voice/1"})]
        if event.type == "input.transcript.final":
            duplicate = event.event_id in self.acknowledged_event_ids
            self.acknowledged_event_ids.add(event.event_id)
            return [
                self.factory.create(
                    "server.ack",
                    {
                        "acked_event_id": event.event_id,
                        "status": "duplicate" if duplicate else "accepted",
                    },
                    turn_id=event.turn_id,
                )
            ]
        return []

    def script_response(
        self,
        sentences: Iterable[str],
        *,
        locale: str,
        turn_id: str | None = None,
        response_id: str | None = None,
        text_chunks: Iterable[str] | None = None,
        interleave_speech: bool = False,
    ) -> ScriptedResponse:
        response_id = response_id or new_uuid4()
        normalized_sentences = tuple(sentences)
        chunks = tuple(text_chunks) if text_chunks is not None else normalized_sentences
        events: list[Envelope] = [
            self.factory.create(
                "response.started",
                {"locale": locale},
                turn_id=turn_id,
                response_id=response_id,
            )
        ]
        committed = 0
        for chunk_index, chunk in enumerate(chunks):
            events.append(
                self.factory.create(
                    "response.text.delta",
                    {"chunk_index": chunk_index, "text": chunk},
                    turn_id=turn_id,
                    response_id=response_id,
                )
            )
            if interleave_speech and committed < len(normalized_sentences):
                committed += 1
                events.append(
                    self.factory.create(
                        "response.speech.commit",
                        {"unit_index": committed, "locale": locale, "text": normalized_sentences[committed - 1]},
                        turn_id=turn_id,
                        response_id=response_id,
                    )
                )
        events.append(
            self.factory.create(
                "response.text.done",
                {"chunk_count": len(chunks)},
                turn_id=turn_id,
                response_id=response_id,
            )
        )
        for unit_index, sentence in enumerate(normalized_sentences[committed:], start=committed + 1):
            events.append(
                self.factory.create(
                    "response.speech.commit",
                    {"unit_index": unit_index, "locale": locale, "text": sentence},
                    turn_id=turn_id,
                    response_id=response_id,
                )
            )
        events.append(
            self.factory.create(
                "response.speech.done",
                {"last_unit_index": len(normalized_sentences)},
                turn_id=turn_id,
                response_id=response_id,
            )
        )
        return ScriptedResponse(response_id=response_id, events=tuple(events))

    @staticmethod
    def exact_duplicate(event: Envelope) -> Envelope:
        return Envelope.from_dict(event.to_dict())

    @staticmethod
    def malformed(event: Envelope, *, remove_field: str) -> Mapping[str, object]:
        data = event.to_dict()
        data.pop(remove_field, None)
        return data
